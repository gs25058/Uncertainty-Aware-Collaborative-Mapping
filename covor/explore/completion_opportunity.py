"""Intervene only when a complete observed-support route can gain body proof.

Otherwise use the EXACT unknown utility. Promises are hypothetical visibility
coverage, not decisions, and never change the map or the human graph.
"""
import heapq
import numpy as np
from .entry_objective import TaskContext
from .route_tasks import build_route_tasks
from .sensing import camera_pose


def missing_body_cells(g,node):
    q=tuple(g.xyz[node]);s=g.support_layer;z=g.node_height[node]
    if not s['support'][q]:return None
    contact=s.get('contact_hit_z_max',s['hit_z_max'])[q]
    bound=float(np.maximum(s['height'][q]+3*s['height_std'][q]+.005,-np.inf if g.lab[q]==1 else contact+.005))
    if not np.isfinite(bound) or bound>z+1e-9:return None
    i,j,k=q;top=int(np.ceil((z+g.body.h_crawl+.02)/g.res-1e-9))-g.ijk[2]
    low=max(k+1,int(np.floor((z+g.body.step)/g.res+1e-9))-g.ijk[2])
    if top>g.lab.shape[2]:return None
    required={(i,j,kk) for kk in range(k+1,top)}
    radius=g.body.w_side/2;r=int(np.ceil(radius/g.res+1))
    for di in range(-r,r+1):
        for dj in range(-r,r+1):
            x,y=i+di,j+dj
            centre=(np.array([x,y])+g.ijk[:2]+.5)*g.res
            distance=np.linalg.norm(np.maximum(abs(centre-g.node_xy[node])-g.res/2,0))
            if distance>radius+1e-9:continue
            if not (0<=x<g.lab.shape[0] and 0<=y<g.lab.shape[1]):return None
            for kk in range(low,top):
                if kk==low and g.lab[x,y,kk]==2 and s['support'][x,y,kk]:
                    upper=float(np.maximum(s['height'][x,y,kk]+3*s['height_std'][x,y,kk]+.005,s['hit_z_max'][x,y,kk]+.005))
                    if upper<=z+g.body.step+1e-9:continue
                required.add((x,y,kk))
    cells=np.array(sorted(required),int)
    labels=g.lab[tuple(cells.T)]
    if (labels==2).any():return None
    return cells[labels==0]


class CompletionOpportunity:
    def __init__(self,analysis,lab,layer,ijk,res,cfg,prior):
        self.g=g=analysis['graph'];self.prior=prior;self.cfg=cfg
        # Use the same bounded potential-route search, solely to find enquiry
        # nodes. No inference about an unseen floor can become a promise.
        context=TaskContext.__new__(TaskContext)
        context.opt=cfg['utility']['task_progress'];context.prior=prior
        context.decisions={d['goal']:d for d in analysis['decisions']};context.query=lambda *args:None
        build_route_tasks(context,analysis)
        nodes=set(n for path in context.value_paths.values() for n in path)
        for path in context.value_paths.values():
            for u,v in zip(path,path[1:]):
                for dest,_,corners in g._raw_neighbors(u):
                    if dest==v:
                        for choices in corners:nodes.update(choices)
                        break
        self.requirements={}
        budget=cfg['utility'].get('completion_max_nodes',1500)
        for n in sorted(nodes):
            if g.node_known[n] or len(self.requirements)>=budget:continue
            cells=missing_body_cells(g,n)
            if cells is not None and len(cells):self.requirements[n]=cells
        self.components=np.full(len(g.xyz),-1,int);component=0
        for seed in np.flatnonzero(g.node_known):
            if self.components[seed]>=0:continue
            stack=[int(seed)];self.components[seed]=component
            while stack:
                u=stack.pop()
                for v,_,_ in g.edges(u,True):
                    if self.components[v]<0:self.components[v]=component;stack.append(v)
            component+=1
        self.root={int(self.components[n]) for n in analysis['confirmed_search']['best']}
        self.targets={d['goal']:g.mission_nodes(goal) for d,goal in zip(analysis['decisions'],prior['goals']) if d['verdict']=='unverified'}
        clouds=list(self.requirements.values());self.cells=np.unique(np.concatenate(clouds),axis=0) if clouds else np.empty((0,3),int)
        mapping={tuple(q):i for i,q in enumerate(self.cells)}
        self.indices={n:np.array([mapping[tuple(q)] for q in cells]) for n,cells in self.requirements.items()}
        self.visibility=TaskContext.__new__(TaskContext);self.visibility.lab=lab;self.visibility.ijk=np.asarray(ijk);self.visibility.res=res;self.visibility.opt=cfg['utility']['task_progress']
        self.points=(self.cells+g.ijk+.5)*res
        self.stats=dict(repairable_body_nodes=len(self.requirements),queried_unknown_cells=len(self.cells),potential_routes=context.soft_route_stats,immutable_contact_failures_excluded=True)

    def target_values(self,probabilities):
        g=self.g;readiness={n:float(np.min(probabilities[ix])) for n,ix in self.indices.items()}
        readiness={n:v for n,v in readiness.items() if v>0}
        # Compress all already-confirmed paths; candidate changes affect only
        # finitely many enquiry nodes. Every diagonal corner remains checked.
        edges={};component_offset=len(g.xyz)
        def vertex(n):return component_offset+int(self.components[n]) if g.node_known[n] else int(n)
        for u,score in readiness.items():
            for v,_,corners in g._raw_neighbors(u):
                sv=1. if g.node_known[v] else readiness.get(v,0.)
                if sv<=0:continue
                margin=min(score,sv)
                for choices in corners:
                    margin=min(margin,max([1. if g.node_known[n] else readiness.get(n,0.) for n in choices]+[0.]))
                if margin<=0:continue
                a,b=vertex(u),vertex(v);edges.setdefault(a,[]).append((b,margin));edges.setdefault(b,[]).append((a,margin))
        best={component_offset+c:1. for c in self.root if c>=0};heap=[(-1.,n) for n in best];heapq.heapify(heap)
        while heap:
            negative,u=heapq.heappop(heap);score=-negative
            if score<best[u]:continue
            for v,coverage in edges.get(u,[]):
                candidate=min(score,coverage)
                if candidate>best.get(v,0.):best[v]=candidate;heapq.heappush(heap,(-candidate,v))
        return {goal:value for goal,nodes in self.targets.items() if (value:=max([best.get(vertex(n),0.) for n in nodes]+[0.]))>0}


def augment(scored,analysis,lab,layer,ijk,res,sd,cfg,prior):
    model=analysis.get('_completion_opportunity')
    if model is None:model=CompletionOpportunity(analysis,lab,layer,ijk,res,cfg,prior);analysis['_completion_opportunity']=model
    for item in scored:
        c=item['candidate'];views=[camera_pose(np.array(c['position']),c['yaw'],pitch,cfg['sensor']['baseline']) for pitch in cfg['candidates']['pitches']]
        probability=model.visibility.point_probabilities(model.points,views,sd) if len(model.points) else np.empty(0)
        item['targets']=model.target_values(probability)
        item['objective_diagnostics']=dict(version='completion_opportunity_v1',**model.stats,goals_with_complete_body_coverage=item['targets'],partial_progress_is_not_completion=True)
    return scored
