"""Planning-only progress towards human decisions, using current observations.

The coarse potential admits unknown space and does NOT assert human support,
connectivity, or reachability. It only weights where evidence would be useful.
Human/ReferenceValidator remain the sole decision/evaluation implementations.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from covor.entry.assignment import visible_fraction
from covor.occupancy import _dda_batch
from .sensing import camera_pose


@dataclass
class Query:
    key: str
    feet: np.ndarray
    body: np.ndarray
    foot_known: np.ndarray
    body_known: np.ndarray
    foot_novelty: np.ndarray
    goals: dict
    kind: str


class TaskContext:
    def __init__(self, analysis, lab, layer, ijk, res, cfg, prior):
        self.lab=lab; self.layer=layer; self.ijk=np.asarray(ijk); self.res=res
        self.opt=cfg['utility']['task_progress']; self.prior=prior
        self.human_body=analysis['graph'].body;self.sensor_cfg=cfg['sensor']
        self.factor=int(self.opt.get('coarse_rows',4)); f=self.factor
        shape=np.array(lab.shape); pad=(-shape)%f
        occupied=np.pad(lab==2,[(0,int(v)) for v in pad],constant_values=True)
        unknown=np.pad(lab==0,[(0,int(v)) for v in pad],constant_values=False)
        dims=occupied.shape; blocks=tuple(v//f for v in dims)
        reshape=(blocks[0],f,blocks[1],f,blocks[2],f)
        self.blocked=occupied.reshape(reshape).any(axis=(1,3,5))
        unknown_fraction=unknown.reshape(reshape).mean(axis=(1,3,5))
        self.shape=np.array(blocks); xyz=np.indices(blocks).reshape(3,-1).T
        world=(xyz*f+self.ijk+f/2)*res
        lo,hi=np.asarray(prior['exploration_box'])
        allowed=((world>=lo)&(world<=hi)).all(1).reshape(blocks)
        self.allowed=allowed&~self.blocked
        # Hypothetical body centres, never flight waypoints or confirmed feet.
        costs=1.+self.opt.get('unknown_travel_penalty',1.)*unknown_fraction
        ids=np.arange(np.prod(blocks)).reshape(blocks); row=[];col=[];values=[]
        for axis in range(3):
            a=[slice(None)]*3;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
            a=tuple(a);b=tuple(b);good=self.allowed[a]&self.allowed[b]
            u=ids[a][good];v=ids[b][good]
            c=(costs[a][good]+costs[b][good])*.5*f*res
            if axis==2:c*=self.opt.get('vertical_cost',2.)
            row.extend([u,v]);col.extend([v,u]);values.extend([c,c])
        self.network=sparse.csr_matrix((np.concatenate(values),(np.concatenate(row),np.concatenate(col))),shape=(ids.size,ids.size))
        self.ids=ids; self.world=world
        self.entry_distance=self.distance(prior['entry'])
        self.goal_distance={};self.best={};self.decisions={d['goal']:d for d in analysis['decisions']}
        for goal in prior['goals']:
            if self.decisions[goal['id']]['verdict']!='unverified':continue
            dest=self.region_ids(goal);self.goal_distance[goal['id']]=self.distance(goal)
            self.best[goal['id']]=float(np.min(self.entry_distance[dest])) if len(dest) else np.inf
        self.queries=[]
        self.boundary_queries(analysis)
        self.soft_route_stats={}
        if self.opt.get('soft_route_progress'):self.anchored_route_queries(analysis)
        for goal in prior['goals']:
            decision=self.decisions[goal['id']]
            if decision['verdict']=='blocked':continue
            if decision['verdict']=='passable':
                self.path_query(goal,decision)
            else:self.goal_query(goal)

    def region_ids(self, region):
        x0,y0,x1,y1=region['xy_bounds']; z0,z1=region['floor_z_bounds']
        # Public nominal floor interval, solely a planning query assumption.
        z=(z0+z1)/2+.45
        w=self.world;inside=(w[:,0]>=x0)&(w[:,0]<=x1)&(w[:,1]>=y0)&(w[:,1]<=y1)&(abs(w[:,2]-z)<=self.factor*self.res*.55)
        return np.flatnonzero(inside&self.allowed.ravel())

    def distance(self, region):
        nodes=self.region_ids(region)
        if not len(nodes):return np.full(self.ids.size,np.inf)
        return dijkstra(self.network,directed=False,indices=nodes,min_only=True)

    def index(self, point):
        q=np.floor((np.asarray(point)/self.res-self.ijk)/self.factor).astype(int)
        if ((q>=0)&(q<self.shape)).all():return int(self.ids[tuple(q)])
        return None

    def alignment(self, point):
        node=self.index(np.asarray(point)+[0,0,.45]);out={}
        if node is None:return out
        slack=self.opt.get('route_slack_m',2.)
        for goal,distance in self.goal_distance.items():
            a=self.entry_distance[node];b=distance[node];best=self.best[goal]
            if np.isfinite(a+b+best):out[goal]=float(np.exp(-max(0.,a+b-best)/slack))
            else:
                # Coarse max pooling can close a narrow real doorway. This
                # low-weight public-goal heuristic never certifies blockage.
                region=next(g for g in self.prior['goals'] if g['id']==goal)
                x0,y0,x1,y1=region['xy_bounds'];z0,z1=region['floor_z_bounds']
                centre=np.array([(x0+x1)/2,(y0+y1)/2,(z0+z1)/2])
                e=self.prior['entry'];e0,e1,e2,e3=e['xy_bounds'];entry=np.array([(e0+e2)/2,(e1+e3)/2,sum(e['floor_z_bounds'])/2])
                out[goal]=float(.2*np.exp(-max(0.,np.linalg.norm(point-centre)-np.linalg.norm(entry-centre))/slack))
            scale=self.opt.get('remaining_distance_scale_m')
            if scale:
                remaining=b if np.isfinite(b) else np.linalg.norm(point-centre)
                out[goal]*=float(scale/(scale+max(0.,remaining)))
        return out

    def query(self,key,feet,goals,kind,body_override=None):
        if not goals or not len(feet):return
        feet=np.asarray(feet,float);n=int(self.opt.get('query_samples',24))
        if len(feet)>n:feet=feet[np.linspace(0,len(feet)-1,n).astype(int)]
        # Crawl is the least permissive height REQUIRED to rule out all modes.
        # These points are enquiry samples; the actual full body test is intact.
        body=np.concatenate([feet+[0,0,z] for z in (.15,.45,.80)])
        if self.opt.get('body_query_mode')=='missing_footprint':
            # Ask about the WHOLE minimum human footprint, especially unknown
            # side/head voxels that a centreline enquiry would miss.
            cells=[];radius=self.human_body.w_side/2
            for point in feet:
                foot=point+[0,0,.023];centre=np.floor(foot/self.res).astype(int)-self.ijk
                rr=int(np.ceil(radius/self.res+1));low=int(np.floor((foot[2]+self.human_body.step)/self.res))-self.ijk[2]
                top=int(np.ceil((foot[2]+self.human_body.h_crawl+.02)/self.res))-self.ijk[2]
                for di in range(-rr,rr+1):
                    for dj in range(-rr,rr+1):
                        xy=centre[:2]+[di,dj];world=(xy+self.ijk[:2]+.5)*self.res
                        distance=np.linalg.norm(np.maximum(abs(world-foot[:2])-self.res*.5,0.))
                        if distance>radius+1e-9:continue
                        start=centre[2] if di==dj==0 else low
                        cells.extend((int(xy[0]),int(xy[1]),k) for k in range(max(0,start),min(self.lab.shape[2],top)))
            if cells:
                cells=np.unique(np.asarray(cells,int),axis=0);good=((cells>=0)&(cells<self.lab.shape)).all(1);cells=cells[good]
                missing=self.lab[tuple(cells.T)]==0
                if missing.any():cells=cells[missing]
                body=(cells+self.ijk+.5)*self.res
        if body_override is not None:body=np.asarray(body_override,float)
        if len(body)>n:body=body[np.linspace(0,len(body)-1,n).astype(int)]
        fi=np.floor(feet/self.res).astype(int)-self.ijk
        bi=np.floor(body/self.res).astype(int)-self.ijk
        valid=((fi>=0)&(fi<np.array(self.lab.shape))).all(1)
        feet=feet[valid];fi=fi[valid]
        bv=((bi>=0)&(bi<np.array(self.lab.shape))).all(1);body=body[bv];bi=bi[bv]
        if not len(feet) or not len(body):return
        known=np.zeros(len(fi),float);votes=np.zeros(len(fi),float)
        for delta in (-2,-1,0,1):
            q=fi.copy();q[:,2]+=delta;ok=((q>=0)&(q<np.array(self.lab.shape))).all(1);ix=np.flatnonzero(ok)
            h=self.layer['height'][tuple(q[ok].T)];near=np.isfinite(h)&(abs(h-feet[ok,2])<.15)
            ready=self.layer['support'][tuple(q[ok].T)].copy()
            if self.opt.get('contact_aware_foot_evidence'):
                upper=h+3*self.layer['height_std'][tuple(q[ok].T)]+.005
                hits=self.layer['hit_z_max'][tuple(q[ok].T)]+.005
                upper=np.maximum(upper,np.where(self.lab[tuple(q[ok].T)]==1,-np.inf,hits))
                ready&=upper<=feet[ok,2]+.023+1e-9
                near&=abs(h-feet[ok,2])<.03
            known[ix]=np.maximum(known[ix],near*ready)
            votes[ix]=np.maximum(votes[ix],near*self.layer['votes'][tuple(q[ok].T)])
        # Observed absence is useful evidence too, but never an invented floor.
        below=fi.copy();below[:,2]-=1;ok=(below[:,2]>=0)
        absent=np.zeros(len(fi),bool);absent[ok]=(self.lab[tuple(fi[ok].T)]==1)&(self.lab[tuple(below[ok].T)]==1)&(votes[ok]==0)
        known[absent]=1.
        bk=(self.lab[tuple(bi.T)]!=0).astype(float)
        novelty=np.where(known>0,0.,1/(1+votes/8.))
        self.queries.append(Query(key,feet,body,known,bk,novelty,goals,kind))

    def boundary_queries(self,analysis):
        g=analysis['graph'];offset=getattr(g,'map_offset',np.zeros(3,int))
        reach=np.zeros(g.lab.shape,bool)
        nodes=np.array(sorted(analysis['confirmed_search']['best']),int)
        if len(nodes):reach[tuple(g.xyz[nodes].T)]=True
        fringe=g.possible&~reach&ndimage.binary_dilation(reach,structure=np.ones((3,3,5),bool))
        points=np.argwhere(fringe)
        if not len(points):
            e=self.prior['entry'];x0,y0,x1,y1=e['xy_bounds'];z=sum(e['floor_z_bounds'])/2
            feet=np.array([[x,y,z] for x in np.linspace(x0,x1,4) for y in np.linspace(y0,y1,4)])
            self.query('entry_evidence',feet,{k:1. for k in self.goal_distance},'entry')
            return
        xyz=np.c_[g.node_xy[g.ids[tuple(points.T)]],g.foot_z[tuple(points.T)]-.025+.002]
        keys=np.floor(xyz/np.array([.8,.8,.3])).astype(int)
        patches=[]
        for key in np.unique(keys,axis=0):
            p=xyz[(keys==key).all(1)];align=self.alignment(np.median(p,axis=0))
            patches.append((sum(align.values()),tuple(key),p,align))
        for _,key,feet,align in sorted(patches,key=lambda p:(-p[0],p[1]))[:self.opt.get('max_boundary_patches',8)]:
            if self.opt.get('bridge_extension_boost'):
                centre=np.floor(np.median(feet,axis=0)/self.res).astype(int)-g.ijk
                rad=np.array([15,15,3]);lo=np.maximum(centre-rad,0);hi=np.minimum(centre+rad+1,g.lab.shape)
                sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));unreached=int((g.confirmed[sl]&~reach[sl]).sum())
                boost=1+min(2.,np.log1p(unreached/50.))
                align={k:v*boost for k,v in align.items()}
            self.query('boundary_'+str(key),feet,align,'boundary')

    def anchored_route_queries(self,analysis):
        """Value-only paths: keep observed support anchors/walls/body geometry,
        relax the consecutive-unknown budget, and charge for missing evidence.
        No graph/map/decision hypotheses are committed to the actual system.
        """
        g=analysis['graph'];starts=g.mission_nodes(self.prior['entry'],True) or g.mission_nodes(self.prior['entry'])
        targets={goal['id']:set(g.mission_nodes(goal)) for goal in self.prior['goals'] if self.decisions[goal['id']]['verdict']=='unverified'}
        targets={key:value for key,value in targets.items() if value};lookup={}
        for goal,nodes in targets.items():
            for node in nodes:lookup.setdefault(node,[]).append(goal)
        distance={int(u):0. for u in starts};parent={int(u):None for u in starts};deps={};heap=[(0.,int(u)) for u in starts];heapq.heapify(heap)
        found={};expanded=0;budget=self.opt.get('soft_path_max_expansions',50000);penalty=self.opt.get('unknown_connection_penalty',4.)
        while heap and expanded<budget and len(found)<len(targets):
            d,u=heapq.heappop(heap)
            if d>distance[u]:continue
            expanded+=1
            for goal in lookup.get(u,[]):found.setdefault(goal,u)
            for v,length,uncertain in g.edges(u,False):
                repair=1.
                if self.opt.get('contact_aware_foot_evidence') and uncertain:
                    q=tuple(g.xyz[v]);std=float(g.support_layer['height_std'][q]);votes=float(g.support_layer['votes'][q])
                    if std>.0066666667:repair+=min(10.,(std/.0066666667)**2*votes/24.)
                cost=length*max(g.node_factor[u],g.node_factor[v])*(1+penalty*bool(uncertain))*repair+g.body.c_rise*abs(g.node_height[v]-g.node_height[u])
                nd=d+cost
                if nd<distance.get(v,np.inf):distance[v]=nd;parent[v]=u;deps[v]=uncertain;heapq.heappush(heap,(nd,int(v)))
        requests={};extent=np.array([1.2,1.2,.3]);offset=np.array([0,0,.15]);sid_cache={}
        def region_nodes(sid):
            if sid not in sid_cache:
                nodes=np.flatnonzero(g.node_region==sid)
                xyz=np.c_[g.node_xy[nodes],g.node_height[nodes]-.023]
                key=tuple(np.floor((np.median(xyz,axis=0)+offset)/extent).astype(int))
                sid_cache[sid]=(key,xyz)
            return sid_cache[sid]
        path_regions={}
        for goal,tip in found.items():
            regions=set();node=tip
            while node is not None:
                regions.update(deps.get(node,()));sid=int(g.node_region[node])
                if sid:regions.add(sid)
                node=parent[node]
            grouped={}
            for sid in sorted(regions):
                key,xyz=region_nodes(sid);grouped.setdefault(key,[]).append(xyz)
            path_regions[goal]=len(grouped)
            for key,clouds in grouped.items():
                request=requests.setdefault(key,dict(clouds=[],goals={}))
                request['clouds'].extend(clouds);request['goals'][goal]=1/max(1,len(grouped))
        ordered=sorted(requests.items(),key=lambda kv:(-sum(kv[1]['goals'].values()),kv[0]))
        for key,request in ordered[:self.opt.get('max_anchored_queries',8)]:
            feet=np.unique(np.concatenate(request['clouds']),axis=0)
            self.query('anchored_'+str(key),feet,request['goals'],'anchored')
        self.soft_route_stats=dict(expanded_nodes=expanded,expansion_budget=budget,goals_with_observed_anchors=len(targets),goals_with_value_only_path=len(found),uncertain_sections_by_goal=path_regions,queried_sections=min(len(ordered),self.opt.get('max_anchored_queries',8)),classification_unchanged=True)

    def goal_query(self,goal):
        x0,y0,x1,y1=goal['xy_bounds'];z0,z1=goal['floor_z_bounds']
        # Mission bounds contain the nominal floor, but never internal shapes.
        z=(z0+z1)/2
        feet=np.array([[x,y,z] for x in np.linspace(x0,x1,5) for y in np.linspace(y0,y1,5)])
        for k,p in enumerate(feet):
            q=np.floor(p/self.res).astype(int)-self.ijk
            if ((q>=0)&(q<np.array(self.lab.shape))).all():
                heights=self.layer['height'][q[0],q[1]]
                ids=np.flatnonzero(np.isfinite(heights)&(heights>=z0)&(heights<=z1))
                if len(ids):feet[k,2]=heights[ids[np.argmin(abs(heights[ids]-z))]]+.002
        self.query('goal_'+goal['id'],feet,{goal['id']:1.},'goal')
        if self.opt.get('closure_weight',0):
            # Mission-region shell enquiries, not a claimed room/door model.
            # Ceiling/side uncertainty matters to the conservative 3D blocked
            # certificate even when the low interior body cells are known.
            shell=[]
            for height in (.45,1.1,1.8,2.5,3.):
                for x in np.linspace(x0,x1,6):shell.extend([[x,y0,z+height],[x,y1,z+height]])
                for y in np.linspace(y0,y1,6):shell.extend([[x0,y,z+height],[x1,y,z+height]])
            for x in np.linspace(x0,x1,4):
                for y in np.linspace(y0,y1,4):shell.append([x,y,z+3.])
            shell=np.asarray(shell);q=np.floor(shell/self.res).astype(int)-self.ijk;good=((q>=0)&(q<self.lab.shape)).all(1);shell=shell[good];q=q[good]
            missing=self.lab[tuple(q.T)]==0
            if missing.any():shell=shell[missing]
            self.query('closure_'+goal['id'],feet,{goal['id']:1.},'closure',body_override=shell)

    def path_query(self,goal,decision):
        if not self.opt.get('stability_weight',0):return
        feet=np.array([p['xyz'] for p in decision['path']]);feet[:,2]-=.023
        # Only existing route queries; no reference route/mesh is available.
        self.query('route_check_'+goal['id'],feet,{goal['id']:1.},'stability')

    def observation_value(self,query,views,sd):
        def visible(points,needed):
            if not np.any(needed):return 0.
            pts=points[needed];cells=np.floor(pts/self.res).astype(int)-self.ijk
            best=0.
            for T in views:
                v=visible_fraction(cells,self.lab,self.ijk,self.res,T,sd,points_override=pts,max_range_m=sd.cfg.z_max)
                best=max(best,v['fraction']-self.opt.get('unknown_occlusion_discount',.5)*v['through_unknown'])
            return best
        fk=query.foot_known.mean();bk=query.body_known.mean()
        fv=visible(query.feet,query.foot_known<1)
        bv=visible(query.body,query.body_known<1)
        fn=float(query.foot_novelty.mean())
        af=min(1.,fk+fv*fn);ab=min(1.,bk+bv*(1-bk))
        return max(0.,min(af,ab)-min(fk,bk)),dict(foot_before=float(fk),body_before=float(bk),foot_visible=fv,body_visible=bv,foot_after=float(af),body_after=float(ab),foot_novelty=fn)

    def foot_quality(self,points,T,sd):
        """Public-profile/sampling heuristic, NOT simulator true error/normal/T."""
        optical=((points-T[:3,3])@T[:3,:3])@sd.R1.T;z=np.maximum(optical[:,2],.01)
        ray=points-T[:3,3];length=np.maximum(np.linalg.norm(ray,axis=1),.01)
        incidence=abs(ray[:,2])/length;stride=self.opt.get('mapping_sample_stride',4)
        density=(sd.fx*self.res/(z*stride))*(sd.fy*self.res/(z*stride))*incidence
        coverage=-np.expm1(-density*self.opt.get('effective_scan_views',4.))
        scale=self.opt.get('foot_precision_scale_m')
        precision=np.ones(len(points))
        if scale:
            profile=self.sensor_cfg['error_model'];sigma=profile['sigma_constant_m']+profile['sigma_quadratic_per_m']*z*z
            height_sigma=sigma*abs(ray[:,2])/z
            precision=1/(1+(height_sigma/scale)**2)
        return np.maximum(.01,coverage*precision)

    def point_probabilities(self,points,views,sd,foot=False):
        out=np.zeros(len(points));discount=self.opt.get('unknown_occlusion_discount',.5)
        for T in views:
            optical=((points-T[:3,3])@T[:3,:3])@sd.R1.T;z=optical[:,2]
            with np.errstate(divide='ignore',invalid='ignore'):
                u=sd.fx*optical[:,0]/z+sd.cx;v=sd.fy*optical[:,1]/z+sd.cy
            w,h=sd.calib.size;valid=(z>=sd.cfg.z_min)&(z<=sd.cfg.z_max)&(u>=0)&(u<w)&(v>=0)&(v<h)
            ids=np.flatnonzero(valid)
            if not len(ids):continue
            ray,vox=_dda_batch(T[:3,3],points[ids],self.res);q=vox-self.ijk
            inside=((q>=0)&(q<self.lab.shape)).all(1);values=np.zeros(len(q),np.uint8);values[inside]=self.lab[tuple(q[inside].T)]
            wall=np.zeros(len(ids),bool);unknown=wall.copy();np.logical_or.at(wall,ray,values==2);np.logical_or.at(unknown,ray,values==0)
            probability=(~wall)*(1-discount*unknown)
            if foot and self.opt.get('foot_sampling_quality'):probability*=self.foot_quality(points[ids],T,sd)
            out[ids]=np.maximum(out[ids],probability)
        return out


def pooled_target_values(items,cfg):
    """Union cell evidence BEFORE the foot/body minimum, then max per goal.

    This retains complementary observations while avoiding counting the same
    surface or goal repeatedly. All predictions are from the current map.
    """
    opt=cfg['utility']['task_progress'];targets={};queries={}
    for item in items:
        for goal,value in item.get('task_channels',{}).get('completion',{}).items():
            targets[goal]=max(targets.get(goal,0.),opt.get('completion_weight',0.)*value*item.get('view_novelty',1.))
        for key,pred in item.get('task_query_predictions',{}).items():
            if key not in queries:queries[key]=dict(pred,foot=np.asarray(pred['foot']),body=np.asarray(pred['body']))
            else:
                queries[key]['foot']=np.maximum(queries[key]['foot'],pred['foot']);queries[key]['body']=np.maximum(queries[key]['body'],pred['body'])
    eta=opt.get('partial_evidence_weight',.15);anchored={}
    for pred in queries.values():
        before_f=pred['foot_before'];before_b=pred['body_before'];f=float(np.mean(pred['foot']));b=float(np.mean(pred['body']))
        value=max(0.,min(f,b)-min(before_f,before_b))+eta*max(0.,(f+b-before_f-before_b)/2)
        for goal,weight in pred['goals'].items():
            if pred['kind']=='anchored':anchored[goal]=anchored.get(goal,0.)+weight*min(1.,value)
            else:targets[goal]=max(targets.get(goal,0.),opt.get(pred['kind']+'_weight',0.)*weight*value)
    for goal,value in anchored.items():targets[goal]=max(targets.get(goal,0.),opt.get('anchored_weight',0.)*min(1.,value))
    return targets


def joint_marginal(item,allocated,cfg):
    before=pooled_target_values(allocated,cfg);after=pooled_target_values([*allocated,item],cfg)
    return sum(max(0.,v-before.get(g,0.)) for g,v in after.items())


def augment(scored,analysis,lab,layer,ijk,res,sd,cfg,prior):
    context=analysis.get('_task_context')
    if context is None:
        context=TaskContext(analysis,lab,layer,ijk,res,cfg,prior);analysis['_task_context']=context
    opt=cfg['utility']['task_progress']
    for item in scored:
        c=item['candidate'];views=[camera_pose(np.array(c['position']),c['yaw'],p,cfg['sensor']['baseline']) for p in cfg['candidates']['pitches']]
        channels={k:{} for k in ('completion','boundary','goal','stability','closure','anchored')};channels['completion']=item['targets'].copy();contributions=[]
        for query in context.queries:
            if opt.get('joint_evidence'):continue
            value,proof=context.observation_value(query,views,sd)
            kind='boundary' if query.kind=='entry' else query.kind
            for goal,weight in query.goals.items():channels[kind][goal]=max(channels[kind].get(goal,0.),value*weight)
            if value>0:contributions.append(dict(query=query.key,kind=query.kind,goals=query.goals,value=value,**proof))
        targets={}
        for kind,data in channels.items():
            weight=opt.get(kind+'_weight',0.)
            for goal,value in data.items():
                if opt.get('channel_aggregation','sum')=='max':targets[goal]=max(targets.get(goal,0.),weight*value)
                else:targets[goal]=targets.get(goal,0.)+weight*value
        novelty=1.;nearby=0
        history=analysis.get('_observed_views')
        radius=opt.get('view_novelty_radius_m',0.)
        if radius and history is not None and len(history):
            positions=np.asarray([v[0] for v in history]);headings=np.asarray([v[1] for v in history])
            distance=np.linalg.norm(positions-np.asarray(c['position']),axis=1)
            angle=abs((headings-c['yaw']+np.pi)%(2*np.pi)-np.pi)
            nearby=int(((distance<radius)&(angle<np.deg2rad(45))).sum())
            novelty=max(.05,1/(1+nearby/opt.get('view_novelty_scale',6.)))
        item['targets']={g:v*novelty for g,v in targets.items()};item['task_channels']=channels;item['view_novelty']=novelty
        item['objective_diagnostics']=dict(version=cfg['utility']['entry_objective'],queries=len(context.queries),hypothesis='coarse unknown-permitted task potential; not a confirmed human route',channels=channels,contributions=contributions,view_novelty=novelty,nearby_past_view_seconds=nearby,anchored_planning=getattr(context,'soft_route_stats',{}))
        if opt.get('joint_evidence'):
            predictions={}
            for query in context.queries:
                kind='boundary' if query.kind=='entry' else query.kind
                if not opt.get(kind+'_weight',0):continue
                fp=np.zeros(len(query.feet));bp=np.zeros(len(query.body));fm=query.foot_known<1;bm=query.body_known<1
                if fm.any():fp[fm]=context.point_probabilities(query.feet[fm],views,sd,foot=True)
                if bm.any():bp[bm]=context.point_probabilities(query.body[bm],views,sd)
                foot=query.foot_known+(1-query.foot_known)*fp*query.foot_novelty*novelty
                body=query.body_known+(1-query.body_known)*bp*novelty
                if not ((foot>query.foot_known+1e-8).any() or (body>query.body_known+1e-8).any()):continue
                predictions[query.key]=dict(kind=kind,goals=query.goals,foot=foot.tolist(),body=body.tolist(),foot_before=float(query.foot_known.mean()),body_before=float(query.body_known.mean()))
                contributions.append(dict(query=query.key,kind=kind,goals=query.goals,foot_before=float(query.foot_known.mean()),body_before=float(query.body_known.mean()),foot_after=float(foot.mean()),body_after=float(body.mean())))
            item['task_query_predictions']=predictions;item['targets']=pooled_target_values([item],cfg)
            item['objective_diagnostics']['joint_prediction_queries']=len(predictions)
            item['objective_diagnostics']['partial_evidence_weight']=opt.get('partial_evidence_weight',.15)
    return scored
