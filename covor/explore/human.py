"""Human adapter: measured foot heights, three-state targets, bounded cut probes."""
from collections import deque
from dataclasses import replace
import numpy as np
from scipy import ndimage
from covor.entry.decision import DecisionGraph, body_modes
from covor.entry.traverse import TravCfg
from covor.entry.path_eval import motion_path


def relaxed_certificate(lab,layer,ijk,res,entry,goal,body,graph=None,context=None):
    """More permissive 3D graph: unlimited unknown runs, all unknown supports.

    Diagonal corner and minimum-width transitions are relaxed (never tightened).
    Only if even this graph cannot escape the observed enclosure is 'blocked'
    returned. Unknown target support or entry support never proves blockage.
    """
    def mask_region(region):
        x0,y0,x1,y1=region['xy_bounds'];z0,z1=region['floor_z_bounds']
        x=(np.arange(lab.shape[0])+ijk[0]+.5)*res;y=(np.arange(lab.shape[1])+ijk[1]+.5)*res
        z=layer['height']+.025
        return (x[:,None,None]>=x0)&(x[:,None,None]<=x1)&(y[None,:,None]>=y0)&(y[None,:,None]<=y1)&(z>=z0)&(z<=z1)&layer['support']
    start=mask_region(entry);dest=mask_region(goal)
    if not start.any() or not dest.any():return None
    cached=context.get('possible') if context is not None else None
    levels=range(max(0,lab.shape[2]-body.rows(body.h_crawl)))
    certificate_body=replace(body,step=float(np.ceil(body.step/res)*res)) if graph is not None else body
    p=None if cached is not None else body_modes(lab,certificate_body,levels,True)[0]
    # Unknown supports may occupy any level. Known supports require the cell
    # immediately above to clear the smallest posture. Observed air is no floor.
    possible=cached if cached is not None else (p>=0)&((lab==0)|layer['support'])
    # A measured surface can lie in a free mixed cell; retain its valid node.
    if p is not None:possible|=(p>=0)&layer['support']
    if graph is not None and cached is None:
        off=getattr(graph,'map_offset',np.zeros(3,int));sl=tuple(slice(int(a),int(a+n)) for a,n in zip(off,graph.lab.shape))
        possible[sl]|=graph.possible
    if context is not None:context['possible']=possible
    step=int(np.floor(body.step/res+1e-9))
    # step=1 is too restrictive for quantised supports near a bin edge. Admit
    # 2 rows as a RELAXATION for the certificate; no false blocked from rounding.
    step=max(step,int(np.ceil(body.step/res)))
    structure=np.zeros((3,3,2*step+1),bool);structure[:]=True;structure[1,1,:]=False;structure[1,1,step]=True
    # ndimage.label requires side 3; propagation accepts the larger stencil.
    seen=ndimage.binary_propagation(dest&possible,structure=structure,mask=possible)
    if (seen&start).any():return None
    if seen[0].any() or seen[-1].any() or seen[:,0].any() or seen[:,-1].any():return None
    if not seen.any():return None
    # Observed closed cut: every absent candidate neighbouring reached nodes
    # must have actual measured evidence, not just a missing support hypothesis.
    return dict(reason='observed_3d_enclosure_even_with_unlimited_unknown_connections',relaxation='2-row step, diagonal corners relaxed',reachable_relaxed_nodes=int(seen.sum()))


class Human:
    def __init__(self,prior,cfg):self.prior=prior;self.cfg=cfg;self.res=cfg['map_resolution'];self.body=TravCfg(res=self.res)

    def analyze(self,lab,layer,ijk,influence=True):
        graph_lab=lab;graph_layer=layer;graph_ijk=ijk;offset=np.zeros(3,int)
        if self.cfg['human'].get('crop_observed_graph'):
            anchors=layer['support']|((layer['votes']>0)&(layer['evidence']>0)&np.isfinite(layer['height']))
            pts=np.argwhere(anchors)
            if len(pts):
                pad=np.ceil(np.array([1.,1.,.3])/self.res).astype(int);head=np.ceil(np.array([1.,1.,2.3])/self.res).astype(int)
                offset=np.maximum(pts.min(0)-pad,0);hi=np.minimum(pts.max(0)+head+1,lab.shape)
            else:hi=np.minimum(np.array([1,1,4]),lab.shape)
            sl=tuple(slice(int(a),int(b)) for a,b in zip(offset,hi));graph_lab=lab[sl];graph_layer={k:v[sl] for k,v in layer.items()};graph_ijk=np.asarray(ijk)+offset
        g=DecisionGraph(graph_lab,graph_ijk,self.res,body=self.body,support_layer=graph_layer,surface_geometry=True,foot_contact_offset=self.cfg['human']['foot_contact_offset'],resolve_low_surface_band=self.cfg['human'].get('resolve_low_surface_band',False),use_surface_xy=self.cfg['human'].get('use_surface_xy',False))
        g.map_offset=offset
        starts=g.mission_nodes(self.prior['entry'],True);ps=g.mission_nodes(self.prior['entry'])
        confirmed=g.search(starts,True);possible=g.reachability_search(ps) if len(g.xyz) else dict(best={},parent={},via={},dist={})
        decisions=[];dependencies={};certificate_context={}
        for goal in self.prior['goals']:
            path,_=g.path(confirmed,g.mission_nodes(goal,True));hyp,deps=g.path(possible,g.mission_nodes(goal))
            proof=None
            if path:verdict='passable';reason='measured_support_and_body'
            else:
                proof=relaxed_certificate(lab,layer,ijk,self.res,self.prior['entry'],goal,self.body,g if self.cfg['human'].get('resolve_low_surface_band') else None,certificate_context) if starts and not hyp else None
                verdict='blocked' if proof else 'unverified'
                reason=proof['reason'] if proof else ('entry_support_or_body_unconfirmed' if not starts else 'incomplete_support_body_or_connections')
            decisions.append(dict(goal=goal['id'],verdict=verdict,reason=reason,path=motion_path(path),possible_path=hyp,uncertain_segments=deps,certificate=proof))
            if hyp:dependencies[goal['id']]=set(deps)
        impact={};cut=[]
        if influence:
            # Same deterministic bounded computation for both policies. A
            # skipped segment is logged as unscored, never assumed important.
            sids=sorted(set().union(*dependencies.values()) if dependencies else [])
            for sid in sids[:self.cfg['candidates']['max_impact_segments']]:
                closed=g.reachability_search(ps,closed=sid);affected=[]
                for goal in self.prior['goals']:
                    if sid not in dependencies.get(goal['id'],set()):continue
                    reached=any(n in closed['best'] for n in g.mission_nodes(goal))
                    closed_path,closed_deps=g.path(closed,g.mission_nodes(goal))
                    cut.append(dict(segment=sid,goal=goal['id'],open_reachable=True,closed_reachable=reached,closed_path=closed_path,closed_path_segments=closed_deps,other_segments='unchanged possibility graph'))
                    if not reached:affected.append(goal['id'])
                if affected:impact[sid]=affected
        return dict(graph=g,decisions=decisions,impact=impact,cut=cut,n_entry=len(starts),n_confirmed=int(g.confirmed.sum()),n_possible=int(g.possible.sum()),confirmed_search=confirmed)


def complete_support(building,ijk,shape,res):
    """Stage-1 only: exhaustive observed support of explicit horizontal faces."""
    if hasattr(building,'complete_support'):return building.complete_support(ijk,shape,res)
    layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan,np.float32),evidence=np.zeros(shape,np.float32),votes=np.zeros(shape,np.uint16))
    if getattr(building,'experiment_cfg',{}).get('human',{}).get('resolve_low_surface_band'):layer['height_std']=np.zeros(shape,np.float32)
    _,nodes,_=building.reference(res)
    for p in nodes.values():
        surface=p.copy();surface[2]-=.005;k=np.floor(surface/res).astype(int)-ijk
        if ((k>=0)&(k<shape)).all():
            q=tuple(k);layer['support'][q]=True;layer['height'][q]=surface[2];layer['evidence'][q]=2;layer['votes'][q]=2
    return layer


def ensure_current_analysis(human,analysis,lab,layer,ijk,map_version):
    """The utility graph must describe the map used for flight and visibility."""
    if analysis is None or analysis.get('map_version')!=map_version:
        analysis=human.analyze(lab,layer,ijk,True)
        analysis['map_version']=map_version
        return analysis,True
    return analysis,False
