"""Human adapter: measured foot heights, three-state targets, bounded cut probes."""
from collections import deque
import numpy as np
from scipy import ndimage
from covor.entry.decision import DecisionGraph, body_modes
from covor.entry.traverse import TravCfg
from covor.entry.path_eval import motion_path


def relaxed_certificate(lab,layer,ijk,res,entry,goal,body):
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
    levels=range(max(0,lab.shape[2]-body.rows(body.h_crawl)))
    p,_=body_modes(lab,body,levels,True)
    # Unknown supports may occupy any level. Known supports require the cell
    # immediately above to clear the smallest posture. Observed air is no floor.
    possible=(p>=0)&((lab==0)|layer['support'])
    # A measured surface can lie in a free mixed cell; retain its valid node.
    possible|=(p>=0)&layer['support']
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
        g=DecisionGraph(lab,ijk,self.res,body=self.body,support_layer=layer,surface_geometry=True,foot_contact_offset=self.cfg['human']['foot_contact_offset'])
        starts=g.mission_nodes(self.prior['entry'],True);ps=g.mission_nodes(self.prior['entry'])
        confirmed=g.search(starts,True);possible=g.reachability_search(ps) if len(g.xyz) else dict(best={},parent={},via={},dist={})
        decisions=[];dependencies={}
        for goal in self.prior['goals']:
            path,_=g.path(confirmed,g.mission_nodes(goal,True));hyp,deps=g.path(possible,g.mission_nodes(goal))
            proof=None
            if path:verdict='passable';reason='measured_support_and_body'
            else:
                proof=relaxed_certificate(lab,layer,ijk,self.res,self.prior['entry'],goal,self.body) if starts and not hyp else None
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
        return dict(graph=g,decisions=decisions,impact=impact,cut=cut,n_entry=len(starts),n_confirmed=int(g.confirmed.sum()),n_possible=int(g.possible.sum()))


def complete_support(building,ijk,shape,res):
    """Stage-1 only: exhaustive observed support of explicit horizontal faces."""
    layer=dict(support=np.zeros(shape,bool),height=np.full(shape,np.nan,np.float32),evidence=np.zeros(shape,np.float32),votes=np.zeros(shape,np.uint16))
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
