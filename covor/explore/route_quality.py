"""Optional observed-evidence route ranking; graph feasibility stays unchanged."""
import numpy as np
from scipy import ndimage


def observed_node_penalty(graph, weight=1.):
    """Prefer precise repeated support in broad confirmed support neighborhoods.

    Neighborhood EDT is an evidence continuity heuristic, NOT true obstacle
    clearance. It does not approve any node, edge, or path. No reference or
    actual sensor-error information is used.
    """
    if not np.isfinite(weight) or weight<0:raise ValueError('route quality weight must be finite and nonnegative')
    g=graph;idx=tuple(g.xyz.T);s=g.support_layer
    contact_limit=max(1e-6,(g.foot_contact_offset-.005)/3)
    scatter=np.nan_to_num(s['height_std'][idx],nan=contact_limit)/contact_limit
    votes=np.maximum(1.,s['votes'][idx].astype(float))
    margin=np.zeros(g.lab.shape,np.float32)
    for k in np.unique(g.xyz[:,2]):
        neighborhood=g.confirmed[:,:,max(0,k-1):min(g.lab.shape[2],k+2)].any(2)
        margin[:,:,k]=ndimage.distance_transform_edt(neighborhood)*g.res
    continuity=.2/np.maximum(g.res,margin[idx])
    penalty=weight*(scatter**2+2/votes+continuity)
    return np.where(g.node_known,penalty,0.)


def rerank_confirmed_analysis(analysis,prior,weight=1.):
    """Fixed-map ablation: reuse exactly the same graph and reachable set."""
    from covor.entry.path_eval import motion_path
    g=analysis['graph'];search=g.search(g.mission_nodes(prior['entry'],True),True,
        node_penalty=observed_node_penalty(g,weight))
    out=dict(analysis,confirmed_search=search);out['decisions']=[]
    for d,goal in zip(analysis['decisions'],prior['goals']):
        path,_=g.path(search,g.mission_nodes(goal,True))
        assert bool(path)==(d['verdict']=='passable')
        out['decisions'].append(dict(d,path=motion_path(path)))
    return out
