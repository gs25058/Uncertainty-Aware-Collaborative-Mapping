"""Inlier and sparse-error second moments from public observations.

No true depth, true normal, smoke transmission, or outlier flag is used.
Temporal fusion retains their correlation; neither term is assumed independent.
"""
import numpy as np
from scipy import ndimage


def uncertainty_components(sd,z,valid,T,cfg,condition='clean'):
    p=cfg['sensor']['error_model'];h,w=z.shape;v,u=np.indices(z.shape)
    raw=np.stack(((u-sd.cx)*z/sd.fx,(v-sd.cy)*z/sd.fy,z),-1)@sd.R1
    world=raw@T[:3,:3].T
    dx=world[1:-1,2:]-world[1:-1,:-2];dy=world[2:,1:-1]-world[:-2,1:-1]
    normal=np.cross(dx,dy);ray=world[1:-1,1:-1]
    incidence=np.ones(z.shape,float)
    denom=np.linalg.norm(normal,axis=-1)*np.linalg.norm(ray,axis=-1)
    cosine=np.abs((normal*ray).sum(-1))/np.maximum(denom,1e-12)
    angle=np.rad2deg(np.arccos(np.clip(cosine,0,1)))
    neighbours=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    severity=np.clip((angle-p['grazing_start_deg'])/(p['grazing_full_deg']-p['grazing_start_deg']),0,1)
    incidence[1:-1,1:-1]=np.where(neighbours,severity,1.)
    edge=np.zeros(z.shape,bool)
    for axis in [0,1]:
        a=np.take(z,range(z.shape[axis]-1),axis=axis);b=np.take(z,range(1,z.shape[axis]),axis=axis)
        bad=(np.isfinite(a)!=np.isfinite(b))|(abs(a-b)>np.maximum(p['edge_absolute_threshold_m'],p['edge_relative_threshold']*np.minimum(a,b)))
        s=[slice(None)]*2;s[axis]=slice(None,-1);edge[tuple(s)]|=bad
        s[axis]=slice(1,None);edge[tuple(s)]|=bad
    if p['edge_dilation_pixels']>0:edge=ndimage.binary_dilation(edge,iterations=p['edge_dilation_pixels'])
    geometry=1+p['geometry_sigma_increase']*np.maximum(incidence,edge)
    degradation=p['degradation']['medium' if condition=='smoke' else condition]
    # Configured degradation upper bound; the actual 3D field/T remains private.
    sigma=(p['sigma_constant_m']+p['sigma_quadratic_per_m']*z*z)*geometry*(1+degradation['k'])
    a,b=p['outlier_min_m'],p['outlier_max_m']
    mixture=np.clip(p['outlier_probability']+degradation['q'],0.,1.)*(a*a+a*b+b*b)/3
    return np.where(valid,sigma,np.nan).astype(np.float32),np.full(z.shape,mixture,np.float32)
