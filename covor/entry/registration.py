"""Conservative relative Z registration from overlapping observed support patches.

This is map registration, not a height sensor or a change to the fusion graph.
Only a constant vertical transform is estimated; robot zero fixes the gauge.
No truth, per-robot trajectory alignment, or unobserved common floor is used.
"""
from dataclasses import dataclass
import itertools
import numpy as np


@dataclass(frozen=True)
class RegistrationCfg:
    min_overlap_columns: int = 40
    max_pair_mad_m: float = .12
    max_correction_m: float = 2.
    max_cycle_residual_m: float = .15


def lowest_observed(surfaces):
    c=surfaces.cfg;out={}
    for key,a in surfaces.cells.items():
        z=a[1]/max(a[0],1e-12);var=max(a[2]/max(a[0],1e-12)-z*z,0.)
        if a[3]<c.min_frames or a[0]-a[4]<c.min_evidence or var>c.max_height_std_m**2:continue
        xy=key[:2]
        if xy not in out or z<out[xy]:out[xy]=z
    return out


def relative_heights(surfaces,cfg=None):
    cfg=cfg or RegistrationCfg();names=sorted(surfaces);patches={r:lowest_observed(surfaces[r]) for r in names}
    rows=[];A=[];b=[];weights=[]
    for a,bname in itertools.combinations(names,2):
        shared=sorted(patches[a].keys()&patches[bname].keys())
        delta=np.array([patches[bname][k]-patches[a][k] for k in shared])
        median=float(np.median(delta)) if len(delta) else 0.
        mad=float(np.median(np.abs(delta-median))) if len(delta) else float('inf')
        accepted=len(delta)>=cfg.min_overlap_columns and mad<=cfg.max_pair_mad_m
        rows.append(dict(a=a,b=bname,overlap_columns=len(delta),median_b_minus_a_m=median if len(delta) else None,
                         mad_m=mad if len(delta) else None,accepted=accepted))
        if accepted:
            row=np.zeros(len(names));row[names.index(bname)]=1;row[names.index(a)]=-1
            A.append(row[1:]);b.append(-median);weights.append(np.sqrt(len(delta))/max(mad,.02))
    offsets={r:0. for r in names}
    if not A:return offsets,rows,dict(accepted=False,reason='no reliable overlap constraints')
    A=np.array(A);b=np.array(b);w=np.array(weights)
    if np.linalg.matrix_rank(A)<len(names)-1:
        return offsets,rows,dict(accepted=False,reason='overlap graph does not connect all robots')
    z=np.linalg.lstsq(A*w[:,None],b*w,rcond=None)[0]
    residual=A@z-b
    if np.max(np.abs(z))>cfg.max_correction_m or np.max(np.abs(residual))>cfg.max_cycle_residual_m:
        return offsets,rows,dict(accepted=False,reason='correction or overlap cycle inconsistent',residual_m=residual.tolist())
    offsets.update(dict(zip(names[1:],map(float,z))))
    return offsets,rows,dict(accepted=True,reason='connected overlap surfaces; fixed robot zero Z gauge',residual_m=residual.tolist())
