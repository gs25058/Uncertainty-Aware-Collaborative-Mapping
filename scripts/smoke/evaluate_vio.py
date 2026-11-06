#!/usr/bin/env python3
"""Score short raw-IR VINS replays on their common time window; no scale fit."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from covor import data as D
from covor.fusion import umeyama_yaw
from common import write_csv,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--run-id",default=None)
    args=ap.parse_args()
    root=Path(args.input)
    design=json.loads((root/"design.json").read_text())
    runroot=root/"repeats"/args.run_id if args.run_id else root
    epoch=D.timeshift("default_3_zigzag_0")
    spl,*_=D._mocap_splines("default_3_zigzag_0",design["args"]["robot"])
    tracks={}
    for name in design["configs"]:
        p=runroot/name/"vio.csv"
        if p.exists() and p.stat().st_size:
            a=np.atleast_2d(np.genfromtxt(p,delimiter=","))[:,:11]
            a=a[np.isfinite(a).all(axis=1)]
            a[:,0]=a[:,0]/1e9-epoch
            a=a[np.r_[True,np.diff(a[:,0])>0]]
            if len(a)>2:tracks[name]=a
    rows=[]
    if not tracks:
        write_json(runroot/"evaluation.json",dict(status="no VIO poses"))
        return
    lo=max(a[0,0] for a in tracks.values());hi=min(a[-1,0] for a in tracks.values())
    tq=np.arange(lo,hi,1/15)
    gt=np.asarray(spl(tq)).T
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name,a in tracks.items():
        est=np.stack([np.interp(tq,a[:,0],a[:,j]) for j in (1,2,3)],axis=1)
        R,t=umeyama_yaw(est,gt)
        aligned=est@R.T+t
        err=np.linalg.norm(aligned-gt,axis=1)
        row=dict(condition=name,run_id=args.run_id or "first",n_poses=len(a),first_t=float(a[0,0]),last_t=float(a[-1,0]),
                 n_input_stereo=design["frames"],common_start=lo,common_end=hi,
                 common_ate_yaw_rmse_m=float(np.sqrt(np.mean(err**2))),
                 common_ate_median_m=float(np.median(err)),
                 max_output_gap_s=float(np.diff(a[:,0]).max()),
                 output_time_span_s=float(a[-1,0]-a[0,0]))
        rows.append(row)
        ax[0].plot(aligned[:,0],aligned[:,1],label=name)
        ax[1].plot(tq,err,label=name)
    ax[0].plot(gt[:,0],gt[:,1],"k--",label="mocap")
    ax[0].axis("equal");ax[0].set_xlabel("x [m]");ax[0].set_ylabel("y [m]")
    ax[1].set_xlabel("time [s]");ax[1].set_ylabel("aligned position error [m]")
    for a in ax:a.legend();a.grid(alpha=.3)
    fig.tight_layout();fig.savefig(runroot/"vio_comparison.png",dpi=140);plt.close(fig)
    write_csv(runroot/"summary.csv",rows)
    write_json(runroot/"evaluation.json",dict(status="complete",common_query_times=tq,
        method="4-DoF yaw+translation alignment for evaluation only; no scale; cleaned MILUV mocap",
        limitation="one robot, short replay, original IMU retained; no three-robot UWB fusion rerun"))
    print(rows)


if __name__=="__main__":main()
