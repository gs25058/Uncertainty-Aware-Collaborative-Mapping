#!/usr/bin/env python3
"""Physical-protocol corridor smoke and one-view reobservation experiment.

True camera pose renders a textured stereo pair; fixed estimated pose integrates
only the SGBM recomputed from that pair. Candidate selection sees no images or
truth. This is synthetic recorded-view replay, not an executed flight planner.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/"scripts"/"synth", ROOT/"scripts"/"entry"):
    sys.path.insert(0,str(p))
import gtsam  # must precede open3d
import cv2
import numpy as np
import octomap

from covor.synth.config import SynthCfg, ROBOTS
from covor.synth import dataset as DS
from covor.synth.render import DepthRenderer, _depth_rectified
from covor.smoke import stereo_smoke, clean_right_depth
from covor.entry import clean3d, traverse as TV
from covor.entry.config import EntryCfg
from covor.reobserve import rank_candidates, goal_verdicts
from textured_scene import textured_scene
from fuse_synth import fuse, CONDITIONS
from run_traverse import figure as traverse_figure
from common import (conditions, write_json, write_csv, digest, make_builder,
                    sparse_map, dense_map, map_metrics, depth_metrics,
                    frame_points, save_example, map_figure, MAPPING_ARMS)

ENTRY=(-2.35,-19.55)
# Mission coordinates fixed before looking at any experiment's target scores.
# Near-entry and distant corridor points plus lateral/room-side requests.
GOALS=[dict(id=f"south_{k}",x=-2.35,y=y) for k,y in enumerate(
    (-19.25,-18.75,-18.25,-17.75,-17.25,-16.75,-16.25,-15.75))]
GOALS += [dict(id=f"corridor_{k}",x=-2.35,y=y) for k,y in enumerate((-14.,-10.,-5.,0.,5.,10.,15.,18.))]
GOALS += [dict(id=f"lateral_{k}",x=x,y=y) for k,(x,y) in enumerate(
    ((-.5,-18.),(0.,-10.),(0.,0.),(2.,0.),(-4.,5.),(1.,15.)))]


def traversal(lab,ijk,res,strict=False,snap_m=1.5):
    if (lab==2).any():
        lab,_=clean3d.clean(lab,EntryCfg(res=res))
    cfg=TV.TravCfg(res=res)
    if strict:
        cfg=replace(cfg,gap=0.)
    return TV.traverse(lab,cfg,ENTRY,ijk,snap_m=snap_m)


def pose_inputs(out,cfg):
    d=out/"poses"
    paths={r:d/f"{r}.npz" for r in ROBOTS}
    if not all(p.exists() for p in paths.values()):
        DS.install_paths(cfg)
        poses,stats=fuse(cfg.seq,CONDITIONS["C_3drone"])
        gt=DS.load_gt_traj(cfg)
        d.mkdir(exist_ok=True)
        for r,p in poses.items():
            np.savez_compressed(paths[r],**p,T_truth=DS.gt_camera_poses(r,p["t"],gt)[0])
        write_json(d/"fusion_stats.json",stats)
    return {r:dict(np.load(p)) for r,p in paths.items()}


def image_input(out,r,i,P,rend):
    cache=out/"input_cache"
    cache.mkdir(exist_ok=True)
    path=cache/f"{r}_{i:05d}.npz"
    if path.exists():
        q=np.load(path)
        assert np.array_equal(q["T_truth"],P["T_truth"][i]),"stale render cache"
        return q["left"],q["right"],q["z"],q["zr"],q["sz"],q["valid"]
    left,right=rend.images(P["T_truth"][i])
    z,sz,valid=_depth_rectified(rend.sd,left,right)
    zr=clean_right_depth(rend.sd,left,right)
    np.savez_compressed(path,left=left,right=right,z=z,zr=zr,sz=sz,valid=valid,
                        T_truth=P["T_truth"][i])
    return left,right,z,zr,sz,valid


def teammates(poses,r,t):
    p=[]
    for other,P in poses.items():
        if other==r:
            continue
        i=np.abs(P["t"]-t).argmin()
        if abs(P["t"][i]-t)<=.5:
            p.append(P["p_body"][i])
    return np.array(p) if p else None


def observe(out,r,i,P,rend,smoke,name,example=False):
    left,right,zc,zr,_,vc=image_input(out,r,i,P,rend)
    # True pose is used for synthesis ONLY, never candidate utility prediction.
    il,ir,meta=stereo_smoke(rend.sd,left,right,zc,zr,P["T_truth"][i],
                           float(P["t"][i]),smoke)
    z,sz,v=_depth_rectified(rend.sd,il,ir)
    pc,sp=frame_points(rend.sd,z,sz,v)
    if example:
        save_example(out,name,left,right,il,ir,zc,z)
    row=dict(condition=name,robot=r,pose_index=i,timestamp=float(P["t"][i]),
             **smoke.metadata(),**depth_metrics(z,v,zc,vc),**meta)
    return pc,sp,row


def eval_map(lab,tg,gt_lab,ijk,res,goals,strict_entry_snap=1.5):
    tm=traversal(lab,ijk,res)
    strict=traversal(lab,ijk,res,strict=True,snap_m=strict_entry_snap)
    verdicts=goal_verdicts(tm,strict,lab,ijk,res,goals)
    counts=dict(n_goals=len(goals),confirmed_reachable_goals=0,
                false_confirmed_gt_unreachable=0,false_confirmed_gt_blocked=0,
                unverified_goals=0,gap_permitted_true_goals=0,
                false_gap_permitted_gt_unreachable=0,false_gap_permitted_gt_blocked=0)
    for row in verdicts:
        xy=np.floor(np.array([row["x"],row["y"]])/res).astype(int)-ijk[:2]
        at=tuple(xy)
        inside=((xy>=0)&(xy<np.array(gt_lab.shape[:2]))).all()
        truth_reach=bool(inside and tg["reach"][at])
        # A GT-unreachable goal is not automatically called blocked. Report
        # occupied target columns separately from GT-unknown/nonconnected ones.
        z=(np.arange(gt_lab.shape[2])+ijk[2]+.5)*res
        solid=bool(inside and not tg["feasible"][at] and
                   (gt_lab[at][(z>=.3)&(z<=.9)]==2).any())
        row.update(gt_reachable=truth_reach,gt_observed_blocked=solid)
        counts["confirmed_reachable_goals"]+=int(row["confirmed"] and truth_reach)
        counts["false_confirmed_gt_unreachable"]+=int(row["confirmed"] and not truth_reach)
        counts["false_confirmed_gt_blocked"]+=int(row["confirmed"] and solid)
        counts["unverified_goals"]+=int(not row["confirmed"])
        counts["gap_permitted_true_goals"]+=int(row["gap_permitted_reach"] and truth_reach)
        counts["false_gap_permitted_gt_unreachable"]+=int(row["gap_permitted_reach"] and not truth_reach)
        counts["false_gap_permitted_gt_blocked"]+=int(row["gap_permitted_reach"] and solid)
    return dict(**TV.score(tm,tg),**counts,strict_reach_columns=int(strict["reach"].sum()),
                entry_snapped_m=tm["entry_snapped_m"],
                strict_entry_snapped_m=strict["entry_snapped_m"]),verdicts,tm


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True)
    ap.add_argument("--frames",type=int,default=100,help="initial frames PER ROBOT")
    ap.add_argument("--stride",type=int,default=12)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--levels",default="all")
    ap.add_argument("--betas",default="0.15,0.4,0.8")
    ap.add_argument("--candidates",type=int,default=24,help="total, balanced across 3 robots")
    ap.add_argument("--skip-selection",action="store_true")
    ap.add_argument("--mapping-arm",choices=MAPPING_ARMS,default="depth_only")
    args=ap.parse_args()
    if args.frames<1 or args.stride<2:
        ap.error("positive frames and stride >=2 required")
    if args.candidates<3 or args.candidates%3:
        ap.error("candidate count must be positive and divisible by three")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    cfg=SynthCfg(name="corridor915f",seq="synth_corridor915f_0")
    gt_path=Path(cfg.outdir())/"gt_voxel_fs.npz"
    zgt=np.load(gt_path)
    gt_lab,ijk,res=zgt["labels"],zgt["ijk_min"],float(zgt["res"])
    poses=pose_inputs(out,cfg)
    scene,tex=textured_scene(cfg.mesh_config())
    renderers={r:DepthRenderer(scene,r,textures=tex) for r in ROBOTS}
    configs=conditions(args.seed,tuple(map(float,args.betas.split(","))))
    if args.levels!="all":
        configs={k:v for k,v in configs.items() if k=="clean" or k in args.levels.split(",")}
    initial={r:list(range(0,len(P["t"]),args.stride))[:args.frames] for r,P in poses.items()}
    candidates=[]
    for r,P in poses.items():
        pool=np.arange(args.stride//2,len(P["t"]),args.stride)
        pool=pool[np.linspace(0,len(pool)-1,max(args.candidates//3,1)).astype(int)]
        for i in pool:
            assert i not in initial[r]
            candidates.append(dict(id=len(candidates),robot=r,index=int(i),
                                   timestamp=float(P["t"][i]),T=P["T"][i].tolist()))
    write_json(out/"design.json",dict(args=vars(args),entry=ENTRY,goals=GOALS,
               initial=initial,candidates=candidates,protocol="true-pose render, estimated-pose integration",
               selection="current-map visibility proxy only; candidate images rendered AFTER selection",
               budget="one frame = one stereo pair; synthetic recorded-view replay, no motion planner",
               occ_cfg=make_builder(args.mapping_arm).cfg.__dict__,conditions={k:v.metadata() for k,v in configs.items()},
               inputs={str(p):digest(p) for p in [gt_path,Path(cfg.mesh_config()),
                   *[out/"poses"/f"{r}.npz" for r in ROBOTS]]}))
    # GT is evaluated separately; rank_candidates has no access to this result.
    tg=traversal(gt_lab,ijk,res)
    summary,frame_rows,goal_rows,selection_rows=[],[],[],[]
    maps,travs={},{}
    started=time.time()
    for name,smoke in configs.items():
        builder=make_builder(args.mapping_arm)
        for r,P in poses.items():
            for n,i in enumerate(initial[r]):
                pc,sp,row=observe(out,r,i,P,renderers[r],smoke,name,example=n==0 and r==ROBOTS[0])
                frame_rows.append(dict(stage="initial",**row))
                if pc is not None:
                    builder.integrate_frame(P["T"][i],float(P["tr"][i]),pc,sp,
                                            teammates=teammates(poses,r,P["t"][i]))
                if (n+1)%25==0:
                    print(name,r,n+1,"initial frames",round(time.time()-started),"s",flush=True)
        builder.finalize()
        lab=dense_map(sparse_map(builder),ijk,gt_lab.shape)
        maps[name]=lab
        np.savez_compressed(out/f"map_{name}_initial.npz",labels=lab,ijk_min=ijk,res=res)
        scores,verdicts,tm=eval_map(lab,tg,gt_lab,ijk,res,GOALS)
        travs[name]=tm
        base=dict(condition=name,mapping_arm=args.mapping_arm,policy="initial",additional_observations=0,**scores,
                  **map_metrics(lab,maps["clean"]))
        summary.append(base)
        goal_rows.extend(dict(condition=name,policy="initial",**v) for v in verdicts)
        if not args.skip_selection:
            # No GT, render pose, measured depth or candidate outcome is passed.
            picks,utilities,importance=rank_candidates(lab,ijk,res,candidates,
                {r:renderers[r].sd for r in ROBOTS},
                {r:P["T"][initial[r][0],:3,3] for r,P in poses.items()},ENTRY,GOALS)
            selection_rows.extend(dict(condition=name,**u) for u in utilities)
            write_json(out/f"selection_{name}.json",dict(picks=picks,scores=utilities,
                       current_labels_sha256=hashlib.sha256(lab.tobytes()).hexdigest(),
                       candidate_image_access=False))
            np.save(out/f"route_importance_{name}.npy",importance)
            # Full .ot preserves log-odds. Binary .bt would collapse evidence.
            tree_path=out/f"initial_{name}.ot"
            builder.tree.write(str(tree_path).encode())
            outcomes={}
            for policy,cid in picks.items():
                if cid not in outcomes:
                    cand=candidates[cid]
                    r,i=cand["robot"],cand["index"]
                    P=poses[r]
                    pc,sp,row=observe(out,r,i,P,renderers[r],smoke,name)
                    frame_rows.append(dict(stage="selected",candidate_id=cid,**row))
                    extra=make_builder(args.mapping_arm)
                    extra.tree=octomap.OcTree.read(str(tree_path).encode())
                    # Value check of full-tree cloning before adding evidence.
                    before=dense_map(sparse_map(extra),ijk,gt_lab.shape)
                    assert np.array_equal(lab,before)
                    if pc is not None:
                        extra.integrate_frame(P["T"][i],float(P["tr"][i]),pc,sp,
                                              teammates=teammates(poses,r,P["t"][i]))
                    extra.finalize()
                    after=dense_map(sparse_map(extra),ijk,gt_lab.shape)
                    s,v,ta=eval_map(after,tg,gt_lab,ijk,res,GOALS)
                    outcomes[cid]=(after,s,v,ta)
                after,s,v,ta=outcomes[cid]
                np.savez_compressed(out/f"map_{name}_{policy}.npz",labels=after,ijk_min=ijk,res=res)
                summary.append(dict(condition=name,mapping_arm=args.mapping_arm,policy=policy,candidate_id=cid,
                    additional_observations=1,**s,**map_metrics(after,maps["clean"]),
                    delta_true_confirmed=s["confirmed_reachable_goals"]-base["confirmed_reachable_goals"],
                    delta_false_confirmed=s["false_confirmed_gt_unreachable"]-base["false_confirmed_gt_unreachable"],
                    delta_gap_true_goals=s["gap_permitted_true_goals"]-base["gap_permitted_true_goals"],
                    delta_false_gap=s["false_gap_permitted_gt_unreachable"]-base["false_gap_permitted_gt_unreachable"]))
                goal_rows.extend(dict(condition=name,policy=policy,**vv) for vv in v)
                if name=="heterogeneous_medium":
                    travs["medium_"+policy]=ta
        write_csv(out/"summary.csv",summary)
        write_csv(out/"frames.csv",frame_rows)
        write_csv(out/"goals.csv",goal_rows)
        write_csv(out/"candidate_scores.csv",selection_rows)
        print(name,"recall",scores["reach_recall"],"false",scores["false_reach_rate"],
              "confirmed goals",scores["confirmed_reachable_goals"],flush=True)
    shown={k:v for k,v in maps.items() if k in ("clean","uniform_medium","heterogeneous_medium","heterogeneous_strong")}
    map_figure(out/"map_comparison.png",shown,ijk,res)
    fig=[("GT",tg)]+[(k,v) for k,v in travs.items() if k in
        ("clean","heterogeneous_medium","heterogeneous_strong")]
    traverse_figure(fig,ijk,res,out/"entry_comparison.png","True-pose render / estimated-pose integration")
    if "medium_route" in travs:
        traverse_figure([(k,travs[k]) for k in ("heterogeneous_medium","medium_nearest","medium_unknown","medium_route")],
                        ijk,res,out/"reobservation_comparison.png","One additional stereo pair per policy")
    write_json(out/"run.json",dict(elapsed_s=time.time()-started,complete=True))


if __name__=="__main__":
    main()
