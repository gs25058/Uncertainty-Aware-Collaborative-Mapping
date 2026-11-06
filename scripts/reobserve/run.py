#!/usr/bin/env python3
"""Reuse smoke maps; select one observation; score target decisions and paths.

Stages provide an auditable separation: reference curation, current-only
selection, then observation/reference access and evaluation. Existing files are
read-only. The study is offline replay with batch-optimized proposal poses.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"scripts/smoke",ROOT/"scripts/entry",Path(__file__).parent):sys.path.insert(0,str(p))
import gtsam  # before open3d
import numpy as np
import cv2
import octomap
from covor.occupancy import StereoDepth,load_stereo_calib
from covor.entry.decision import DecisionGraph,DecisionCfg
from covor.entry.assignment import choose
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg,motion_path
from covor.entry import traverse as TV
from common import write_json,write_csv,digest,make_builder,sparse_map,dense_map
from evaluation import legacy_paths,metrics


def current_cases(source):
    old=json.load(open(source/"design.json"))
    def region(x,y,r):return dict(xy_bounds=[x-r,y-r,x+r,y+r],floor_z_bounds=[-.05,.35])
    return dict(scene="corridor915f",entry=region(*old["entry"],.04),
        goals=[dict(id=f'corridor915f:south:{g["id"]}',old_id=g["id"],kind="fixed_corridor_arrival_patch",
                    **region(g["x"],g["y"],.08)) for g in old["goals"]],
        decision_cfg=asdict(DecisionCfg()),path_eval_cfg=asdict(PathEvalCfg()),
        body=asdict(TV.TravCfg()),reference_transform=np.eye(4).tolist(),
        missing_real_scene_types=["annotated closed unique door","annotated alternative door route",
                                  "independently annotated required posture passage"],
        scope="single scanned floor; no wall/floor-crossing snap; fixed 0.16 m arrival squares",
        source_design_sha256=digest(source/"design.json"))


def reference_scene():
    from covor.synth import mesh_gt as MG
    import trimesh
    mc=json.load(open(ROOT/"results/synth_corridor915f/mesh_config.json"))
    mesh=trimesh.load(mc["obj"],force="mesh",process=False)
    mesh.apply_transform(np.asarray(mc["T_mesh_world"]))
    return MG.raycasting_scene(mesh),mc


def prepare(source,out):
    # Fixed areas derive from previous mission coordinates, not policy outcomes.
    cases=current_cases(source)
    write_json(out/"cases_geometry.json",cases)
    gt=np.load(ROOT/"results/synth_corridor915f/gt_voxel_fs.npz")
    scene,mc=reference_scene()
    graph=DecisionGraph(gt["labels"],gt["ijk_min"],float(gt["res"]))
    ref=graph.decide(cases["entry"],cases["goals"])
    validator=ReferenceValidator(gt["labels"],gt["ijk_min"],float(gt["res"]),scene)
    references=[]
    for goal,d in zip(cases["goals"],ref["decisions"]):
        path=motion_path(d["path"])
        check=validator.validate(path,cases["entry"],goal) if path else dict(status="not_proposed",reason=d["reason"])
        # A reference reachable label needs an independent mesh-swept witness.
        # A graph failure alone never establishes unreachable ground truth.
        if check["status"]=="valid":state="reachable"
        elif d["verdict"]=="blocked" and d["blockage_certificate"]:
            state="blocked"
        else:state="undetermined"
        row=dict(case_id=goal["id"],reference_state=state,reason=check["reason"],
                 graph_verdict=d["verdict"],mesh_witness=check,blockage_certificate=d["blockage_certificate"])
        references.append(row)
        write_json(out/"reference_paths"/(goal["old_id"]+".json"),dict(path=path,validation=check))
        print("reference",goal["old_id"],state,check["reason"],flush=True)
    write_json(out/"references.json",references)
    write_json(out/"cases.json",dict(cases,goals=[dict(g,reference=references[i]) for i,g in enumerate(cases["goals"])]))
    write_json(out/"reference_provenance.json",dict(mesh=mc["obj"],mesh_sha256=digest(mc["obj"]),
          gt_sha256=digest(ROOT/"results/synth_corridor915f/gt_voxel_fs.npz"),
          reference_not_just_same_classifier=True,config_frozen_before_policy_execution=True))


def proposals(source,old):
    # Only estimated pose arrays are accessed in this phase. They ARE batch
    # optimized with later data: label retrospective, never online selection.
    refs={}
    for r,indices in old["initial"].items():
        with np.load(source/"poses"/f"{r}.npz") as p:refs[r]=p["T"][indices[-1],:3,3]
    out=[]
    for c in old["candidates"]:
        out.append(dict(c,move_cost=float(np.linalg.norm(np.asarray(c["T"])[:3,3]-refs[c["robot"]])),
                        pose_provenance="offline batch-fused; later measurements used; not causal"))
    return out


def select(source,out,names,max_move):
    cases=json.load(open(out/"cases_geometry.json"));old=json.load(open(source/"design.json"))
    candidates=proposals(source,old)
    calibrations={r:StereoDepth(load_stereo_calib(r)) for r in old["initial"]}
    write_json(out/"candidates.json",dict(candidates=candidates,max_move_m=max_move,
               motion_cost="Euclidean distance from each robot LAST initial camera pose; retrospective view catalog, no flight feasibility claim",
               camera={r:dict(size=s.calib.size,fx=s.fx,fy=s.fy,cx=s.cx,cy=s.cy,R1=s.R1) for r,s in calibrations.items()}))
    for name in names:
        q=np.load(source/f"map_{name}_initial.npz")
        graph=DecisionGraph(q["labels"],q["ijk_min"],float(q["res"]),DecisionCfg(**cases["decision_cfg"]))
        analysis=graph.decide(cases["entry"],cases["goals"],with_influence=True)
        picks,scores,reason=choose(graph,analysis,candidates,calibrations,max_move)
        log=dict(condition=name,picks=picks,reason=reason,scores=scores,
                 current_map_sha256=digest(source/f"map_{name}_initial.npz"),
                 candidate_depth_access=False,ground_truth_access=False,
                 proposal_pose_scope="retrospective batch optimization; not an online planning experiment",
                 segments=[dict(s,affected_goals=analysis["impact"][s["id"]],importance=len(analysis["impact"][s["id"]])) for s in graph.segments])
        write_json(out/"selection"/f"{name}.json",log)
        write_json(out/"initial_decisions"/f"{name}.json",analysis["decisions"])
        write_json(out/"hypothesis_paths"/f"{name}.json",dict(open_all=analysis["decisions"],
               one_segment_closed=analysis["cut_evidence"],
               definition="close only one segment; all other possible connections unchanged; reachability impact, not immediate confirmation"))
        # Save exact graph masks/segment IDs for independent replay of the cuts.
        (out/"graphs").mkdir(exist_ok=True)
        np.savez_compressed(out/"graphs"/f"{name}.npz",confirmed=graph.confirmed,
                            possible=graph.possible,segments=graph.region,ijk_min=graph.ijk,res=graph.res)
        print("selection",name,picks,"positive segments",sum(bool(v) for v in analysis["impact"].values()),reason,flush=True)


def observation(source,out,condition,candidate,smoke,renderers):
    # Called exclusively AFTER the selection file has been written.
    from covor.synth.render import DepthRenderer,_depth_rectified
    from covor.smoke import stereo_smoke,clean_right_depth
    from covor.synth.config import SynthCfg
    from textured_scene import textured_scene
    r,i=candidate["robot"],candidate["index"]
    p=np.load(source/"poses"/f"{r}.npz")
    oldcache=source/"input_cache"/f"{r}_{i:05d}.npz"
    sd=StereoDepth(load_stereo_calib(r))
    if oldcache.exists():
        c=np.load(oldcache)
        assert np.array_equal(c["T_truth"],p["T_truth"][i])
        left,right,z,zr,sz,valid=(c[k] for k in ("left","right","z","zr","sz","valid"))
        cache_source=str(oldcache)
    else:
        if "scene" not in renderers:
            cfg=SynthCfg(name="corridor915f",seq="synth_corridor915f_0")
            renderers["scene"],renderers["textures"]=textured_scene(cfg.mesh_config())
        renderer=DepthRenderer(renderers["scene"],r,textures=renderers["textures"])
        left,right=renderer.images(p["T_truth"][i])
        z,sz,valid=_depth_rectified(sd,left,right);zr=clean_right_depth(sd,left,right)
        cache_source="new true-pose render: this withheld frame absent from old cache"
    clean_z=z.copy()
    if smoke.beta:
        left,right,meta=stereo_smoke(sd,left,right,z,zr,p["T_truth"][i],float(p["t"][i]),smoke)
        z,sz,valid=_depth_rectified(sd,left,right)
    pc,sp=sd.backproject(z,sz,valid,downsample=4) if valid.sum()>=100 else (np.empty((0,3)),np.empty(0))
    directory=out/"observations";directory.mkdir(exist_ok=True)
    path=directory/f"{condition}_candidate{candidate['id']}.npz"
    np.savez_compressed(path,left=left,right=right,Z=z,sigma_Z=sz,valid=valid,P_cam=pc,sZ=sp,
                        T_est=p["T"][i],T_truth=p["T_truth"][i],timestamp=p["t"][i],
                        robot=r,frame_index=i,candidate_id=candidate["id"])
    write_json(path.with_suffix(".json"),dict(cache_source=cache_source,selection_committed=True,
        condition=condition,robot=r,frame_index=i,timestamp=float(p["t"][i]),
        valid_fraction=float(valid.mean()),n_points=len(pc),tr=float(p["tr"][i]),T=p["T"][i],
        depth_difference_mean=float(np.nanmean(np.abs(z-clean_z))),path=str(path)))
    return pc,sp,p["T"][i],float(p["tr"][i]),float(p["t"][i])


def evaluate(source,out,names):
    from covor.smoke import SmokeCfg
    from run_corridor import traversal as original_traversal,teammates
    cases=json.load(open(out/"cases_geometry.json"));refs={r["case_id"]:r for r in json.load(open(out/"references.json"))}
    old=json.load(open(source/"design.json"))
    candidates={c["id"]:c for c in json.load(open(out/"candidates.json"))["candidates"]}
    # Ground truth and actual candidate data enter this phase only.
    gt=np.load(ROOT/"results/synth_corridor915f/gt_voxel_fs.npz");ijk=gt["ijk_min"];res=float(gt["res"])
    scene,_=reference_scene();validator=ReferenceValidator(gt["labels"],ijk,res,scene)
    area_gt=original_traversal(gt["labels"],ijk,res)
    poses={r:dict(np.load(source/"poses"/f"{r}.npz")) for r in old["initial"]}
    all_rows=[];summaries=[];areas=[];replay=[];renderers={}
    for name in names:
        log=json.load(open(out/"selection"/f"{name}.json"))
        assert digest(source/f"map_{name}_initial.npz")==log["current_map_sha256"]
        initial=np.load(source/f"map_{name}_initial.npz")["labels"]
        variants={"initial":(initial,0,None)}
        smoke=SmokeCfg(**old["conditions"][name]);observation_cache={}
        for policy,cid in log["picks"].items():
            c=candidates[cid]
            if cid not in observation_cache:
                pc,sp,T,tr,t=observation(source,out,name,c,smoke,renderers)
                b=make_builder(old.get('args',{}).get('mapping_arm','depth_only'))
                b.tree=octomap.OcTree.read(str(source/f"initial_{name}.ot").encode())
                before=dense_map(sparse_map(b),ijk,gt["labels"].shape)
                assert np.array_equal(before,initial),"initial evidence clone differs"
                if len(pc):b.integrate_frame(T,tr,pc,sp,teammates=teammates(poses,c["robot"],t))
                b.finalize();lab=dense_map(sparse_map(b),ijk,initial.shape)
                observation_cache[cid]=lab
            variants[policy]=(observation_cache[cid],1,c)
        before_by_engine={}
        for policy,(lab,count,candidate) in variants.items():
            directory=out/"maps";directory.mkdir(exist_ok=True)
            snapshot=directory/f"{name}_{policy}.npz"
            np.savez_compressed(snapshot,labels=lab,ijk_min=ijk,res=res)
            g=DecisionGraph(lab,ijk,res,DecisionCfg(**cases["decision_cfg"]))
            decision=g.decide(cases["entry"],cases["goals"])
            engines={"observed_graph":decision["decisions"]}
            for engine,gap in (("legacy_confirmed",0.),("legacy_gap",.5)):
                rows,_=legacy_paths(lab,ijk,res,cases["entry"],cases["goals"],gap)
                engines[engine]=rows
            # Preserve original area metrics with original cleanup, snap and gap.
            area=TV.score(original_traversal(lab,ijk,res),area_gt)
            areas.append(dict(condition=name,policy=policy,additional_observations=count,**area))
            for engine,decisions in engines.items():
                rows=[]
                for goal,d in zip(cases["goals"],decisions):
                    path=motion_path(d["path"])
                    check=validator.validate(path,cases["entry"],goal) if d["verdict"]=="passable" else dict(status="not_proposed",reason="no_passable_verdict",first_failure=None)
                    pid=f"{name}_{policy}_{engine}_{goal['old_id']}"
                    pathfile=out/"paths"/f"{pid}.json"
                    write_json(pathfile,dict(case_id=goal["id"],entry=cases["entry"],goal=goal,
                        proposed_path=path,possible_path=d.get("possible_path",[]),
                        uncertain_segments=d.get("uncertain_segments",[]),validation=check,
                        transition_policy="shrink at departure; expand at arrival; same spatial route",
                        reference_transform=cases["reference_transform"]))
                    row=dict(case_id=goal["id"],condition=name,policy=policy,engine=engine,
                        verdict=d["verdict"],reason=d["reason"],reference_state=refs[goal["id"]]["reference_state"],
                        reference_reason=refs[goal["id"]]["reason"],path_status=check["status"],path_reason=check["reason"],
                        first_failure=json.dumps(check.get("first_failure")),path_file=str(pathfile),
                        n_path_points=len(path),gap_dependency=any(not p.get("observed_support",True) for p in path),
                        uncertain_segments=json.dumps(d.get("uncertain_segments",[])),
                        additional_observations=count,move_cost=0. if candidate is None else candidate["move_cost"],
                        candidate_id="" if candidate is None else candidate["id"],snapshot=str(snapshot))
                    rows.append(row)
                baseline=before_by_engine.get(engine)
                if policy=="initial":before_by_engine[engine]=rows
                summaries.append(dict(condition=name,policy=policy,engine=engine,additional_observations=count,
                    move_cost=rows[0]["move_cost"],fallback=log["reason"]["fallback"] if policy=="decision" else False,
                    candidate_id=rows[0]["candidate_id"],**metrics(rows,baseline)))
                all_rows.extend(rows)
                print("evaluate",name,policy,engine,"proposed",summaries[-1]["n_proposed"],
                      "valid",summaries[-1]["n_valid_paths"],"invalid",summaries[-1]["n_invalid_paths"],flush=True)
            replay.append(dict(condition=name,policy=policy,initial_source=str(source/f"initial_{name}.ot"),
                initial_frame_order=old["initial"],selected=candidate,additional_observations=count,
                snapshot=str(snapshot),actual_observation=None if candidate is None else str(out/"observations"/f"{name}_candidate{candidate['id']}.npz")))
            write_csv(out/"goals_paths.csv",all_rows);write_csv(out/"summary.csv",summaries);write_csv(out/"area_metrics.csv",areas)
            write_json(out/"replay.json",replay)
    write_json(out/"complete.json",dict(complete=True,conditions=names,n_goal_rows=len(all_rows),
        n_area_rows=len(areas),source_preserved=True,pose_scope="offline batch optimized; future data used in poses"))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",default="results/smoke_20261103/corridor")
    ap.add_argument("--out",required=True)
    ap.add_argument("--conditions",default="clean,uniform_weak,uniform_medium,uniform_strong,heterogeneous_weak,heterogeneous_medium,heterogeneous_strong")
    ap.add_argument("--stage",choices=("prepare","select","evaluate","all"),default="all")
    ap.add_argument("--max-move",type=float,default=6.)
    args=ap.parse_args();source=Path(args.source).resolve();out=Path(args.out).resolve()
    if out==source or source in out.parents:raise ValueError("use a separate new study output root")
    out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    names=args.conditions.split(",")
    if args.stage in ("prepare","all"):prepare(source,out)
    if args.stage in ("select","all"):select(source,out,names,args.max_move)
    if args.stage in ("evaluate","all"):evaluate(source,out,names)


if __name__=="__main__":main()
