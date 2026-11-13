#!/usr/bin/env python3
"""Read-only root-cause probes; oracle/sensitivity arms are NOT performance runs."""
import argparse
from dataclasses import replace
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"scripts/reobserve",ROOT/"scripts/smoke"):sys.path.insert(0,str(p))
import gtsam
import numpy as np
from scipy import ndimage
from covor.entry import traverse as TV,clean3d
from covor.entry.config import EntryCfg
from covor.entry.decision import DecisionGraph
from covor.entry.path_eval import ReferenceValidator
from common import write_csv,write_json,make_builder,sparse_map,dense_map,frame_points
from run import reference_scene

SOURCE=ROOT/"results/smoke_20261103/corridor"
STUDY=ROOT/"results/decision_paths_20261103"


def cases():return json.load(open(STUDY/"cases_geometry.json"))


def graph_stats(lab,ijk,res,name,goals,entry):
    graph=DecisionGraph(lab,ijk,res)
    start=graph.mission_nodes(entry,True);potential=graph.mission_nodes(entry)
    search=graph.search(start,True)
    reach=graph.area_result(search)
    pp=graph.search(potential)
    inside_goal=sum(bool(graph.mission_nodes(g,True)) for g in goals)
    reachable_goal=sum(any(n in search["best"] for n in graph.mission_nodes(g,True)) for g in goals)
    possible_goal=sum(any(n in pp["best"] for n in graph.mission_nodes(g)) for g in goals)
    # Existence of a locally observed support/body mode is weaker than global
    # entry-to-goal continuity; report both to avoid conflating them.
    row=dict(condition=name,n_confirmed_nodes=int(graph.confirmed.sum()),
        n_possible_nodes=int(graph.possible.sum()),entry_confirmed_nodes=len(start),
        entry_possible_nodes=len(potential),entry_component_nodes=len(search["best"]),
        entry_component_columns=int(reach.sum()),n_goals=len(goals),
        goals_with_local_confirmed_nodes=inside_goal,confirmed_goals=reachable_goal,
        possible_goals=possible_goal)
    z=(np.arange(lab.shape[2])+ijk[2]+.5)*res
    band=(z>=-.2)&(z<=.4)
    row["observed_floor_support_columns"]=int(graph.support[:,:,band].any(2).sum())
    row["confirmed_floor_body_columns"]=int(graph.confirmed[:,:,band].any(2).sum())
    # At the real GT floor's voxel (world Z index 0), distinguish occupied
    # support from ray-carved free and unknown. Region is fixed southern path.
    x=(np.arange(lab.shape[0])+ijk[0]+.5)*res
    y=(np.arange(lab.shape[1])+ijk[1]+.5)*res
    region=(x[:,None]>=-2.8)&(x[:,None]<=-1.5)&(y[None,:]>=-19.5)&(y[None,:]<=-15.5)
    k=-int(ijk[2])
    for label,title in ((0,"unknown"),(1,"free"),(2,"occupied")):
        row["south_floor_"+title]=int(((lab[:,:,k]==label)&region).sum())
    detailed=[]
    for goal in goals:
        nodes=graph.mission_nodes(goal,True)
        detailed.append(dict(condition=name,goal=goal["id"],local_confirmed_nodes=len(nodes),
                             entry_connected=sum(n in search["best"] for n in nodes),
                             local_possible_nodes=len(graph.mission_nodes(goal))))
    return row,detailed


def graphs(out):
    cfg=cases();rows=[];detail=[]
    names=json.load(open(SOURCE/"design.json"))["conditions"]
    for name in names:
        q=np.load(SOURCE/f"map_{name}_initial.npz")
        row,d=graph_stats(q["labels"],q["ijk_min"],float(q["res"]),name,cfg["goals"],cfg["entry"])
        rows.append(row);detail.extend(d)
        cleaned,_=clean3d.clean(q["labels"],EntryCfg())
        cleanrow,cd=graph_stats(cleaned,q["ijk_min"],float(q["res"]),name+"_cleanup_diagnostic",cfg["goals"],cfg["entry"])
        cleanrow["new_occupied_by_cleanup"]=int(((cleaned==2)&(q["labels"]!=2)).sum())
        rows.append(cleanrow);detail.extend(cd)
        print("graph",row,flush=True)
    write_csv(out/"graph_bottlenecks.csv",rows);write_csv(out/"goal_connectivity.csv",detail)
    selected=[]
    for p in sorted((STUDY/"selection").glob("*.json")):
        s=json.load(open(p));positive=[r for r in s["segments"] if r["importance"]]
        pick=s["picks"]["decision"];scores=next(r for r in s["scores"] if r["id"]==pick)
        selected.append(dict(condition=s["condition"],n_segments=len(s["segments"]),
            n_positive_segments=len(positive),n_influenced_goals=len({g for r in positive for g in r["affected_goals"]}),
            selected=pick,fallback=s["reason"]["fallback"],priority=scores["priority"],
            contribution=json.dumps(scores["segments"])))
    write_csv(out/"selection_bottlenecks.csv",selected)


def paths(out):
    cfg=cases();goals={g["id"]:g for g in cfg["goals"]}
    gt=np.load(ROOT/"results/synth_corridor915f/gt_voxel_fs.npz")
    scene,_=reference_scene();validator=ReferenceValidator(gt["labels"],gt["ijk_min"],.1,scene)
    import csv
    source=list(csv.DictReader(open(STUDY/"goals_paths.csv")))
    # Deduplicate same spatial/mode route for computation, retain every row in
    # the results. A common prefix is not many independent wall-crossing errors.
    cache={};results=[]
    for r in source:
        if r["path_status"]=="not_proposed":continue
        q=json.load(open(r["path_file"]));path=q["proposed_path"]
        for shift in (0.,.05,.1):
            key=(hashlib.sha256(json.dumps(path,sort_keys=True).encode()).hexdigest(),shift,r["case_id"])
            if key not in cache:
                altered=[dict(p,xyz=[p["xyz"][0],p["xyz"][1],p["xyz"][2]+shift]) for p in path]
                cache[key]=validator.validate(altered,cfg["entry"],goals[r["case_id"]])
            check=cache[key]
            results.append(dict(condition=r["condition"],policy=r["policy"],engine=r["engine"],
                case_id=r["case_id"],diagnostic_common_z_shift_m=shift,status=check["status"],
                reason=check["reason"],first_failure=json.dumps(check.get("first_failure")),
                failure_path_index=check.get("index",0),n_path_points=len(path),
                original_status=r["path_status"],original_reason=r["path_reason"]))
    write_csv(out/"path_z_sensitivity.csv",results)
    counters=[]
    for engine in ("legacy_confirmed","legacy_gap"):
        for shift in (0.,.05,.1):
            rr=[r for r in results if r["engine"]==engine and r["diagnostic_common_z_shift_m"]==shift]
            counters.append(dict(engine=engine,shift=shift,n_queries=len(rr),
                valid=sum(r["status"]=="valid" for r in rr),invalid=sum(r["status"]=="invalid" for r in rr),
                unverifiable=sum(r["status"]=="unverifiable" for r in rr),
                first_point_failure=sum(r["status"]=="invalid" and r["failure_path_index"]==0 for r in rr),
                reason_counts=json.dumps(dict(collections.Counter(r["reason"] for r in rr)))))
    write_csv(out/"path_z_summary.csv",counters)
    example=next(r for r in source if r["condition"]=="clean" and r["policy"]=="initial"
                 and r["engine"]=="legacy_confirmed" and r["case_id"].endswith("south_0"))
    path=json.load(open(example["path_file"]))["proposed_path"]
    entry_checks=[]
    for shift in (0.,.05,.1):
        for posture in (0,1,2):
            altered=[dict(p,xyz=[p["xyz"][0],p["xyz"][1],p["xyz"][2]+shift],
                          posture=max(posture,p["posture"]),width=1) for p in path]
            check=validator.validate(altered,cfg["entry"],goals[example["case_id"]])
            entry_checks.append(dict(shift_z=shift,posture_at_least=posture,**check))
    write_csv(out/"entry_pose_sensitivity.csv",entry_checks)
    print("path sensitivity",counters,flush=True)


def oracle(out):
    from covor.synth.render import DepthRenderer
    from covor.synth import mesh_gt as MG
    from covor.occupancy import load_stereo_calib
    from covor import data as D
    from run_corridor import teammates
    cfg=cases();old=json.load(open(SOURCE/"design.json"))
    poses={r:dict(np.load(SOURCE/"poses"/f"{r}.npz")) for r in old["initial"]}
    truth={}
    for r,P in poses.items():
        Tbody=P["T_truth"]@np.linalg.inv(D.load_body_T_cam(r,0))
        truth[r]=dict(t=P["t"],p_body=Tbody[:,:3,3])
    scene,_=reference_scene()
    arms={name:make_builder() for name in ("sgbm_true_pose","ideal_true_pose","ideal_estimated_pose")}
    gt=np.load(ROOT/"results/synth_corridor915f/gt_voxel_fs.npz");ijk=gt["ijk_min"];res=float(gt["res"])
    pose_rows=[]
    for r,P in poses.items():
        errors=np.linalg.norm(P["T"][:,:3,3]-P["T_truth"][:,:3,3],axis=1)
        pose_rows.append(dict(robot=r,absolute_position_rmse_m=float(np.sqrt(np.mean(errors**2))),
                             median=float(np.median(errors)),p90=float(np.percentile(errors,90))))
        renderer=DepthRenderer(scene,r)
        for n,i in enumerate(old["initial"][r]):
            c=np.load(SOURCE/"input_cache"/f"{r}_{i:05d}.npz")
            assert np.array_equal(c["T_truth"],P["T_truth"][i])
            pc,sz=frame_points(renderer.sd,c["z"],c["sz"],c["valid"])
            if pc is not None:arms["sgbm_true_pose"].integrate_frame(P["T_truth"][i],0,pc,sz,teammates=teammates(truth,r,P["t"][i]))
            pc,sz=renderer.frame(P["T_truth"][i],mode="ideal")
            if pc is not None:
                arms["ideal_true_pose"].integrate_frame(P["T_truth"][i],0,pc,sz,teammates=teammates(truth,r,P["t"][i]))
                arms["ideal_estimated_pose"].integrate_frame(P["T"][i],float(P["tr"][i]),pc,sz,teammates=teammates(poses,r,P["t"][i]))
            if (n+1)%25==0:print("oracle",r,n+1,flush=True)
    summaries=[];details=[]
    for name,b in arms.items():
        b.finalize();lab=dense_map(sparse_map(b),ijk,gt["labels"].shape)
        np.savez_compressed(out/f"map_{name}.npz",labels=lab,ijk_min=ijk,res=res)
        raw,d=graph_stats(lab,ijk,res,name,cfg["goals"],cfg["entry"])
        summaries.append(raw);details.extend(d)
        from run_corridor import traversal
        a=traversal(lab,ijk,res);target=traversal(gt["labels"],ijk,res)
        raw.update(TV.score(a,target))
        print("oracle result",name,raw,flush=True)
    write_csv(out/"pose_errors.csv",pose_rows);write_csv(out/"oracle_maps.csv",summaries)
    write_csv(out/"oracle_goal_connectivity.csv",details)


def candidates(out):
    from covor.occupancy import StereoDepth,load_stereo_calib
    rows=[]
    for path in sorted((STUDY/"selection").glob("*.json")):
        log=json.load(open(path));name=log["condition"]
        winner=next(r for r in log["scores"] if r["id"]==log["picks"]["decision"])
        if not winner["segments"]:continue
        obs=np.load(STUDY/"observations"/f"{name}_candidate{winner['id']}.npz")
        before=np.load(SOURCE/f"map_{name}_initial.npz")
        after=np.load(STUDY/"maps"/f"{name}_decision.npz")["labels"]
        graph=np.load(STUDY/"graphs"/f"{name}.npz")
        T=obs["T_est"];points=obs["P_cam"]
        sd=StereoDepth(load_stereo_calib(winner["robot"]))
        points=points[np.linalg.norm(points,axis=1)<=5.]
        vox=np.floor((points@T[:3,:3].T+T[:3,3])/.1).astype(int)-before["ijk_min"]
        endpoints={tuple(p) for p in vox}
        post=DecisionGraph(after,before["ijk_min"])
        for region in winner["segments"]:
            feet=np.argwhere(graph["segments"]==region["id"])
            existing=before["labels"][tuple(feet.T)];new=after[tuple(feet.T)]
            hits=sum(tuple(p) in endpoints for p in feet)
            near=sum(any(tuple(p+np.array([0,0,k])) in endpoints for k in (-1,0,1)) for p in feet)
            surfaces=(feet+before["ijk_min"]+.5)*.1
            surfaces[:,2]+=.05+1e-5
            camera=(surfaces-T[:3,3])@T[:3,:3]@sd.R1.T
            z=camera[:,2]
            with np.errstate(divide="ignore",invalid="ignore"):
                u=sd.fx*camera[:,0]/z+sd.cx;v=sd.fy*camera[:,1]/z+sd.cy
            fov=(z>.3)&(np.linalg.norm(surfaces-T[:3,3],axis=1)<=5)&(u>=0)&(u<sd.calib.size[0])&(v>=0)&(v<sd.calib.size[1])
            rows.append(dict(condition=name,segment=region["id"],n_foot_nodes=len(feet),
                foot_hypothesis_z_min=float(surfaces[:,2].min()),foot_hypothesis_z_max=float(surfaces[:,2].max()),
                n_support_in_frustum=int(fov.sum()),
                expected_support_visibility=region["support"]["fraction"],
                expected_body_visibility=region["body"]["fraction"],
                actual_foot_endpoint_hits=hits,actual_endpoint_hits_within_one_z_voxel=near,
                foot_unknown_before=int((existing==0).sum()),foot_occupied_before=int((existing==2).sum()),
                foot_unknown_after=int((new==0).sum()),foot_occupied_after=int((new==2).sum()),
                newly_confirmed_nodes=int(post.confirmed[tuple(feet.T)].sum())))
    write_csv(out/"candidate_realization.csv",rows)
    print("candidate realization",rows,flush=True)


def poses(out):
    from covor.fusion import umeyama_yaw,Robot,Cfg
    from covor.synth.config import SynthCfg
    from covor.synth import dataset as DS
    from covor import data as D
    sys.path.insert(0,str(ROOT/"scripts/synth"))
    from fuse_synth import FAITHFUL
    cfg=SynthCfg(name="corridor915f",seq="synth_corridor915f_0")
    DS.install_paths(cfg)
    tracks={p.stem:dict(np.load(p)) for p in sorted((SOURCE/"poses").glob("*.npz"))}
    all_est=np.concatenate([p["T"][:,:3,3] for p in tracks.values()])
    all_gt=np.concatenate([p["T_truth"][:,:3,3] for p in tracks.values()])
    R,t=umeyama_yaw(all_est,all_gt)
    rows=[]
    for k,(name,p) in enumerate(tracks.items()):
        est=p["T"][:,:3,3];gt=p["T_truth"][:,:3,3];delta=est-gt
        own_R,own_t=umeyama_yaw(est,gt)
        robot=Robot(cfg.seq,name,k,Cfg(**FAITHFUL))
        robot.align_to_world()
        initial=np.array([(q.matrix()@D.load_body_T_cam(name,0))[:3,3] for q in robot.init_world])
        assert len(initial)==len(est) and np.allclose(robot.t,p["t"],atol=1e-6)
        initial_delta=initial-gt
        rows.append(dict(robot=name,raw_rmse=float(np.sqrt(np.mean(np.sum(delta**2,1)))),
            mean_dx=float(delta[:,0].mean()),mean_dy=float(delta[:,1].mean()),mean_dz=float(delta[:,2].mean()),
            common_aligned_rmse=float(np.sqrt(np.mean(np.sum((est@R.T+t-gt)**2,1)))),
            individual_aligned_rmse=float(np.sqrt(np.mean(np.sum((est@own_R.T+own_t-gt)**2,1)))),
            initial_rmse=float(np.sqrt(np.mean(np.sum(initial_delta**2,1)))),initial_mean_dz=float(initial_delta[:,2].mean())))
    write_csv(out/"pose_frame_decomposition.csv",rows)
    write_json(out/"common_alignment_diagnostic.json",dict(R=R,t=t,scope="diagnostic only; not applied to benchmark maps"))
    gt=DS.load_gt_traj(cfg);geometry=[]
    for a,b in (("ifo001","ifo002"),("ifo001","ifo003"),("ifo002","ifo003")):
        query=np.linspace(0,120,1201)
        pa=np.stack([np.interp(query,gt[a]["t"],gt[a]["p"][:,i]) for i in range(3)],1)
        pb=np.stack([np.interp(query,gt[b]["t"],gt[b]["p"][:,i]) for i in range(3)],1)
        d=pa-pb;distance=np.linalg.norm(d,axis=1);sensitivity=np.abs(d[:,2])/distance
        geometry.append(dict(pair=a+"_"+b,median_range_m=float(np.median(distance)),
            median_abs_vertical_fraction=float(np.median(sensitivity)),
            scope="body-centre single-range vertical derivative magnitude; not a full observability/covariance test"))
    write_csv(out/"range_geometry.csv",geometry)
    print("pose decomposition",rows,"geometry",geometry,flush=True)


def floor_trace(out):
    from covor.synth.render import DepthRenderer
    from covor.occupancy import _dda_batch
    import open3d as o3d
    ideal=np.load(out/"map_ideal_true_pose.npz")
    gt=np.load(ROOT/"results/synth_corridor915f/gt_voxel_fs.npz")
    ijk=ideal["ijk_min"];res=.1
    entry=np.array([-2.35,-19.55,.05]);index=np.floor(entry/res).astype(int)-ijk
    choices=[]
    for di in range(-3,4):
        for dj in range(-3,4):
            cell=index+np.array([di,dj,0])
            if ideal["labels"][tuple(cell)]==1 and gt["labels"][tuple(cell)]==2:
                choices.append((di*di+dj*dj,tuple(cell)))
    if not choices:raise RuntimeError("no local false-free floor cell to trace")
    cell=np.array(min(choices)[1]);world=cell+ijk;centre=(world+.5)*res
    scene,_=reference_scene()
    hit=scene.cast_rays(o3d.core.Tensor(np.array([[centre[0],centre[1],.5,0,0,-1]],np.float32)))["t_hit"].numpy()[0]
    rows=[];free_total=occ_total=0.;nfree=nocc=0;logodds=0.
    old=json.load(open(SOURCE/"design.json"))
    for r,ids in old["initial"].items():
        P=dict(np.load(SOURCE/"poses"/f"{r}.npz"));renderer=DepthRenderer(scene,r)
        for i in ids:
            T=P["T_truth"][i]
            if np.linalg.norm(T[:3,3]-centre)>5.2:continue
            pc,sz=renderer.frame(T,mode="ideal")
            if pc is None:continue
            points=pc@T[:3,:3].T+T[:3,3]
            length=np.linalg.norm(points-T[:3,3],axis=1)
            good=(length>res)&(length<=5);points=points[good];sz=sz[good]
            weights=np.exp(-sz**2/.5)
            ends=np.floor(points/res).astype(int)
            terminal=(ends==world).all(1)
            ray,free=_dda_batch(T[:3,3],points,res)
            crossing=(free==world).all(1)
            nf=int(crossing.sum());no=int(terminal.sum())
            if nf+no==0:continue
            dl_free=float((weights[ray[crossing]]*-.4).sum())
            dl_occ=float((weights[terminal]*.85).sum())
            logodds=float(np.clip(logodds+dl_free+dl_occ,-2,3.5))
            free_total+=dl_free;occ_total+=dl_occ;nfree+=nf;nocc+=no
            rows.append(dict(robot=r,index=i,timestamp=float(P["t"][i]),n_free_rays=nf,n_endpoint_rays=no,
                             free_delta=dl_free,occupied_delta=dl_occ,replayed_logodds=logodds))
    write_csv(out/"floor_evidence_frames.csv",rows)
    result=dict(world_voxel=world,voxel_centre=centre,reference_mesh_surface_z=float(.5-hit),
       gt_label=int(gt["labels"][tuple(cell)]),oracle_map_label=int(ideal["labels"][tuple(cell)]),
       total_free_rays=nfree,total_occupied_endpoints=nocc,free_evidence_sum=free_total,
       occupied_evidence_sum=occ_total,replayed_final_logodds=logodds,
       interpretation="A voxel contains both floor surface and traversed air; summed free evidence can erase observed support even with ideal geometry and poses")
    write_json(out/"floor_evidence_trace.json",result)
    print("floor trace",result,flush=True)


def report(out):
    import csv,shutil
    target=ROOT/"reports/diagnosis_20261104";target.mkdir(parents=True,exist_ok=True)
    for p in out.iterdir():
        if p.suffix in (".csv",".json"):shutil.copy2(p,target/p.name)
    def read(name):return list(csv.DictReader(open(out/name)))
    def table(rows,keys):
        def f(x):
            try:
                n=float(x);return str(int(n)) if n.is_integer() else f"{n:.4f}"
            except (ValueError,TypeError):return str(x)
        return "\n".join(["| "+" | ".join(keys)+" |","|"+"---|"*len(keys)]+
            ["| "+" | ".join(f(r.get(k,"")) for k in keys)+" |" for r in rows])
    oracle_rows=read("oracle_maps.csv")
    graph_rows=read("graph_bottlenecks.csv")
    pose_rows=read("pose_frame_decomposition.csv")
    floor=json.load(open(out/"floor_evidence_trace.json"))
    text=f'''# 현재 성능 저하의 원인 진단 — 2026-11-04

현재 `corridor915f`, C_3drone anchor-free, seed 0, 초기 3대×100프레임에 대한 진단이다.
기존 실험의 지도·포즈·후보·경로는 보존했다. 운영 알고리즘이나 판정 문턱은 바꾸지 않았다.
원인 분리를 위해 참 포즈/이상 깊이 oracle과 제시 경로의 높이·자세 민감도를 별도 실행했다.
이 결과는 진단 대조이며 기존 성능 결과나 후보 선택의 입력으로 사용하지 않는다.

## 1. 확인된 두 개의 독립 병목

**지도 정합 오차가 기존 면적 성능을 크게 떨어뜨리고, 점유 복셀을 발판으로 직접 쓰는 방식이 새 확인 경로를 끊는다.**
무처리 SGBM+추정 포즈의 기존 면적 recall은 0.081168, IoU는 0.080797이다.
같은 SGBM 관측을 참 포즈에 적분한 대조에서는 recall 0.967060, IoU 0.911434였다.
반면 참 포즈와 이상적 메시 깊이를 모두 써도 새 확인 그래프의 목표 연결은 0/22다.
따라서 포즈 정합만 고쳐도 새 확인 지표가 자동으로 회복된다고 볼 수 없다.

{table(oracle_rows,['condition','reach_recall','reach_iou','false_reach_rate','entry_confirmed_nodes','entry_component_columns','confirmed_goals','possible_goals'])}

표의 recall/IoU/false-reach는 기존 cleanup·gap 규약의 **면적 지표**, confirmed_goals는 raw 관측 증거를
요구하는 새 그래프의 목표 연결 수다. oracle은 같은 300프레임·격자·가중식을 사용했다.
참 포즈 대조는 모든 드론의 자세와 동료 마스킹 좌표도 참 좌표로 일관되게 바꾼 상한 대조다.
오도달률도 달라지므로 이 값을 실제 시스템의 달성 성능으로 채택하지 않는다.

## 2. 후단이 드론 사이의 상대 고도를 크게 바꾼다

{table(pose_rows,['robot','initial_rmse','initial_mean_dz','raw_rmse','mean_dz','individual_aligned_rmse','common_aligned_rmse'])}

최적화 전 카메라 초기값의 평균 Z 오차는 약 0이었지만, 최적화 후 ifo002는 +0.510 m,
ifo003은 +1.389 m로 변했다. ifo001은 +0.026 m다. 초기값은 기존 전 궤적 GT 정렬을 사용한 것이므로
초기값 성능을 실제 배치 가능한 VIO 기준선으로 해석하지 않는다. 여기서는 최적화가 상대 높이를
얼마나 이동했는지만 분리해 확인했다.

드론별로 다시 정렬한 ATE 0.121/0.093/0.085 m는 이 상수 정합 오차를 상당 부분 흡수한다.
공통 yaw+이동 변환 하나로 정렬해도 각 RMSE는 0.629/0.174/0.753 m다.
따라서 지도 전체의 공통 원점 이동만으로 해결되는 오차가 아니다.

왜 그런 해가 나오는가에 대해서는 **긴 복도의 약한 수직 거리 기하와 높이 제약 부재**가 유력하다.
실제 GT 몸 중심 거리의 수직 민감도 |Δz|/거리 중앙값은 약 0.0092–0.0160이다.
드론 간 거리는 약 8.4–24.9 m인 데 비해 고도 차이는 작고, 중력 prior는 자세의 기울기를 관측하며
Z 위치를 직접 고정하지 않는다. 현재 조건은 앵커·높이 팩터도 없다.
이는 측정한 기하와 설정에서의 해석이며 전체 그래프의 관측성 rank 검정이나 Jacobian 결함 검정을
대체하지는 않는다. 포즈 평균 Z 이동 자체와 oracle 지도 차이는 직접 확인한 사실이다.

## 3. 완벽한 센서에서도 free 증거가 바닥 표면을 지운다

추적한 세계 복셀은 {floor['world_voxel']}, 중심 {floor['voxel_centre']} m다.
메시 바닥은 Z={floor['reference_mesh_surface_z']:.6f} m에 있고 GT는 occupied지만,
**참 포즈+이상 깊이로 만든 지도도 이 셀을 free로 분류**했다.

- 점유 종단 광선 {floor['total_occupied_endpoints']}개: log-odds 합 +{floor['occupied_evidence_sum']:.4f}
- free 통과 광선 {floor['total_free_rays']}개: log-odds 합 {floor['free_evidence_sum']:.4f}
- 프레임 순서대로 재생한 최종 log-odds {floor['replayed_final_logodds']:.4f} → free

원인은 같은 0.10 m 복셀에 얇은 바닥 표면과 그 위의 공기가 함께 들어가기 때문이다.
다른 바닥 지점으로 향하는 광선이 이 복셀의 공기 부분을 지나면서 free 증거를 더한다.
현재 점유지도는 부피 점유를 위한 합산 구조이고, 새 판정기는 그 셀이 occupied일 때만 발판으로 인정한다.
즉, **바닥을 보았다는 증거가 있는데도 발판 판정에는 남지 않는 표현의 불일치**가 있다.
새 가능성 그래프는 free 발판을 후보로 넣지 않으므로 이렇게 지워진 바닥에서 확정·가능 경로가 함께 끊긴다.
이는 SGBM 오차만으로 설명되지 않는다.

무처리 지도에서 지정 입구의 확인 연결 성분은 16셀이다. 중간 균일·중간 비균일·강한 비균일 조건은
입구의 확인 노드 자체가 0개다. 다음 표는 raw 지도와 cleanup을 추가한 진단을 함께 보인다.
cleanup으로 늘어난 지지를 관측으로 확인된 발판으로 채택하지는 않았다.

{table(graph_rows,['condition','entry_confirmed_nodes','entry_component_columns','goals_with_local_confirmed_nodes','confirmed_goals','possible_goals'])}

## 4. 재관측이 필요한 증거를 거의 추가하지 못한다

양의 중요도가 있는 실제 구간은 조건별 0–2개뿐이고, 7조건 중 4조건은 nearest 대체를 사용했다.
선택된 중요 구간의 몸 공간 가시성은 약 94–98%지만 주요 발판 가시성은 3–18%다.
주요 발판들은 시야각 안에는 들어왔다. 낮은 비율의 상당 부분은 관측된 셀에 의한 가림이다.
이 발판 높이 가설들은 약 −0.20…0.00 m로, 입구의 실제 바닥 약 +0.05 m보다 아래에 있다.
현재 지도의 바닥 높이 오차와 이웃 ±2행 추론이 아래쪽 발판 가설을 만들고,
이 가설을 전부 같은 관측 대상으로 평균 내는 것도 점수의 물리적 의미를 약하게 한다.

{table(read('candidate_realization.csv'),['condition','segment','n_foot_nodes','n_support_in_frustum','expected_support_visibility','expected_body_visibility','actual_foot_endpoint_hits','foot_unknown_before','foot_unknown_after','newly_confirmed_nodes'])}

예를 들어 약한 균일 조건은 미확인 발판 41개가 관측 뒤에도 41개였고,
약한 비균일 조건의 주요 구간도 97→88개로 일부만 줄었다. 단일 프레임은 국소 증거를 개선했지만
입구부터 목표까지의 모든 연결을 확인하지는 못했다. 중요도는 조건부 연결 의존도이지
그 프레임을 추가하면 유효 경로가 완성될 확률이 아니다.
현재 후보 점수에는 실제 스테레오 성공률이나 갱신 후 log-odds 문턱 통과까지의 예측도 없다.

## 5. 경로 검증의 100% 무효는 대부분 같은 출발점 실패다

기존 판정기의 제안 204건은 각각 독립적인 204개 장애물 통과 실패가 아니다.
132건은 첫 경로점의 몸 충돌, 72건은 입구/층 조건 실패로 끝났으며 여러 목표가 같은 시작 자세를 공유했다.
무처리 대표 경로는 발 Z=0.0 m를 제시하지만 실제 바닥은 약 0.0501 m다.
높이만 +0.05 m 옮기면 이번에는 약 Z=1.95 m의 입구 상부 구조물에 서 있는 몸이 충돌했다.
같은 XY 경로에 +0.05 m와 숙이기·옆으로 모드를 적용한 **진단**에서는 첫 목표 경로가 유효해졌다.

{table(read('entry_pose_sensitivity.csv'),['shift_z','posture_at_least','status','reason'])}

이는 바닥 높이와 자세 선택이 함께 틀린 사례다. 단순히 허용오차를 키우거나 좌표를 일괄 이동하는
것으로 모든 경로를 고칠 수 없다. 높이/자세 변경은 사후 민감도 검사이며 원래 경로를 대체하거나
기존 성공률에 넣지 않았다. 현재 실험에서는 이 공통 입구 실패가 다른 후보 정책의 효과를 가린다.

## 6. 평가 표본의 한계는 별도다

22개 목표 중 독립 메시 검증으로 도달 가능한 것은 10개, 기준 판정 불가는 12개였고
확정된 차단 목표는 0개다. 따라서 잘못된 경로를 취소하는 능력이나 차단 판정 정확도까지
일반화할 표본이 부족하다. 이 문제는 지도의 물리적 실패 원인과 구분해야 한다.

## 7. 수정 우선순위

1. **후단의 드론 간 상대 높이부터 확인한다.** 로봇별 정렬 ATE와 함께 공통 프레임 오차·평균 Z 오차를
   회귀 지표로 삼고, 초기/최종 상대 높이·거리 Jacobian·게이지·높이 관련 제약을 검사한다.
   관측 가능한 시작/지면 정보를 어떤 제약으로 쓸지는 실제 운용 조건에 맞춰 별도 실험해야 한다.
2. **발판/바닥 증거를 부피 점유와 분리한다.** 표면 높이·지지 증거와 불확실성을 유지하는 표현을 검토한다.
   free 바닥을 임의로 occupied로 바꾸거나 미확인을 통과 가능으로 완화하는 처방은 진단에서 검증하지 않았다.
3. **입구의 실제 바닥·상부 여유와 제시 자세를 일관되게 만든다.** 발판 높이, 접촉 허용오차,
   몸 충돌의 의미를 정의하고 양자화된 XYZ가 현장 표면과 어느 정도 일치해야 하는지 검증한다.
4. **그다음 재관측 정책을 비교한다.** 입구와 일부 목표의 유효 경로가 존재하고 한 국소 구간의
   미확인만 남은 고정 사례에서, 한 프레임으로 실제 발판 증거가 늘어나는 후보를 포함한다.
   경로 의존도 점수와 예상 관측 후 확정 이득은 다른 양이므로 각각 검증한다.

## 재실행과 보존

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=python
O=results/diagnosis_20261104
$PY scripts/reobserve/diagnose.py --out $O --stage graphs
$PY scripts/reobserve/diagnose.py --out $O --stage poses
$PY scripts/reobserve/diagnose.py --out $O --stage oracle
$PY scripts/reobserve/diagnose.py --out $O --stage paths
$PY scripts/reobserve/diagnose.py --out $O --stage candidates
$PY scripts/reobserve/diagnose.py --out $O --stage floor_trace
$PY scripts/reobserve/diagnose.py --out $O --stage report
```

oracle 3D 격자는 `{out.resolve()}`에, 진단 CSV·JSON은 이 보고서 옆에 있다.
운영 파라미터·후보 선택·기존 성능 CSV는 이번 진단으로 변경하지 않았다.
새 코드는 `scripts/reobserve/diagnose.py`다.
'''
    (target/"RESULTS.md").write_text(text)
    write_json(target/"artifacts.json",{str(p.relative_to(target)):dict(bytes=p.stat().st_size,sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest())
         for p in sorted(target.rglob("*")) if p.is_file() and p.name!="artifacts.json"})
    print(target/"RESULTS.md")


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--out",required=True)
    ap.add_argument("--stage",choices=("graphs","paths","oracle","candidates","poses","floor_trace","report"),required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    globals()[args.stage](out)


if __name__=="__main__":main()
