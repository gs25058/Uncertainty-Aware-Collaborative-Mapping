#!/usr/bin/env python3
"""Recorded MILUV frames only; no independent target/path ground truth claims."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"scripts",ROOT/"scripts/smoke"):sys.path.insert(0,str(p))
import cv2,numpy as np,octomap
import build_occupancy as original
from covor.occupancy import StereoDepth,load_stereo_calib
from covor.synth.render import _depth_rectified
from covor.smoke import SmokeCfg,stereo_smoke,clean_right_depth
from covor.entry.decision import DecisionGraph
from covor.entry.assignment import choose
from common import make_builder,sparse_map,dense_map,write_json,write_csv,digest,map_metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="results/smoke_20261103/miluv")
    ap.add_argument("--out",required=True)
    args=ap.parse_args();src=Path(args.source);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    manifest=json.load(open(src/"manifest.json"));lo=np.array(manifest["map_domain"]["ijk_min"]);shape=manifest["map_domain"]["shape"]
    arm=manifest.get('args',{}).get('mapping_arm','depth_only')
    frames=manifest["frames"];robots=sorted({f["robot"] for f in frames})
    poses={r:dict(np.load(Path(original.VO)/f"occ_{original.SEQ}_{r}_af.npz")) for r in robots}
    sd={r:StereoDepth(load_stereo_calib(r)) for r in robots}
    mates={r:original.Teammates(r,robots,in_tag="_af") for r in robots}
    configs={"clean":SmokeCfg(),"heterogeneous_medium":SmokeCfg(beta=.4,field="heterogeneous",seed=0)}
    builders={k:make_builder(arm) for k in configs}
    # Prior run saved labels but no full-evidence .ot or per-frame smoke depths.
    # Rebuild just this missing stage and assert exact old labels afterwards.
    for n,f in enumerate(frames):
        r=f["robot"];i=f["pose_index"];P=poses[r]
        raw=[cv2.imread(f[k],0) for k in ("left","right")]
        left,right=sd[r].rectify_pair(*raw);z,sz,v=_depth_rectified(sd[r],left,right)
        zr=clean_right_depth(sd[r],left,right)
        for name,cfg in configs.items():
            if cfg.beta:
                il,ir,_=stereo_smoke(sd[r],left,right,z,zr,P["T"][i],float(P["t"][i]),cfg)
                Z,S,V=_depth_rectified(sd[r],il,ir)
            else:Z,S,V=z,sz,v
            if V.sum()>=100:
                pc,sp=sd[r].backproject(Z,S,V,downsample=4)
                builders[name].integrate_frame(P["T"][i],float(P["tr_sigma_pos"][i]),pc,sp,teammates=mates[r].at(P["t"][i]))
        if (n+1)%24==0:print("MILUV initial evidence",n+1,"/",len(frames),flush=True)
    initial={}
    for name,b in builders.items():
        b.finalize();lab=dense_map(sparse_map(b),lo,shape)
        old=np.load(src/f"map_{name}.npz")["labels"]
        assert np.array_equal(lab,old),"rebuild differs from prior fixed-input map"
        initial[name]=lab;b.tree.write(str(out/f"initial_{name}.ot").encode())
    clean_graph=DecisionGraph(initial["clean"],lo)
    known=np.argwhere(clean_graph.confirmed)
    # Explicitly virtual mission patches: no independent rescue-entry annotation
    # exists in MILUV. Regions are derived once from CURRENT clean map evidence.
    if len(known):
        start=known[np.lexsort((known[:,2],known[:,1],known[:,0]))[0]]
        targets=known[np.linspace(0,len(known)-1,6).astype(int)]
    else:
        camera=poses[robots[0]]["T"][frames[0]["pose_index"],:3,3]
        start=np.floor(camera/.1).astype(int)-lo;start[2]=np.clip(-lo[2],0,shape[2]-10)
        targets=np.array([start+np.array([d,0,0]) for d in (0,5,10,15,20,25)])
    def patch(p,ident):
        xyz=(p+lo+.5)*.1;foot=(p[2]+lo[2]+1)*.1
        return dict(id=ident,xy_bounds=[xyz[0]-.04,xyz[1]-.04,xyz[0]+.04,xyz[1]+.04],floor_z_bounds=[foot-.05,foot+.05])
    entry=patch(start,"virtual_entry");goals=[patch(p,f"virtual_goal_{i}") for i,p in enumerate(targets)]
    candidates=[]
    for r,P in poses.items():
        initial_ids={f["pose_index"] for f in frames if f["robot"]==r};last=max(initial_ids)
        pool=np.array([i for i in range(8,last+1,16) if i not in initial_ids])
        index=original.FrameIndex(r)
        for i in pool[np.linspace(0,len(pool)-1,8).astype(int)]:
            timestamp=index.pair(P["t"][i])
            if timestamp is None:continue
            candidates.append(dict(id=len(candidates),robot=r,index=int(i),timestamp=float(P["t"][i]),image_timestamp=float(timestamp),
                                  T=P["T"][i].tolist(),move_cost=float(np.linalg.norm(P["T"][i,:3,3]-P["T"][last,:3,3]))))
    write_json(out/"design.json",dict(entry=entry,goals=goals,candidates=candidates,
        source_manifest_sha256=digest(src/"manifest.json"),scope="recorded IR replay; virtual mission patches; no independent target/path truth",
        poses="retrospective batch-fused; future data in optimization",rebuilt_initial_labels_exact=True))
    selections={}
    for name,lab in initial.items():
        graph=DecisionGraph(lab,lo);analysis=graph.decide(entry,goals,True)
        picks,scores,reason=choose(graph,analysis,candidates,sd,max_move_m=6.)
        write_json(out/f"selection_{name}.json",dict(picks=picks,scores=scores,reason=reason,candidate_images_read=False))
        selections[name]=(picks,reason,analysis)
    # Actual recorded candidate pixels are read only after BOTH selections exist.
    rows=[];goal_rows=[]
    for name,cfg in configs.items():
        picks,reason,_=selections[name]
        variants={"initial":(initial[name],None)};done={}
        for policy,cid in picks.items():
            c=candidates[cid];r=c["robot"];i=c["index"];P=poses[r];t=c["image_timestamp"]
            if cid not in done:
                paths=[Path(original.DATA)/original.SEQ/r/cam/original._fname_for(r,cam,t) for cam in ("infra1","infra2")]
                raw=[cv2.imread(str(p),0) for p in paths]
                left,right=sd[r].rectify_pair(*raw);z,sz,v=_depth_rectified(sd[r],left,right)
                if cfg.beta:
                    zr=clean_right_depth(sd[r],left,right)
                    left,right,_=stereo_smoke(sd[r],left,right,z,zr,P["T"][i],c["timestamp"],cfg)
                    z,sz,v=_depth_rectified(sd[r],left,right)
                b=make_builder(arm);b.tree=octomap.OcTree.read(str(out/f"initial_{name}.ot").encode())
                pc,sp=sd[r].backproject(z,sz,v,downsample=4)
                if v.sum()>=100:b.integrate_frame(P["T"][i],float(P["tr_sigma_pos"][i]),pc,sp,teammates=mates[r].at(P["t"][i]))
                b.finalize();done[cid]=dense_map(sparse_map(b),lo,shape)
                np.savez_compressed(out/f"observation_{name}_{cid}.npz",left=left,right=right,Z=z,valid=v,P_cam=pc,sZ=sp,T=P["T"][i],timestamp=P["t"][i])
                write_json(out/f"observation_{name}_{cid}.json",dict(recorded_paths=[str(p) for p in paths],sha256=[digest(p) for p in paths],candidate=c,valid_fraction=float(v.mean())))
            variants[policy]=(done[cid],c)
        for policy,(lab,c) in variants.items():
            graph=DecisionGraph(lab,lo);analysis=graph.decide(entry,goals)
            rows.append(dict(condition=name,policy=policy,additional_observations=int(c is not None),
                             move_cost=0. if c is None else c["move_cost"],candidate_id="" if c is None else c["id"],
                             n_passable=sum(d["verdict"]=="passable" for d in analysis["decisions"]),
                             n_unverified=sum(d["verdict"]=="unverified" for d in analysis["decisions"]),
                             independent_reference="unavailable",**map_metrics(lab,initial["clean"])))
            goal_rows.extend(dict(condition=name,policy=policy,**d) for d in analysis["decisions"])
            np.savez_compressed(out/f"map_{name}_{policy}.npz",labels=lab,ijk_min=lo,res=.1)
        write_csv(out/"summary.csv",rows);write_json(out/"goal_paths.json",goal_rows)
    print("MILUV replay complete",flush=True)


if __name__=="__main__":main()
