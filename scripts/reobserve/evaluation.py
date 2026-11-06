"""Goal/path metrics. No metric re-plans an invalid proposed path."""
from dataclasses import replace
import numpy as np
from covor.entry import traverse as TV, clean3d
from covor.entry.config import EntryCfg
from covor.entry.path_eval import motion_path


def legacy_paths(labels,ijk,res,entry,goals,gap):
    lab,_=clean3d.clean(labels,EntryCfg(res=res))
    cfg=replace(TV.TravCfg(res=res),gap=gap)
    x0,y0,x1,y1=entry["xy_bounds"]
    result=TV.traverse(lab,cfg,((x0+x1)/2,(y0+y1)/2),ijk,snap_m=0,return_paths=True)
    rows=[]
    for goal in goals:
        candidates=[]
        for (i,j),state in result["path_best"].items():
            x,y=(np.array([i,j])+ijk[:2]+.5)*res
            foot=(state[2]+ijk[2]+1)*res
            a,b,c,d=goal["xy_bounds"];lo,hi=goal["floor_z_bounds"]
            if a<=x<=c and b<=y<=d and lo<=foot<=hi:candidates.append((result["cost"][i,j],state))
        path=[]
        if candidates:
            _,state=min(candidates)
            while state is not None:
                i,j,k,g=state;posture,width=result["path_modes"][state]
                p=(np.array([i,j,k])+ijk+.5)*res;p[2]=(k+ijk[2]+1)*res
                path.append(dict(xyz=p.tolist(),posture=posture,width=width,
                                 observed_support=g==0,legacy_gap=g>0))
                state=result["path_parent"][state]
            path.reverse()
        rows.append(dict(goal=goal["id"],verdict="passable" if path else "unverified",
                         reason="legacy_path" if path else "legacy_no_path_not_blockage",
                         path=path,uncertain_segments=[]))
    return rows,result


def metrics(rows,before=None):
    n=len(rows);reachable=sum(r["reference_state"]=="reachable" for r in rows)
    valid=sum(r["path_status"]=="valid" for r in rows)
    invalid=sum(r["path_status"]=="invalid" for r in rows)
    checked=valid+invalid
    hit=sum(r["reference_state"]=="reachable" and r["path_status"]=="valid" for r in rows)
    decided=sum(r["verdict"] in ("passable","blocked") for r in rows)
    out=dict(n_cases=n,n_reference_reachable=reachable,
        n_reference_blocked=sum(r["reference_state"]=="blocked" for r in rows),
        n_reference_undetermined=sum(r["reference_state"]=="undetermined" for r in rows),
        valid_path_confirmed=hit,valid_path_confirm_rate=hit/reachable if reachable else "N/A",
        n_proposed=sum(r["verdict"]=="passable" for r in rows),n_valid_paths=valid,n_invalid_paths=invalid,
        n_unverifiable_paths=sum(r["path_status"]=="unverifiable" for r in rows),
        n_checked_paths=checked,wrong_path_rate=invalid/checked if checked else "N/A",
        n_decided=decided,decision_rate=decided/n if n else "N/A",
        n_unverified=sum(r["verdict"]=="unverified" for r in rows),
        correct_blocked=sum(r["verdict"]=="blocked" and r["reference_state"]=="blocked" for r in rows),
        false_blocked_reachable=sum(r["verdict"]=="blocked" and r["reference_state"]=="reachable" for r in rows))
    keys=("unverified_to_valid_path","unverified_to_correct_blocked","new_invalid_guidance",
          "passable_revoked","invalid_guidance_revoked","valid_guidance_lost")
    out.update({k:0 for k in keys})
    if before:
        baseline={r["case_id"]:r for r in before}
        for r in rows:
            b=baseline[r["case_id"]]
            out["unverified_to_valid_path"]+=int(b["verdict"]=="unverified" and r["verdict"]=="passable" and r["path_status"]=="valid" and r["reference_state"]=="reachable")
            out["unverified_to_correct_blocked"]+=int(b["verdict"]=="unverified" and r["verdict"]=="blocked" and r["reference_state"]=="blocked")
            out["new_invalid_guidance"]+=int(r["path_status"]=="invalid" and b["path_status"]!="invalid")
            revoked=b["verdict"]=="passable" and r["verdict"]!="passable"
            out["passable_revoked"]+=int(revoked)
            out["invalid_guidance_revoked"]+=int(revoked and b["path_status"]=="invalid")
            out["valid_guidance_lost"]+=int(b["path_status"]=="valid" and r["path_status"]!="valid")
    return out
