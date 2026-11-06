#!/usr/bin/env python3
"""Reload saved maps, verify reported metrics and source hashes, expose entry snaps."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from common import digest,write_csv,write_json
from run_corridor import eval_map,traversal,GOALS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args();src=Path(args.input);out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.load(open(src/"miluv"/"manifest.json"))
    checks={p:digest(p)==h for p,h in manifest["pose_hashes"].items()}
    nimages=0
    for f in manifest["frames"]:
        for side in ("left","right"):
            assert digest(f[side])==f[side+"_sha256"]
            nimages+=1
    design=json.load(open(src/"corridor"/"design.json"))
    checks.update({p:digest(p)==h for p,h in design["inputs"].items()})
    provenance=json.load(open(ROOT/"meshes"/"PROVENANCE.json"))
    mesh=ROOT/"meshes/2026-09-15/2026. 9. 15.floorfix.obj"
    expected=next(d["obj_md5"] for d in provenance["derived"] if d["obj"]==mesh.name)
    checks[str(mesh)]=hashlib.md5(mesh.read_bytes()).hexdigest()==expected
    assert all(checks.values()),"input changed during experiment"
    gt=np.load(ROOT/"results/synth_corridor915f/gt_voxel_fs.npz")
    lab,ijk,res=gt["labels"],gt["ijk_min"],float(gt["res"])
    tg=traversal(lab,ijk,res)
    rows=list(csv.DictReader(open(src/"corridor"/"summary.csv")))
    snaps=[];conservative=[];goal_rows=[]
    for row in rows:
        q=np.load(src/"corridor"/f'map_{row["condition"]}_{row["policy"]}.npz')
        s,v,_=eval_map(q["labels"],tg,lab,ijk,res,GOALS)
        for k in ("reach_recall","false_reach_rate","n_map_reach","n_false_reach",
                  "confirmed_reachable_goals","false_confirmed_gt_unreachable",
                  "false_confirmed_gt_blocked","gap_permitted_true_goals"):
            assert abs(float(row[k])-float(s[k]))<1e-12,(row["condition"],row["policy"],k)
        snaps.append(dict(condition=row["condition"],policy=row["policy"],
                          entry_snapped_m=s["entry_snapped_m"],strict_entry_snapped_m=s["strict_entry_snapped_m"]))
        legacy=s["confirmed_reachable_goals"]
        if s["strict_entry_snapped_m"] is not None and s["strict_entry_snapped_m"]>0:
            s,v,_=eval_map(q["labels"],tg,lab,ijk,res,GOALS,strict_entry_snap=0.)
        corrected=dict(row,**s,legacy_confirmed_reachable_goals=legacy,
                       confirmation_entry_rule="exact entry; no snap; no unsupported gaps")
        conservative.append(corrected)
        goal_rows.extend(dict(condition=row["condition"],policy=row["policy"],**g) for g in v)
    for r in conservative:
        base=next(b for b in conservative if b["condition"]==r["condition"] and b["policy"]=="initial")
        if r["policy"]!="initial":
            r["delta_true_confirmed"]=r["confirmed_reachable_goals"]-base["confirmed_reachable_goals"]
            r["delta_false_confirmed"]=r["false_confirmed_gt_unreachable"]-base["false_confirmed_gt_unreachable"]
    (out/"corridor").mkdir(exist_ok=True)
    write_csv(out/"corridor"/"summary_conservative.csv",conservative)
    write_csv(out/"corridor"/"goals_conservative.csv",goal_rows)
    write_csv(out/"entry_snaps.csv",snaps)
    code=[ROOT/"covor/smoke.py",ROOT/"covor/reobserve.py",*sorted((ROOT/"scripts/smoke").glob("*.py"))]
    write_json(out/"audit.json",dict(passed=True,reloaded_maps=len(rows),original_image_hash_checks=nimages,
         input_hash_checks=checks,source_code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code},
         gt_entry_snap_m=tg["entry_snapped_m"]))
    print("verified",len(rows),"maps,",nimages,"original images; source hashes unchanged")


if __name__=="__main__":main()
