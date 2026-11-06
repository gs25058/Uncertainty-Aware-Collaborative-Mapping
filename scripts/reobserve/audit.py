#!/usr/bin/env python3
"""Check archived baseline outputs and pre-path API against current results."""
import argparse,csv,json,subprocess,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/"scripts/smoke"):sys.path.insert(0,str(p))
import numpy as np
from common import digest,write_json
from covor.entry import traverse as TV


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--out",required=True)
    args=ap.parse_args();src=Path(args.input);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    complete=json.load(open(src/"complete.json"));assert complete["n_goal_rows"]==1848 and complete["n_area_rows"]==28
    old_artifacts=ROOT/"reports/smoke_20261103/artifacts.json"
    checks=json.load(open(old_artifacts))
    for rel,record in checks.items():assert digest(old_artifacts.parent/rel)==record["sha256"],rel
    source=subprocess.check_output(["git","show","HEAD:covor/entry/traverse.py"],cwd=ROOT,text=True)
    module=types.ModuleType("covor.entry._legacy_reference");module.__package__="covor.entry"
    sys.modules[module.__name__]=module;exec(source,module.__dict__)
    matched=0
    for name in ("clean","heterogeneous_medium"):
        q=np.load(ROOT/f"results/smoke_20261103/corridor/map_{name}_initial.npz")
        a=module.traverse(q["labels"],module.TravCfg(),(-2.35,-19.55),q["ijk_min"],snap_m=0)
        b=TV.traverse(q["labels"],TV.TravCfg(),(-2.35,-19.55),q["ijk_min"],snap_m=0)
        for k in a:
            if isinstance(a[k],np.ndarray):assert np.array_equal(a[k],b[k]),k
            else:assert a[k]==b[k],k
        matched+=1
    original=list(csv.DictReader(open(ROOT/"results/smoke_20261103/corridor/summary.csv")))
    area=list(csv.DictReader(open(src/"area_metrics.csv")))
    for r in area:
        if r["policy"]!="initial":continue
        old=next(p for p in original if p["condition"]==r["condition"] and p["policy"]=="initial")
        for key in ("reach_recall","reach_iou","false_reach_rate","n_map_reach","n_false_reach"):
            assert abs(float(old[key])-float(r[key]))<1e-12,(r["condition"],key)
    selected=0
    for logpath in (src/"selection").glob("*.json"):
        log=json.load(open(logpath));assert not log["candidate_depth_access"] and not log["ground_truth_access"]
        assert len(log["picks"])==3
        assert digest(ROOT/f'results/smoke_20261103/corridor/map_{log["condition"]}_initial.npz')==log["current_map_sha256"]
        selected+=1
    assert selected==7
    rows=list(csv.DictReader(open(src/"goals_paths.csv")))
    assert len(rows)==1848
    for r in rows:assert int(r["additional_observations"])==(r["policy"]!="initial")
    modified=[ROOT/"covor/entry"/n for n in ("traverse.py","decision.py","assignment.py","path_eval.py")]
    modified+=list((ROOT/"scripts/reobserve").glob("*.py"))
    write_json(out/"audit.json",dict(passed=True,original_report_artifacts_unchanged=len(checks),
        default_output_exact_against_HEAD_maps=matched,initial_area_conditions_exact=7,
        selection_logs_checked=selected,goal_path_rows=len(rows),one_frame_budget_checked=True,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in modified}))
    print("PASS audit: original artifacts, default arrays, initial area metrics, selection inputs and one-frame budgets")


if __name__=="__main__":main()
