#!/usr/bin/env python3
"""Reuse the tested study audit with the corrected world factory, process-local."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_recon_scene import ReconFireWorld
import covor.explore.fire_world as world
# All truth regeneration and figure geometry in the shared audit use this
# factory. No running simulator or on-disk source module is modified.
world.FireWorld=ReconFireWorld
import scripts.explore.post_fire_study as shared
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_recon_20261107');p.add_argument('--report',default='reports/fire_recon_20261107');p.add_argument('--skip-plots',action='store_true');a=p.parse_args();root=Path(a.root)
assert json.load(open(root/'main/complete.json'))['all_expected_runs_retained']
cfg=json.load(open(root/'main/fixed_config.json'));cases={q['scene']:q for q in json.load(open(root/'evaluation_cases.json'))}
for i in cfg['scene_seeds']:assert ReconFireWorld(i,cfg).geometry_hash()==cases[i]['geometry_sha256']
if a.skip_plots:shared.plot=lambda *args:None
shared.analyze(root,Path(a.report))
