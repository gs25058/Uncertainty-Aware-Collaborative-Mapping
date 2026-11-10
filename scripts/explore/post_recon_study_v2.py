#!/usr/bin/env python3
"""Corrected-scene factory plus explicit zero-count-aware postprocessing."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.fire_recon_scene import ReconFireWorld
import covor.explore.fire_world as world
world.FireWorld=ReconFireWorld
from scripts.explore.post_observation_audit import analyze
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_recon_20261107');p.add_argument('--report',default='reports/fire_recon_20261107');p.add_argument('--skip-plots',action='store_true');a=p.parse_args();root=Path(a.root);assert json.load(open(root/'main/complete.json'))['all_expected_runs_retained'];cfg=json.load(open(root/'main/fixed_config.json'));cases={c['scene']:c for c in json.load(open(root/'evaluation_cases.json'))}
for i in cfg['scene_seeds']:assert ReconFireWorld(i,cfg).geometry_hash()==cases[i]['geometry_sha256']
analyze(root,Path(a.report),a.skip_plots)
