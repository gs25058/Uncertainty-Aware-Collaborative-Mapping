#!/usr/bin/env python3
"""Map-only support/contact/body/connectivity diagnosis; no truth is opened."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.human import Human
from covor.explore.runner import write_json, csv_write


def diagnose(run, map_path, config, out):
    prior = json.loads((run / 'prior.json').read_text())
    with np.load(map_path) as data:
        lab, ijk = data['labels'], data['ijk']
        layer = {k: data[k] for k in data.files if k not in ('labels', 'ijk', 'res', 'version')}
    analysis = Human(prior, config).analyze(lab, layer, ijk, False)
    g = analysis['graph']; s = g.support_layer
    observed = g.surface_candidates
    supported = s['support'] & np.isfinite(s['height'])
    std_clear = s['height'] + 3*s['height_std'] + .005 <= g.foot_z + 1e-9
    hit_clear = (g.lab == 1) | (s['hit_z_max'] + .005 <= g.foot_z + 1e-9)
    contact = supported & std_clear & hit_clear
    reachable = np.zeros(g.lab.shape, bool)
    nodes = list(analysis['confirmed_search']['best'])
    if nodes: reachable[tuple(g.xyz[nodes].T)] = True
    masks = dict(observed_surface=observed, support=supported,
                 support_rejected_scatter=supported & ~std_clear,
                 support_rejected_hit=supported & ~hit_clear,
                 contact_eligible=contact,
                 contact_eligible_body_unconfirmed=contact & ~g.confirmed,
                 confirmed=g.confirmed, confirmed_entry_connected=reachable,
                 confirmed_disconnected=g.confirmed & ~reachable)
    rows = []
    for goal in prior['goals']:
        x0,y0,x1,y1=goal['xy_bounds']; z0,z1=goal['floor_z_bounds']
        x=(np.arange(g.lab.shape[0])+g.ijk[0]+.5)*g.res
        y=(np.arange(g.lab.shape[1])+g.ijk[1]+.5)*g.res
        roi=(x[:,None,None]>=x0)&(x[:,None,None]<=x1)&(y[None,:,None]>=y0)&(y[None,:,None]<=y1)&(g.foot_z>=z0)&(g.foot_z<=z1)
        decision=next(d for d in analysis['decisions'] if d['goal']==goal['id'])
        rows.append(dict(goal=goal['id'],verdict=decision['verdict'],**{key:int((mask&roi).sum()) for key,mask in masks.items()}))
    out.mkdir(parents=True,exist_ok=False)
    csv_write(out/'goals.csv',rows)
    write_json(out/'summary.json',dict(map=str(map_path),current_observed_map_only=True,
        truth_opened=False,counts={key:int(mask.sum()) for key,mask in masks.items()},
        scatter_contact_limit_m=(config['human']['foot_contact_offset']-.005)/3,
        note='Rejections overlap; body_unconfirmed also contains unknown cells. No counterfactual masks are used for decisions.'))
    print(json.dumps({key:int(mask.sum()) for key,mask in masks.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--map',type=Path);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());cfg=cfg.get('configuration',cfg)
    diagnose(a.run,a.map or a.run/'maps/final.npz',cfg,a.out)
