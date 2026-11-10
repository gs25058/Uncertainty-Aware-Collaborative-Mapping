#!/usr/bin/env python3
"""Observed-depth-only support sampling diagnostic, with exact baseline replay.

    Occupancy and all-hit maxima are reused unchanged from the original map.
    SupportSurfaces is independent of that volume and is re-fused from actual
    public depth/quality/pose frames. Geometry is opened only for final eval.
"""
import argparse,csv,hashlib,io,json,sys
from dataclasses import asdict,replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.depth_observation import camera_for
from covor.explore.sensing import SharedMap
from covor.entry.surfaces import SupportSurfaces
from covor.explore.human import Human
from covor.explore.runner import csv_write,evaluate,write_json
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    source=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison.csv').exists():raise FileExistsError(out)
    metadata=json.load(open(source/'config.json'));cfg=metadata['configuration'];assert cfg['pose_mode']=='perfect_odometry'
    prior=json.load(open(source/'prior.json'));sd=camera_for(cfg);m=SharedMap(prior,cfg)
    surfaces={stride:SupportSurfaces(replace(m.surfaces.cfg,stride=stride)) for stride in [4,2]}
    records=list(csv.DictReader(open(source/'frames.csv')));inputs=[]
    for n,row in enumerate(records,1):
        path=source/'frames'/(row['frame_id']+'.npz');raw=path.read_bytes()
        with np.load(io.BytesIO(raw)) as q:
            T=np.array(json.loads(row['camera_pose']));np.testing.assert_allclose(T,q['T'],rtol=0,atol=1e-12)
            for surface in surfaces.values():surface.integrate(sd,q['Z'],q['sigma'],q['valid'],T,teammates=q['teammates'])
        inputs.append(dict(frame_id=row['frame_id'],time=row['time'],map_version=row['map_version'],sha256=hashlib.sha256(raw).hexdigest()))
        if n%200==0:print('REPLAYED',n,'/',len(records),'actual frames',flush=True)
    with np.load(source/'maps/final.npz') as q:original={k:q[k].copy() for k in q.files}
    assert int(original['version'])==len(records) and np.array_equal(original['ijk'],m.ijk)
    baseline=surfaces[4].dense(m.ijk,m.shape)
    checks={k:np.array_equal(v,original[k],equal_nan=True) for k,v in baseline.items()}
    write_json(out/'baseline_verification.json',dict(all_exact=all(checks.values()),fields=checks,actual_frames=len(records),no_geometry_read_during_fusion=True))
    assert all(checks.values()),checks
    # Independent geometry is introduced only after both observed-only fusions.
    b=ReconFireWorld(metadata['scene_seed'],cfg);gt,ijk=b.labels(cfg['map_resolution']);b.render_scene(False)
    assert b.prior()==prior;truth=json.load(open(source/'reference_targets.json'))
    validator=ReferenceValidator(gt,ijk,m.res,b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
    summaries=[]
    for stride,surface in surfaces.items():
        folder=out/f'stride{stride}';folder.mkdir(exist_ok=True)
        layer=surface.dense(m.ijk,m.shape);layer['hit_z_max']=original['hit_z_max'].copy()
        analysis=Human(prior,cfg).analyze(original['labels'],layer,m.ijk,False)
        rows,summary=evaluate(analysis,truth,validator,prior,cfg['duration'],len(records))
        summary.update(surface_stride=stride,source=str(source),mode='fixed_recorded_trajectory_support_sampling_ablation',actual_frames=len(records),occupancy_and_all_hit_maxima_unchanged=True)
        write_json(folder/'summary.json',summary);write_json(folder/'support_config.json',asdict(surface.cfg));csv_write(folder/'target_paths.csv',rows)
        np.savez_compressed(folder/'map.npz',labels=original['labels'],ijk=m.ijk,res=m.res,version=len(records),**layer)
        summaries.append(summary);print('RESULT',json.dumps(summary),flush=True)
    csv_write(out/'comparison.csv',summaries);csv_write(out/'input_order.csv',inputs)
    write_json(out/'scope.json',dict(source_run=str(source),fusion_inputs='same actual public Z/sigma/valid/estimated camera poses/teammates',unchanged=['volumetric labels','all-hit height maxima','min2 actual frames per support','plane/slope/evidence/height rules','human dimensions/postures/step','goal/reference/actual path validation'],changed='support tangent/plane sampling stride4 versus2; no hole filling or floor inference',not_closed_loop_performance=True))


if __name__=='__main__':main()
