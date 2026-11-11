#!/usr/bin/env python3
"""Replay actual public depths into an alternative observed-surface map."""
import argparse,csv,json,sys,time,hashlib
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.surface_fusion import SurfaceFusion,FusionCfg
from covor.explore.depth_observation import camera_for
from covor.explore.human import Human
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.runner import write_json,csv_write,evaluate
from covor.entry.path_eval import ReferenceValidator,PathEvalCfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--metadata-run',type=Path);p.add_argument('--map',type=Path);p.add_argument('--volume',type=Path)
    p.add_argument('--decomposed-quality',action='store_true')
    p.add_argument('--finite-time-correlation',action='store_true')
    p.add_argument('--local-contact',action='store_true')
    a=p.parse_args()
    if a.decomposed_quality and not a.finite_time_correlation:p.error('decomposed quality requires finite temporal covariance')
    a.out.mkdir(parents=True,exist_ok=False);metadata_run=a.metadata_run or a.run
    start=time.monotonic();meta=json.loads((metadata_run/'config.json').read_text());cfg=meta['configuration'];prior=json.loads((metadata_run/'prior.json').read_text())
    sd=camera_for(cfg);model=cfg['sensor']['error_model']
    fusion=SurfaceFusion(FusionCfg(capture_period=cfg['sensor_period'],correlation_time=model['correlation_time_s'],variance_weights=tuple(model['noise_variance_weights']),finite_time_correlation=a.finite_time_correlation,local_contact_refinement=a.local_contact))
    frames=list(csv.DictReader((a.run/'frames.csv').open()));order=[]
    source=Path(__file__).resolve().parents[2]/'covor/explore/surface_fusion.py'
    (a.out/'surface_fusion_source.py').write_bytes(source.read_bytes())
    if a.volume:
        import open3d as o3d
        saved=json.loads(a.volume.with_suffix('.json').read_text());assert saved['source']==str(a.run.resolve()) and saved['frames']==len(frames)
        saved_cfg=asdict(FusionCfg());saved_cfg.update(saved['configuration'])
        expected=asdict(fusion.cfg)
        assert json.loads(json.dumps({k:v for k,v in saved_cfg.items() if k!='local_contact_refinement'}))==json.loads(json.dumps({k:v for k,v in expected.items() if k!='local_contact_refinement'}))
        assert saved.get('decomposed_quality',False)==a.decomposed_quality
        fusion.vbg=o3d.t.geometry.VoxelBlockGrid.load(str(a.volume));fusion.frames=len(frames)
    for n,row in enumerate(frames,1):
        path=a.run/'frames'/(row['frame_id']+'.npz')
        if not a.volume:
            with np.load(path) as q:
                T=np.array(json.loads(row['camera_pose']))
                sigma=q['sigma'];extra=None
                if a.decomposed_quality:
                    from covor.explore.observed_depth_quality import uncertainty_components
                    sigma,extra=uncertainty_components(sd,q['Z'],q['valid'],T,cfg,meta['condition'])
                fusion.integrate(sd,q['Z'],sigma,q['valid'],T,q['teammates'],cfg['sensor']['teammate_exclusion_radius'],extra,time=float(row['time']))
        order.append(dict(frame_id=row['frame_id'],time=row['time'],map_version=row['map_version']))
        if n%100==0:print('FUSED',n,'/',len(frames),'wall',round(time.monotonic()-start,1),flush=True)
    if not a.volume:
        fusion.vbg.save(str(a.out/'volume.npz'))
        write_json(a.out/'volume.json',dict(source=str(a.run.resolve()),frames=len(frames),configuration=asdict(fusion.cfg),source_sha256=hashlib.sha256((a.out/'surface_fusion_source.py').read_bytes()).hexdigest(),decomposed_quality=a.decomposed_quality))
    with np.load(a.map or a.run/'maps/final.npz') as q:
        labels=q['labels'];ijk=q['ijk'];layer={k:q[k] for k in q.files if k not in ('labels','ijk','res','version')}
    lab,updated,diagnostic=fusion.reconstruct(labels,layer,ijk,cfg['map_resolution'],cfg['human']['foot_contact_offset'])
    print('RECONSTRUCTED',diagnostic,flush=True)
    np.savez_compressed(a.out/'map.npz',labels=lab,ijk=ijk,res=cfg['map_resolution'],version=len(frames),**updated)
    analysis=Human(prior,cfg).analyze(lab,updated,ijk,False)
    write_json(a.out/'paths_before_reference.json',analysis['decisions'])
    # Actual paths are fixed before independent reference geometry is opened.
    b=ReconFireWorld(meta['scene_seed'],cfg);gt,gtijk=b.labels(cfg['map_resolution']);b.render_scene(False)
    validator=ReferenceValidator(gt,gtijk,cfg['map_resolution'],b.scene,cfg=PathEvalCfg(step_limit_m=cfg['human']['exact_step_limit'],contact_epsilon_m=cfg['human']['contact_epsilon_m']))
    truth=json.loads((metadata_run/'reference_targets.json').read_text());rows,summary=evaluate(analysis,truth,validator,prior,cfg['duration'],len(frames))
    baseline=json.loads((a.run/'summary.json').read_text());summary.update(source=str(a.run),map_model='quality_weighted_observed_tsdf',wall_seconds=time.monotonic()-start,diagnostic=diagnostic)
    write_json(a.out/'summary.json',summary);csv_write(a.out/'targets.csv',rows);csv_write(a.out/'input_order.csv',order)
    csv_write(a.out/'comparison.csv',[dict(method='original',**baseline),dict(method='observed_surface_fusion',**summary)])
    print('RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
