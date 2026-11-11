#!/usr/bin/env python3
"""Causal policy fork of a completed, probability-preserving checkpoint.

No teleport/reset/extra observation is introduced. This is a late-stage repair
experiment, not a replacement for an entrance-start full-mission comparison.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore import runner
from covor.explore.execution_state import DurableExecution, atomic_json
from covor.explore.fire_recon_scene import ReconFireWorld
from covor.explore.execution_acceleration import enable_exact_motion_cache


class RepairBranch(DurableExecution):
    def __init__(self,out,parent,additional):
        super().__init__(out,period=8.)
        self.parent=Path(parent).resolve();self.additional=additional

    def prepare(self,cfg,scene,seed,condition,policy,n,T,dt):
        local=(self.folder/'latest.json').exists()
        # Normal signature/source checks govern every resume of THIS branch.
        super().prepare(cfg,scene,seed,condition,policy,n,T,dt)
        if local:return
        metadata=json.loads((self.parent/'config.json').read_text())
        old=metadata['configuration']
        assert (scene,seed,condition)==(metadata['scene_seed'],metadata['run_seed'],metadata['condition'])
        assert n==old['n_drones'] and dt==old['dt']
        for key in set(old)|set(cfg):
            if key not in ('utility','candidates','duration','protocol_version'):
                assert old.get(key)==cfg.get(key),f'physical/evaluation setting changed: {key}'
        latest=json.loads((self.parent/'.execution/latest.json').read_text())
        parent_checkpoint=self.parent/'.execution'/latest['directory']
        commit=json.loads((parent_checkpoint/'commit.json').read_text())
        for name,digest in commit['files'].items():
            assert hashlib.sha256((parent_checkpoint/name).read_bytes()).hexdigest()==digest
        with (parent_checkpoint/'runtime.pkl').open('rb') as f:self.record=pickle.load(f)
        t0=self.record['tick']*dt
        assert t0==old['duration'] and T==t0+self.additional
        assert (self.parent/'summary.json').exists(),'fork completed missions only'
        self.restore_path=parent_checkpoint; self.saved_elapsed=0.
        self.record['runtime']['T']=T
        # Preserve prefix logs/frames for complete replay, without writing to
        # parent files. All new observations are made at their actual new times.
        for name,size in self.record['log_sizes'].items():
            source=self.parent/name
            if source.exists():
                with source.open('rb') as f:(self.out/name).write_bytes(f.read(size))
        frames=self.out/'frames';frames.mkdir(exist_ok=True)
        for source in (self.parent/'frames').glob('*.npz'):
            dest=frames/source.name
            if not dest.exists():os.link(source,dest)
        atomic_json(self.out/'branch_manifest.json',dict(parent=str(self.parent),
            parent_checkpoint=str(parent_checkpoint),parent_checkpoint_commit=commit,
            start_time=t0,end_time=T,additional_time=self.additional,
            initial_map_hashes=self.record['map_hashes'],
            initial_states=[dict(drone=s['id'],position=s['p'].tolist(),failed=s['failed'],
                task_until=float(s['task_until']),trajectory=s['traj'].record() if s['traj'] is not None else None)
                for s in self.record['runtime']['states']],
            preserved='full OctoMap probabilities, support statistics, hit maxima, states, reservations, pending tasks, observation history',
            changed='planning utility/common candidate setting only; no parent truth/measurements used to rank new actions'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--policy',choices=['unknown','entry'],required=True)
    p.add_argument('--additional',type=float,default=32.);a=p.parse_args()
    if (a.out/'summary.json').exists():raise FileExistsError(a.out)
    metadata=json.loads((a.parent/'config.json').read_text());cfg=json.loads(a.config.read_text())
    cfg['duration']=metadata['configuration']['duration']+a.additional
    runner.make_world=lambda c,i:ReconFireWorld(i,c)
    accelerator=enable_exact_motion_cache()
    execution=RepairBranch(a.out,a.parent,a.additional)
    summary=runner.run(cfg,metadata['scene_seed'],metadata['run_seed'],metadata['condition'],a.policy,a.out,execution=execution)
    baseline=json.loads((a.parent/'summary.json').read_text())
    deltas={k:summary[k]-baseline[k] for k in ['captures','travel_distance','failed_drones','collision_events','valid_confirmed','correct_blocked','invalid_paths','false_blocked']}
    atomic_json(a.out/'repair_delta.json',dict(start_time=baseline['time'],end_time=summary['time'],**deltas))
    atomic_json(a.out/'execution_acceleration.json',dict(hits=accelerator.hits,misses=accelerator.misses,mode='exact native motion memoization'))


if __name__=='__main__':main()
