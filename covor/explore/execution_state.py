"""Exact execution checkpoints and causal replay of recorded depth observations.

This changes execution I/O only. OctoMap's full .ot format preserves log odds;
.bt/max-likelihood export is deliberately not used. Sensor caches are read only
at the requested current pose/time, after the normal policy/trajectory update.
"""
import csv,hashlib,json,os,pickle,shutil,time
from pathlib import Path
import numpy as np
from covor.occupancy import _sigmoid
from .sensing import camera_pose

class CheckpointStop(RuntimeError):pass

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def array_hashes(lab,layer):
    return {k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in dict(labels=lab,**layer).items()}

class CachedSensor:
    def __init__(self,sensor,execution,cfg):
        self.original=sensor;self.sd=sensor.sd;self.execution=execution;self.cfg=cfg;self.audit={}
        if execution.cache:
            path=execution.cache/'simulator_audit.csv'
            if path.exists():
                with path.open() as f:self.audit={q['frame_id']:q for q in csv.DictReader(f)}
    def _private(self,fid):return self.execution.folder/'audit_frames'/(fid+'.json')
    def _remember(self,fid,row):atomic_json(self._private(fid),row)
    def _last_audit(self,path):
        with path.open('rb') as f:
            header=next(csv.reader([f.readline().decode().strip()]));f.seek(max(0,path.stat().st_size-4096));line=f.read().splitlines()[-1].decode()
        return dict(zip(header,next(csv.reader([line]))))
    def capture(self,position,yaw,pitch,t,teammates,directory,frame_id,ideal=False,drone_id=0):
        directory=Path(directory);dest=directory/(frame_id+'.npz');choices=[dest]
        if self.execution.cache:choices.append(self.execution.cache/'frames'/(frame_id+'.npz'))
        source=next((p for p in choices if p.exists()),None);row=None
        private=self._private(frame_id)
        if private.exists():row=json.loads(private.read_text())
        elif frame_id in self.audit:row=self.audit[frame_id]
        if source is not None and row is not None:
            with np.load(source) as q:
                expected=camera_pose(position,yaw,pitch,self.cfg['sensor']['baseline'])
                if not np.allclose(q['T'],expected,rtol=0,atol=1e-12):raise AssertionError(f'cache pose differs: {frame_id}')
                if float(q['time'])!=t or int(q['drone_id'])!=drone_id:raise AssertionError('cache timestamp/drone mismatch')
                if not np.allclose(q['teammates'],np.asarray(teammates).reshape(-1,3),rtol=0,atol=1e-12):raise AssertionError('cache teammate pose differs')
                cam=q['T'].copy();z=q['Z'].copy();sigma=q['sigma'].copy();valid=q['valid'].copy()
            directory.mkdir(parents=True,exist_ok=True);self._remember(frame_id,row)
            if not dest.exists():
                try:os.link(source,dest)
                except OSError:shutil.copy2(source,dest)
            auditpath=directory.parent/'simulator_audit.csv';exists=auditpath.exists()
            with auditpath.open('a') as f:
                w=csv.DictWriter(f,fieldnames=list(row))
                if not exists:w.writeheader()
                w.writerow(row)
            self.execution.cached_frames+=1
            return cam,z,sigma,valid,dict(valid_ratio=float(valid.mean()),render_wall=0.,smoke_wall=0.,stereo_wall=0.,backend=self.cfg['sensor']['backend'],profile=self.cfg['sensor']['profile'],cache_replay=True)
        if dest.exists():
            # A capture interrupted before its private audit was committed.
            archive=self.execution.folder/'incomplete_frames';archive.mkdir(exist_ok=True);dest.replace(archive/(frame_id+f'.{time.time_ns()}.npz'))
        result=self.original.capture(position,yaw,pitch,t,teammates,directory,frame_id,ideal,drone_id)
        row=self._last_audit(directory.parent/'simulator_audit.csv');self._remember(frame_id,row);self.execution.new_frames+=1
        result[-1]['cache_replay']=False
        return result

class DurableExecution:
    def __init__(self,out,cache=None,period=8.,stop_after=None):
        self.out=Path(out);self.folder=self.out/'.execution';self.cache=Path(cache) if cache else None;self.period=period;self.stop_after=stop_after;self.cached_frames=0;self.new_frames=0;self.record=None;self.saved_elapsed=0.;self.sources={}
    def prepare(self,cfg,scene,seed,condition,policy,n,T,dt):
        self.cfg=cfg;self.dt=dt
        if abs(self.period/dt-round(self.period/dt))>1e-9:raise ValueError('checkpoint period must be a multiple of dt')
        signature=hashlib.sha256(json.dumps(dict(cfg=cfg,scene=scene,seed=seed,condition=condition,policy=policy,n=n,T=T,dt=dt),sort_keys=True).encode()).hexdigest();self.signature=signature
        for name in ['runner.py','execution_state.py','policy.py','human.py','motion.py','sensing.py','depth_observation.py','fire_recon_scene.py']:
            p=Path(__file__).with_name(name);self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if cfg.get('utility',{}).get('entry_objective','legacy')!='legacy' or cfg.get('candidates',{}).get('body_reobserve_views'):
            p=Path(__file__).with_name('entry_objective.py');self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if cfg.get('candidates',{}).get('body_reobserve_views'):
            p=Path(__file__).with_name('candidate_views.py');self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if cfg.get('motion',{}).get('controller')=='guarded_v1' or cfg.get('motion',{}).get('travel_facing_camera'):
            p=Path(__file__).with_name('flight_control.py');self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for rel in ['occupancy.py','entry/decision.py','entry/path_eval.py','entry/surfaces.py','entry/traverse.py','explore/fire_world.py','explore/fire_reference.py','explore/fire_smoke.py']:
            p=Path(__file__).resolve().parents[1]/rel;self.sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        self.folder.mkdir(parents=True,exist_ok=True);latest=self.folder/'latest.json'
        if latest.exists():
            info=json.loads(latest.read_text());path=self.folder/info['directory'];commit=json.loads((path/'commit.json').read_text());assert commit['signature']==signature and commit['sources']==self.sources,'checkpoint code/config changed'
            for name,h in commit['files'].items():assert hashlib.sha256((path/name).read_bytes()).hexdigest()==h
            with (path/'runtime.pkl').open('rb') as f:self.record=pickle.load(f)
            self.restore_path=path;self.saved_elapsed=self.record['elapsed'];self.cached_frames=self.record['cached_frames'];self.new_frames=self.record['new_frames']
            tails=self.folder/'recovery_tails'/str(time.time_ns());tails.mkdir(parents=True)
            for name,size in self.record['log_sizes'].items():
                p=self.out/name
                if not p.exists():assert size==0;continue
                assert p.stat().st_size>=size
                if p.stat().st_size>size:
                    with p.open('rb') as f:f.seek(size);(tails/(name+'.tail')).write_bytes(f.read())
                    with p.open('r+b') as f:f.truncate(size)
            atomic_json(self.folder/'restored.json',dict(tick=self.record['tick'],time=self.record['tick']*dt,directory=str(path),signature=signature))
        else:
            for name in ['decisions.jsonl','candidates.jsonl','assignments.jsonl','simulator_audit.csv']:
                if (self.out/name).exists() and (self.out/name).stat().st_size:raise FileExistsError('non-checkpoint output exists; use a new output path')
    def wrap_sensor(self,sensor):return CachedSensor(sensor,self,self.cfg)
    def restore_map(self,m):
        if self.record is None:return None
        import octomap
        data=self.record['map'];m.__dict__.update(data['attributes']);m.builder.__dict__.update(data['builder']);m.surfaces.__dict__.update(data['surfaces']);m.builder.tree=octomap.OcTree.read(str(self.restore_path/'map.ot').encode());c=m.builder.cfg
        m.builder.tree.setClampingThresMin(_sigmoid(c.clamp_min));m.builder.tree.setClampingThresMax(_sigmoid(c.clamp_max));m.builder.tree.setOccupancyThres(_sigmoid(c.tau_occ))
        lab,layer=m.dense();assert array_hashes(lab,layer)==self.record['map_hashes'],'restored map differs'
        return self.record['runtime'],lab,layer
    def maybe_save(self,tick,m,runtime,logs,elapsed,lab,layer):
        if tick%int(round(self.period/self.dt)) and tick!=int(round(runtime['T']/self.dt)):return
        begin=time.monotonic()
        for f in logs:f.flush()
        folder=self.folder/f't{tick:05d}';temporary=self.folder/f'.t{tick:05d}.{os.getpid()}.tmp';temporary.mkdir(exist_ok=True)
        assert m.builder.tree.write(str(temporary/'map.ot').encode()),'full OctoMap serialization failed'
        state=dict(tick=tick,elapsed=elapsed,cached_frames=self.cached_frames,new_frames=self.new_frames,runtime=runtime,map_hashes=array_hashes(lab,layer),map=dict(attributes={k:v for k,v in m.__dict__.items() if k not in ['builder','surfaces']},builder={k:v for k,v in m.builder.__dict__.items() if k!='tree'},surfaces=m.surfaces.__dict__),log_sizes={name:(self.out/name).stat().st_size if (self.out/name).exists() else 0 for name in ['decisions.jsonl','candidates.jsonl','assignments.jsonl','simulator_audit.csv']})
        with (temporary/'runtime.pkl').open('wb') as f:pickle.dump(state,f,pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
        commit=dict(signature=self.signature,sources=self.sources,files={name:hashlib.sha256((temporary/name).read_bytes()).hexdigest() for name in ['map.ot','runtime.pkl']});atomic_json(temporary/'commit.json',commit)
        if folder.exists():shutil.rmtree(folder)
        temporary.replace(folder);atomic_json(self.folder/'latest.json',dict(directory=folder.name,tick=tick,time=tick*self.dt,map_version=m.version,checkpoint_wall=time.monotonic()-begin,cached_frames=self.cached_frames,new_frames=self.new_frames))
        if self.cache:
            source=self.cache/'maps'/f't{tick:05d}.npz'
            if source.exists():
                with np.load(source) as q:
                    assert np.array_equal(lab,q['labels']),f'cached prefix labels differ at {tick}'
                    for k,v in layer.items():assert np.array_equal(v,q[k],equal_nan=True),(tick,k)
                with (self.folder/'prefix_verification.jsonl').open('a') as f:f.write(json.dumps(dict(tick=tick,time=tick*self.dt,map_version=m.version,all_map_fields_exact=True))+'\n')
        # Keep two complete probability/state generations; frame/map history stays.
        previous=sorted(p for p in self.folder.glob('t*') if p.is_dir())
        for p in previous[:-2]:shutil.rmtree(p)
        if self.stop_after is not None and tick*self.dt>=self.stop_after:raise CheckpointStop(f'validation stop at {tick*self.dt}')
