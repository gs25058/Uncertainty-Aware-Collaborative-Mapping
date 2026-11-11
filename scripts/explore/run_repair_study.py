#!/usr/bin/env python3
"""Frozen-source paired late-stage method study; no implicit old-batch restart."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.execution_state import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--configs',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--additional',type=float,default=32.)
    p.add_argument('--scenes',type=int,nargs='+',default=[0,1,4,8]);a=p.parse_args()
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'study.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snap=root/'source_snapshot';manifest=root/'source_manifest.json'
    if not manifest.exists():
        hashes={}
        for source in sorted([*Path('covor').rglob('*.py'),*Path('scripts/explore').glob('*.py'),*Path('tests').glob('*.py')]):
            dest=snap/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            hashes[str(source)]=hashlib.sha256(dest.read_bytes()).hexdigest()
        atomic_json(manifest,hashes)
        (root/'source.patch').write_bytes(subprocess.check_output(['git','diff','--','covor','scripts','tests']))
    else:
        hashes=json.loads(manifest.read_text())
        assert all(hashlib.sha256((snap/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    configs=root/'configs';configs.mkdir(exist_ok=True)
    for name in ['precision_common','repair_common']:
        source=a.configs/(name+'.json');dest=configs/source.name
        if dest.exists():assert dest.read_bytes()==source.read_bytes()
        else:shutil.copy2(source,dest)
    variants={'unknown_common':('unknown','precision_common'),'precision_common':('entry','precision_common'),'repair_common':('entry','repair_common')}
    jobs=[]
    for scene in a.scenes:
        for condition in ['clean','strong']:
            parent=Path(f'results/entry_optimization_20261109/round03/precision_v4/scene{scene}_seed0_{condition}_entry').resolve()
            for variant,(policy,config) in variants.items():
                name=f'scene{scene}_seed0_{condition}_{policy}'
                folder=root/variant/name;folder.parent.mkdir(exist_ok=True)
                command=[sys.executable,str(snap/'scripts/explore/run_repair_branch.py'),'--parent',str(parent),'--config',str(configs/(config+'.json')),'--out',str(folder),'--policy',policy,'--additional',str(a.additional)]
                jobs.append(dict(variant=variant,scene=scene,condition=condition,policy=policy,parent=str(parent),out=str(folder),command=command))
    specification=dict(scenes=a.scenes,seed=0,conditions=['clean','strong'],variants=variants,additional_time=a.additional,jobs=jobs)
    spec_path=root/'fixed_spec.json'
    if spec_path.exists():assert json.loads(spec_path.read_text())==json.loads(json.dumps(specification))
    else:atomic_json(spec_path,specification)
    atomic_json(root/'status.json',dict(phase='running',pid=os.getpid(),started=time.time(),expected=len(jobs)))
    def run(job):
        out=Path(job['out'])
        if (out/'summary.json').exists():return dict(**job,status='complete',summary=json.loads((out/'summary.json').read_text()))
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with out.with_suffix('.log').open('a') as log:
            ret=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
        return dict(**job,returncode=ret,status='complete' if (out/'summary.json').exists() else 'failed')
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run,job):job for job in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:row=future.result()
            except Exception as exc:row=dict(**pending[future],status='failed',error=repr(exc))
            rows.append(row);atomic_json(root/'trial_results.json',rows)
            print('FINISHED',row['variant'],row['scene'],row['condition'],row['status'],len(rows),'/',len(jobs),flush=True)
    failures=[q for q in rows if q['status']!='complete']
    atomic_json(root/'status.json',dict(phase='failed' if failures else 'complete',expected=len(jobs),finished=len(rows),failures=failures,completed=time.time()))
    if failures:raise RuntimeError(failures)


if __name__=='__main__':main()
