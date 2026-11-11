#!/usr/bin/env python3
"""Reassess completed causal branches using one common whole-target cut rule."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.execution_state import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/method_refinement_20261110'))
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();a.root=a.root.resolve()
    spec=json.loads((a.root/'paired32/fixed_spec.json').read_text());out=a.root/'full_goal_reassessment';out.mkdir(exist_ok=True)
    jobs=[]
    for q in spec['jobs']:
        dest=out/q['variant']/f'scene{q["scene"]}_{q["condition"]}'
        command=[sys.executable,str(a.root/'reassessment_source/scripts/explore/replay_route_quality.py'),
            '--runs',q['out'],'--out',str(dest),'--complete-goal-cut','--skip-quality']
        jobs.append(dict(source=q['out'],out=str(dest),command=command,variant=q['variant'],scene=q['scene'],condition=q['condition']))
    atomic_json(out/'commands.json',jobs)
    def run(q):
        source=Path(q['source']);dest=Path(q['out']);dest.parent.mkdir(exist_ok=True)
        while not (source/'summary.json').exists():
            state=json.loads((a.root/'paired32/status.json').read_text())
            if state['phase']=='failed':raise RuntimeError('parent study failed; retain its failures')
            time.sleep(2)
        if (dest/'summary.csv').exists():return dict(**q,status='complete')
        with dest.with_suffix('.log').open('a') as log:
            ret=subprocess.run(q['command'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')).returncode
        return dict(**q,status='complete' if (dest/'summary.csv').exists() and ret==0 else 'failed',returncode=ret)
    rows=[];atomic_json(out/'status.json',dict(phase='running',pid=os.getpid(),expected=24))
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run,q):q for q in jobs}
        for future in concurrent.futures.as_completed(pending):
            try:row=future.result()
            except Exception as exc:row=dict(**pending[future],status='failed',error=repr(exc))
            rows.append(row);atomic_json(out/'trial_results.json',rows)
            print('REASSESSED',row['variant'],row['scene'],row['condition'],row['status'],len(rows),flush=True)
    failed=[q for q in rows if q['status']!='complete']
    atomic_json(out/'status.json',dict(phase='failed' if failed else 'complete',expected=24,finished=len(rows),failures=failed))
    assert not failed,failed


if __name__=='__main__':main()
