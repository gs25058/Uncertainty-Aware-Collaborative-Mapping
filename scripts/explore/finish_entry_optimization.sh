#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec ${PYTHON:-python} - <<'PY'
import datetime,fcntl,json,subprocess,sys,time
from pathlib import Path
root=Path('results/entry_optimization_20261109');report=Path('reports/entry_optimization_20261109');lock=(root/'finalization.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('finalization already active')
while True:
 p=root/'pipeline_status.json'
 if p.exists():
  q=json.load(open(p));phase=q['phase']
  if phase=='complete':sys.exit('already complete')
  if phase in ['execution_error','no_eligible_new_variant']:raise RuntimeError(q)
  if phase=='simulation_complete_pending_final_audit':break
 time.sleep(15)
def status(phase,**kw):
 p=root/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,updated=datetime.datetime.now().astimezone().isoformat(),**kw),indent=2)+'\n');tmp.replace(p)
status('final_audit_running');commands=[['scripts/explore/verify_entry_optimization.py','--root',str(root),'--report',str(report)],['scripts/explore/report_entry_optimization.py','--root',str(root),'--report',str(report)]]
for i,args in enumerate(commands):
 with (root/f'finalization_{i}.log').open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
for name in ['common_views_control','common_positions_control']:
 common=root/name/'status.json'
 if common.exists() and json.load(open(common))['phase']=='complete':
  with (root/(name+'_report.log')).open('w') as f:subprocess.run([sys.executable,'scripts/explore/report_entry_batch.py','--batch',str(root/name),'--baseline',str(root/name/'unknown'),'--out',str(report/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
status('complete',development_rounds=3,development_trials=128,validation_trials=100,independent_audit=True,all_failures_and_unknown_goals_retained=True);print('OPTIMIZATION, SEALED VALIDATION AND AUDIT COMPLETE',flush=True)
PY
