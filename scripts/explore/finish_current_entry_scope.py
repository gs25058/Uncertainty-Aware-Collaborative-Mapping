#!/usr/bin/env python3
"""Wait for already-started batches, then audit/report. Never start experiments."""
import datetime,fcntl,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from covor.explore.execution_state import atomic_json
primary=Path('results/entry_optimization_20261109');control=Path('results/entry_control_20261109')
lock=(primary/'current_scope_finalization.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('current scope finalizer already active')
assert (primary/'user_scope_finish_current.json').exists()
while True:
    ready=True
    for b in [primary/'round03',control/'development']:
        q=json.load(open(b/'status.json'))
        if q['phase']=='failed':raise RuntimeError(('retain all failed cases for diagnosis',str(b),q))
        ready&=q['phase'] in ['complete','user_scope_closed']
    ready&=(control/'diagnostics/school_fixed_trajectory_ideal/ideal_depth/summary.json').exists()
    ready&=(control/'diagnostics/school_support_sampling/comparison.csv').exists()
    if ready:break
    time.sleep(15)
for root in [primary,control]:atomic_json(root/'pipeline_status.json',dict(phase='current_scope_audit',updated=datetime.datetime.now().astimezone().isoformat(),new_experiments_disabled=True))
try:
    with (primary/'current_scope_report.log').open('w') as f:
        subprocess.run([sys.executable,'scripts/explore/report_current_entry_work.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
    checked=json.load(open('reports/entry_optimization_20261109/current/audit.json'))
    for root in [primary,control]:atomic_json(root/'pipeline_status.json',dict(phase='current_scope_complete_pending_commit',updated=datetime.datetime.now().astimezone().isoformat(),new_full_missions=checked['new_full_missions'],not_started=len(checked['not_started']),heldout_validation_not_executed=True,commit_not_yet_done=True))
except Exception as exc:
    for root in [primary,control]:atomic_json(root/'pipeline_status.json',dict(phase='current_scope_audit_failed',updated=datetime.datetime.now().astimezone().isoformat(),error=str(exc),raw_results_preserved=True))
    raise
print('CURRENT EXPERIMENTS AND REPORT FINISHED; READY FOR HUMAN-READABLE REVIEW AND GIT COMMIT/PUSH',flush=True)
