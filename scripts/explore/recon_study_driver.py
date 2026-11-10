#!/usr/bin/env python3
"""Complete geometry gates, reuse exact old trials, run all changed trials."""
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_recon_20261107');p.add_argument('--report',default='reports/fire_recon_20261107');p.add_argument('--config',default='configs/fire_recon_20261107.json');p.add_argument('--workers',type=int,default=20);a=p.parse_args();root=Path(a.root);report=Path(a.report);report.mkdir(parents=True,exist_ok=True)
def status(phase,**kw):(root/'pipeline_status.json').write_text(json.dumps(dict(phase=phase,**kw),indent=2)+'\n')
def run(args,name):
 print('RUN',args,flush=True)
 with (root/'intake'/name).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
cfg=json.load(open(a.config));unchanged=[int(q['scene']) for q in csv.DictReader(open(root/'intake/asset_geometry_checks.csv')) if int(q['added_landings'])==0]
reusable=[Path('results/fire_closed_loop_20261107/main')/f'scene{i}_seed{s}_{c}_{p0}'/'summary.json' for i in unchanged for s in cfg['run_seeds'] for c in cfg['conditions'] for p0 in cfg['policies']]
status('waiting_for_corrected_geometry_gates_and_reusable_trials')
while True:
 for name in ['suite_gates.log']:
  if 'Traceback' in (root/'intake'/name).read_text():status('geometry_gate_failed',log=name);raise RuntimeError(name)
 for f in Path('results/fire_closed_loop_20261107/main').glob('*.log'):
  if 'Traceback' in f.read_text():status('original_run_failed',log=str(f));raise RuntimeError(f)
 gs=json.load(open(root/'gates_complete/status.json'))
 if gs['phase']=='complete' and all(f.exists() for f in reusable):break
 time.sleep(25)
status('reusing_physically_identical_two_storey_runs')
run(['scripts/explore/reuse_recon_runs.py','--config',a.config,'--out',str(root/'main')],'reuse_runs.log')
reuse=json.load(open(root/'main/reuse_checks.json'));cfg=json.load(open(a.config));expected=len(cfg['scene_seeds'])*len(cfg['run_seeds'])*len(cfg['conditions'])*len(cfg['policies']);status('main_matrix_running',expected_runs=expected,reused_runs=reuse['reused_runs'],fresh_runs=expected-reuse['reused_runs'])
run(['scripts/explore/run_recon_matrix.py','--config',a.config,'--gates',str(root/'gates_complete'),'--out',str(root/'main'),'--workers',str(a.workers)],'matrix.log')
status('all_runs_complete_postprocessing',expected_runs=expected,reused_runs=reuse['reused_runs']);(root/'reference_refinements.json').write_text('[]\n')
run(['scripts/explore/summarize.py','--root',str(root/'main'),'--out',str(report/'data')],'summarize.log')
run(['scripts/explore/post_recon_study.py','--root',str(root),'--report',str(report)],'postprocess.log')
run(['scripts/explore/analyze_fire_decisions.py','--root',str(root/'main'),'--out',str(report/'data')],'decision_coverage.log')
run(['scripts/explore/report_recon_study.py','--root',str(root),'--report',str(report)],'report.log')
status('complete',runs=expected,reused_runs=reuse['reused_runs'],fresh_runs=expected-reuse['reused_runs']);print('RECON STUDY COMPLETE',flush=True)
