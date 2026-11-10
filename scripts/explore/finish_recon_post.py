#!/usr/bin/env python3
"""Complete postprocessing with the count-aware adapter, preserving frozen code.

The original matrix driver/analysis are unchanged while trials run. If the old
postprocessor hits a legitimate zero-valid row, this separate continuation
records the failure and completes only aggregation/audit/reporting.
"""
import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path('results/fire_recon_20261107');report=Path('reports/fire_recon_20261107')
def run(args,log):
 print('RUN',args,flush=True)
 with (root/'intake'/log).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
while True:
 status=json.load(open(root/'pipeline_status.json'))
 if status['phase']=='complete':print('original postprocessor completed without correction',flush=True);sys.exit(0)
 log=root/'intake/study_driver_v2.log';text=log.read_text()
 if 'CalledProcessError' in text:
  if not (root/'main/complete.json').exists():raise RuntimeError('matrix did not complete; inspect original failure')
  post=(root/'intake/postprocess.log').read_text()
  if "could not convert string to float: ''" not in post:raise RuntimeError('different postprocessing failure; inspect postprocess.log')
  break
 time.sleep(25)
(root/'pipeline_status.json').write_text(json.dumps(dict(phase='count_aware_postprocessing',original_failure_log='intake/postprocess.log'),indent=2)+'\n')
run(['scripts/explore/post_recon_study_v2.py','--root',str(root),'--report',str(report)],'postprocess_v2.log')
run(['scripts/explore/analyze_fire_decisions.py','--root',str(root/'main'),'--out',str(report/'data')],'decision_coverage.log')
run(['scripts/explore/report_recon_study.py','--root',str(root),'--report',str(report)],'report.log')
path=report/'RESULTS.md';path.write_text(path.read_text()+'\n## 유효 깊이 0개 프레임의 집계 보완\n\n최초 후처리는 관측 유효 픽셀이 0개인 프레임의 빈 오차 칸에서 실패했다. raw CSV·깊이·지도·후보·판정·실행 소스는 변경하지 않았다. 별도 `post_observation_audit.py`는 이 프레임의 오차를 N/A로 남기고 표본 가중치 0으로 집계한다. 촬영 수와 비용에는 포함한다. [해당 프레임](data/zero_valid_frames.csv), 최초 실패 `intake/postprocess.log`, 수정 감사 `intake/postprocess_v2.log`, `postprocessing_zero_valid_manifest.json`을 보존한다.\n')
files=[Path('scripts/explore')/p for p in ['finish_recon_post.py','post_recon_study_v2.py','post_observation_audit.py']];(root/'postprocessing_fix_manifest.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n');reuse=json.load(open(root/'main/reuse_checks.json'));(root/'pipeline_status.json').write_text(json.dumps(dict(phase='complete',runs=40,reused_runs=reuse['reused_runs'],fresh_runs=40-reuse['reused_runs'],count_aware_postprocessing=True),indent=2)+'\n');print('ALL RECON RESULTS AND AUDITS COMPLETE',flush=True)
