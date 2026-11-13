#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec ${PYTHON:-python} - <<'PY'
import csv,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('results/fire_recon_20261108_resume');report=Path('reports/fire_recon_20261108_resume');lock=(root/'post.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('postprocessing monitor already running')
def status(phase,**kw):
 p=root/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,**kw),indent=2)+'\n');tmp.replace(p)
while not (root/'main/complete.json').exists():
 p=root/'pipeline_status.json'
 if p.exists() and json.load(open(p))['phase']=='execution_failed':raise RuntimeError('execution failed; logs and checkpoints retained')
 time.sleep(20)
for p,h in json.load(open(root/'configuration_manifest.json')).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
status('postprocessing',runs=40,reused_runs=20,resumed_trials=20)
commands=[(['scripts/explore/summarize.py','--root',str(root/'main'),'--out',str(report/'data')],'summarize.log'),(['scripts/explore/post_recon_study_v2.py','--root',str(root),'--report',str(report)],'postprocess.log'),(['scripts/explore/analyze_fire_decisions.py','--root',str(root/'main'),'--out',str(report/'data')],'decisions.log'),(['scripts/explore/report_recon_study.py','--root',str(root),'--report',str(report)],'report.log'),(['scripts/explore/analyze_floor_routes.py','--root',str(root),'--report',str(report)],'floors.log')]
for args,name in commands:
 print('RUN',args,flush=True)
 with (root/'audit'/name).open('w') as f:subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT,check=True)
rows=list(csv.DictReader(open(report/'data/floor_goal_routes.csv')));s=['','## 층별 최종 경로 확인','', '| 조건 | 정책 | 층 | 유효/R | 정차단/B | 미확인 | 실내 촬영 | 유효 깊이 있는 촬영 |','|---|---|---|---|---|---|---|---|']
for q in rows:s.append(f"| {q['condition']} | {q['policy']} | {q['floor']} | {q['valid_routes']}/{q['reachable']} | {q['correct_blocked']}/{q['blocked']} | {q['unknown']} | {q['captures']} | {q['captures_with_valid_depth']} |")
s+=['','층은 사전에 고정한 목표 도착 층이다. 드론이 위층 높이에서 촬영한 것과 입구부터 사람 경로를 확인한 것을 구분한다. 전체 목표를 그대로 나눈 표이며 결과에 따라 목표/분모를 바꾸지 않았다.','', '## 실행 중단과 정확한 재개','', '2026-11-08 00:45 무렵 계산 프로세스가 종료되어 원래 보완 20회의 약96–128초 기록을 보존했다. 기록된 현재 시각·드론·카메라/동료 포즈가 일치하는 깊이만 재생해 내부 상태를 재구성하고 나머지 가상 시간을 실행했다. 캐시 관측은 각 프레임 ID당 한 번 적분했으며 새 독립 관측이나 추가 시간 예산으로 세지 않았다. 관측 가치·센서 수치·배정·이동·사람 판정의 알고리즘은 유지했다.','', '공통 runner의 선택형 실행 I/O에 확률을 보존하는 OctoMap .ot, 표면 누적 모멘트/투표/증거, 기체/예약 궤적, 카운터와 로그 위치를 저장하는 원자적 체크포인트를 추가했다. 파괴적인 .bt 형식은 쓰지 않는다. 중단 전 동결 코드의 연속 실행과 캐시 재생 중단/복구 실행을 비교해27관측의 지도 전체 배열·궤적·이벤트·후보/배정·목표 경로/지표가 정확히 같음을 검증했다(`validation/execution_v1/verification.json`).','', '유효 깊이0개 프레임의 원래 오차는 N/A이며 오차 풀링 가중치는0이다. raw CSV/깊이/지도/판정은 수정하지 않고 별도 count-aware 후처리를 적용했다. [0개 프레임 CSV](data/zero_valid_frames.csv). CPU 시간에는 재생과 상태 저장이 포함되므로 카메라를 매번 새로 렌더한 비용과 혼동하지 않는다. 주 시간축은 동일한180초의 가상 임무다.']
p=report/'RESULTS.md';p.write_text(p.read_text()+'\n'.join(s)+'\n')
pointer=Path('RESULTS_fire_recon.md');pointer.write_text(pointer.read_text().replace('reports/fire_recon_20261107/','reports/fire_recon_20261108_resume/'))
files=[Path('scripts/explore/finish_resumable_recon.sh'),Path('covor/explore/execution_state.py'),Path('scripts/explore/run_recon_resumable_matrix.py')];(root/'execution_recovery_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
status('complete',runs=40,reused_runs=20,resumed_trials=20,audited=True);print('ALL RESUMED RESULTS COMPLETE',flush=True)
PY
