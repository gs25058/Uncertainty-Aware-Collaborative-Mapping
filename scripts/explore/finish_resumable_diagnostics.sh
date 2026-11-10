#!/usr/bin/env bash
set -euo pipefail
cd /src/gs25058/cr_RNE/covor_slam
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec /src/gs25058/miniconda3/envs/covor/bin/python - <<'PY'
import concurrent.futures,csv,fcntl,hashlib,json,subprocess,sys,time
from collections import Counter
from pathlib import Path
csv.field_size_limit(sys.maxsize)
root=Path('results/fire_recon_20261108_resume');report=Path('reports/fire_recon_20261108_resume')
lock=(root/'diagnostics.lock').open('a+')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:sys.exit('diagnostic monitor already running')
if (root/'diagnostics_complete.json').exists():sys.exit('diagnostics already complete')
while json.load(open(root/'pipeline_status.json'))['phase']!='complete':time.sleep(20)
names=[f'scene1_seed0_{c}_{p}' for c in ['clean','strong'] for p in ['unknown','entry']]
# School was fixed as the preflight scene before policy outcomes were known.
def one(name):
 out=root/'diagnostics'/name;out.mkdir(parents=True,exist_ok=True)
 with (out/'diagnosis.log').open('w') as f:
  subprocess.run([sys.executable,'scripts/explore/diagnose_fire_paths.py','--run',str(root/'main'/name),'--out',str(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
 return list(csv.DictReader(open(out/'reference_route_first_break.csv')))
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for part in pool.map(one,names):rows.extend(part)
p=report/'data/reference_route_first_break.csv'
if rows:
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
invalid=[q for q in csv.DictReader(open(report/'data/targets_final.csv')) if q['path_status']=='invalid']
p=report/'data/invalid_route_examples.csv'
with p.open('w',newline='') as f:
 fields=['run','goal','reference_reachable','verdict','path_status','path_reason','first_failure'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:q[k] for k in fields} for q in invalid)
lines=['','## 부분 지도의 첫 단절과 무효 안내 사례','','사전 연결 검사에 사용한 학교(scene1)를 그대로 선택해 clean/strong × unknown/entry의 최종 지도에 독립 기준 경로를 대조했다. 아래는 시스템이 아직 통과 가능으로 답하지 않은 도달 가능 목표들에 대해, 그 특정 기준 경로에서 처음 부족한 단계다. 대체 경로 전체의 불가능 증명이 아니며 정답을 지도나 정책에 반영하지 않았다.','', '| 실행 | 미확인 양성 목표 | 첫 단절 단계와 개수 |','|---|---|---|']
for name in names:
 subset=[q for q in rows if q['run']==name];counts=Counter(json.loads(q['first_failure'])['stage'] if q['first_failure'] else 'reference_route_cells_pass_but_other_graph_condition' for q in subset)
 lines.append('| '+name+' | '+str(len(subset))+' | '+', '.join(f'{k}: {v}' for k,v in sorted(counts.items()))+' |')
lines+=['','[첫 실패 좌표·발판 증거·높이·자세의 상세 CSV](data/reference_route_first_break.csv). 완전 관측에서는 학교의 유효 경로20/20과 차단4/4가 확인된 것과 구분한다.','']
if invalid:
 q=invalid[0];lines += [f"무효 안내의 한 예는 `{q['run']}` / `{q['goal']}`다. 목표의 기준 도달 가능 여부는 `{q['reference_reachable']}`지만, 실제 제시 경로는 `{q['first_failure']}`에서 `{q['path_reason']}`로 실패했다. 목표에 다른 경로로 도달할 수 있어도 이 안내는 무효로 유지했다. 전체 최종 무효 안내 {len(invalid)}건은 [실패 위치 CSV](data/invalid_route_examples.csv)에 있다.",'']
else:lines+=['최종 무효 안내는0건이다. 제안 수와 유효 경로 확인율을 함께 보고하며 무효 안내0만으로 정책 우위를 주장하지 않는다.','']
p=report/'RESULTS.md';marker='## 부분 지도의 첫 단절과 무효 안내 사례'
if marker not in p.read_text():p.write_text(p.read_text()+'\n'.join(lines)+'\n')
files=[Path('scripts/explore/finish_resumable_diagnostics.sh'),Path('scripts/explore/diagnose_fire_paths.py')]
(root/'diagnostics_complete.json').write_text(json.dumps(dict(runs=names,rows=len(rows),invalid_final_routes=len(invalid),no_planning_feedback=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
print('POSTRUN DIAGNOSTICS COMPLETE',flush=True)
PY
