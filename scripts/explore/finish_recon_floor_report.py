#!/usr/bin/env python3
"""Add pre-fixed floor strata after all audits/reports finish."""
import csv,hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path('results/fire_recon_20261107');report=Path('reports/fire_recon_20261107')
while json.load(open(root/'pipeline_status.json'))['phase']!='complete':time.sleep(25)
with (root/'intake/floor_analysis.log').open('w') as f:subprocess.run([sys.executable,'scripts/explore/analyze_floor_routes.py','--root',str(root),'--report',str(report)],stdout=f,stderr=subprocess.STDOUT,check=True)
rows=list(csv.DictReader(open(report/'data/floor_goal_routes.csv')));lines=['','## 층별 경로 확인과 실제 관측','', '| 조건 | 정책 | 층 | 유효 경로/R | 올바른 차단/B | 미확인 | 실내 촬영 | 유효 깊이가 있는 촬영 |','|---|---|---|---|---|---|---|---|']
for q in rows:lines.append(f"| {q['condition']} | {q['policy']} | {q['floor']} | {q['valid_routes']}/{q['reachable']} | {q['correct_blocked']}/{q['blocked']} | {q['unknown']} | {q['captures']} | {q['captures_with_valid_depth']} |")
lines+=['','분모는 사전에 고정한 같은 목표를 도착 층으로 나눈 값이다. 층별 결과를 보고 목표를 바꾸거나 제외하지 않았다. 실내 관측은 공통 탐색 외곽 XY 안의 실제 body pose로 분류하고, 기체가 해당 층 높이에 도달한 것과 사람의 입구–목표 전체 경로를 확인한 것을 구분한다. 단일 촬영의 valid 픽셀이 있어도 발판·몸 공간 증거가 완성됐다는 뜻은 아니다.','', '[층별 목표 CSV](data/floor_goal_routes.csv) · [실행별 실제 층 관측·체류 CSV](data/floor_observation_coverage.csv).']
p=report/'RESULTS.md';s=p.read_text();marker='## 층별 경로 확인과 실제 관측'
if marker not in s:p.write_text(s+'\n'.join(lines)+'\n')
files=[Path(__file__),Path('scripts/explore/analyze_floor_routes.py')];(root/'postprocessing_floor_manifest.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n');(root/'floor_report_complete.json').write_text(json.dumps(dict(complete=True,groups=len(rows)))+'\n');print('FLOOR REPORT COMPLETE',flush=True)
