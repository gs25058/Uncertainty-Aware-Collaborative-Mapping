#!/usr/bin/env python3
"""Generate result documents from completed, audited observations only."""
import argparse,csv,hashlib,json,subprocess,sys,time
from collections import Counter
from pathlib import Path
csv.field_size_limit(sys.maxsize)
p=argparse.ArgumentParser();p.add_argument('--root',default='results/fire_closed_loop_20261107');p.add_argument('--report',default='reports/fire_closed_loop_20261107');p.add_argument('--wait',action='store_true');a=p.parse_args();root=Path(a.root);report=Path(a.report)
while a.wait and not (root/'post_complete.json').exists():time.sleep(30)
assert json.load(open(root/'post_complete.json'))['audit_passed']
def read(name):return list(csv.DictReader(open(report/'data'/name)))
def n(q,k):return int(float(q[k]))
def ratio(q,num,den):
 v=n(q,num);d=n(q,den);return f'{v}/{d} ({100*v/d:.2f}%)' if d else f'{v}/0 (N/A)'
def mdtable(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
groups=read('summary_by_method.csv');runs=read('runs_final.csv');activity=read('policy_activity.csv');paths=read('paths_final_compact.csv');ops=read('operations.csv');temporal=read('temporal_goal_summary.csv');pairs=read('paired_comparison.csv');maps=read('map_counts.csv');qc=json.load(open(root/'audit/main_checks.json'));gate=read('reference_and_complete_gate.csv');sensor=read('sensor_observations.csv')
main_rows=[]
for q in groups:
 main_rows.append([q['condition'],q['policy'],ratio(q,'valid_paths','reachable_denominator'),ratio(q,'correct_blocked','blocked_denominator'),f"{100*float(q['balanced_score']):.2f}%",ratio(q,'invalid_paths','checked_path_denominator'),q['unknown'],ratio(q,'decided','total_goal_denominator')])
text=['# 복층 확대 화재 건물 폐루프 실험 결과','',f'**10개 건물 × 무연기/강한 저하 × unknown/entry, 총 {len(runs)}회**를 각각 4대·180초·4Hz·perfect_odometry로 완료했다. 주 비교는 공통 다중 드론 탐색 파이프라인에서 관측 가치 함수 비교다. 소스 77개와 설정을 시작 전에 동결했고, 완료 후 해시 일치와 기록 입력 재생을 확인했다.','',mdtable(['조건','정책','유효 경로/도달 가능','올바른 차단/차단','균형 점수','무효/검증 경로','미확인','판정/전체'],main_rows),'','분모는 조건·정책별 같은 218개 방 구역이다(기준 도달 가능 112, 차단 106). 무연기도 정상 깊이 오차가 있는 조건이다. 경로가 없는 목표는 자동 차단으로 바꾸지 않았다. 잘못된 경로 비율이 0이어도 적은 제안 수와 낮은 판정률을 함께 해석해야 한다.']
conclusions=[]
for c in ('clean','strong'):
 u=next(q for q in groups if q['condition']==c and q['policy']=='unknown');e=next(q for q in groups if q['condition']==c and q['policy']=='entry');dv=n(e,'valid_paths')-n(u,'valid_paths');db=n(e,'correct_blocked')-n(u,'correct_blocked');di=n(e,'invalid_paths')-n(u,'invalid_paths');conclusions.append(f"- {c}: entry−unknown 유효 경로 {dv:+d}개, 올바른 차단 {db:+d}개, 무효 경로 {di:+d}개, 균형 점수 {(float(e['balanced_score'])-float(u['balanced_score']))*100:+.2f}%p.")
text+=['','## 실제 관측된 차이','',*conclusions,'','이번 10장면·실행 seed 0의 고정 비교다. 여러 실행 seed의 반복, 통신/포즈 오차, 열·가스 위험은 포함하지 않았다. 실제 화재 안전성이나 실제 장비 성능을 검증한 결과가 아니다.']
activity_rows=[]
for c in ('clean','strong'):
 for p0 in ('unknown','entry'):
  aa=[q for q in activity if q['condition']==c and q['policy']==p0];pp=[q for q in paths if q['condition']==c and q['policy']==p0];total=sum(float(q['mission_seconds']) for q in aa);fallback=sum(float(q['fallback_seconds']) for q in aa)
  activity_rows.append([c,p0,f'{fallback:.0f}/{total:.0f}s' if p0=='entry' else 'N/A',sum(n(q,'positive_entry_epochs') for q in aa),sum(n(q,'epochs') for q in aa),sum(n(q,'assignments') for q in aa),sum(q['valid_over_1m']=='True' for q in pp),sum(n(q,'upper_floor_trajectory_samples') for q in aa)])
text+=['','## 정책이 작동한 범위와 경로 길이','',mdtable(['조건','정책','entry 대체 규칙 시간','양의 entry 후보가 있던 배정 시각','전체 배정 시각','실제 배정','1m 초과 유효 경로','상층 실제 궤적 표본'],activity_rows),'','대체 규칙 시간은 상태를 평가한 4초 배정 구간의 합이다. 양의 후보가 있어도 비행 경로·예약·최소 작업 유지 조건 때문에 실제 배정을 못 할 수 있다. entry 값이 없는 동안 두 정책은 같은 unknown 규칙을 쓴다. 목표까지 가능 경로가 생기기 전의 넓은 미관측 구간은 현재 구간 절단 점수로 중요도를 부여하기 어렵다. 이를 개선했다고 간주하지 않고 그대로 측정했다.','', '목표를 방 전체 바닥 영역으로 고정했으므로 입구와 겹친 목표의 한 점/짧은 경로도 주 지표에 포함된다. 보조 열은 길이 1m 초과만 세어 이 효과를 구분한다. 이 보조값은 다른 목표나 더 엄격한 분모로 실험을 다시 선정한 결과가 아니다.','']
operation_rows=[]
for g in groups:
 rr=[q for q in runs if q['condition']==g['condition'] and q['policy']==g['policy']];oo=[q for q in ops if any(r['run']==q['run'] for r in rr)]
 operation_rows.append([g['condition'],g['policy'],sum(n(q,'captures') for q in rr),f"{sum(float(q['travel_distance']) for q in rr):.2f}",sum(n(q,'failed_drones') for q in rr),sum(n(q,'geometry_collisions') for q in oo),sum(n(q,'teammate_collision_drone_events') for q in oo),sum(n(q,'planning_failures') for q in rr),f"{sum(float(q['reservation_wait_executed']) for q in oo):.2f}"])
text+=['## 실제 운용과 실패','',mdtable(['조건','정책','촬영 수','팀 이동 거리 합[m]','실패 기체','형상 충돌','동료 충돌 기체 이벤트','계획 실패 시도','실행 예약 대기[s]'],operation_rows),'','동료 충돌은 기체별 이벤트 수로 표시한다. 계획 실패는 후보/배정 시도의 실패이며 임무 전체 실패와 다르다. 실패 드론을 재생성하지 않았고 모든 실행을 최종 분모에 남겼다. 진입 대기는 회당 12 drone·s(출발 전 tick 반올림 항목은 operations.csv 참조)이며 임무 180초 안에 포함된다. CPU 계산 시간은 시뮬레이션 180초보다 길며 실시간 실행을 주장하지 않는다.']
transition_rows=[]
for c in ('clean','strong'):
 for p0 in ('unknown','entry'):
  names={q['run'] for q in runs if q['condition']==c and q['policy']==p0};tt=[q for q in temporal if q['run'] in names]
  transition_rows.append([c,p0,*[sum(n(q,k) for q in tt) for k in ['unknown_to_valid_events','unknown_to_correct_blocked_events','new_wrong_route_events','revoked_passable_events','revoked_invalid_events','lost_valid_events']],sum(q['lost_earlier_valid_at_final']=='True' for q in tt)])
text+=['','## 시각별 전이와 최종 상태','',mdtable(['조건','정책','미확인→유효','미확인→정차단','새 무효 안내','통과 취소','무효 안내 취소','유효 안내 상실 사건','이전에 유효했으나 최종 상실 목표'],transition_rows),'','전이 열은 8초 평가와 마지막 시각 사이의 **사건 수**다. 같은 목표가 여러 번 바뀌면 반복 집계된다. 마지막 열만 최종 상태 기준 목표 수다. 최종 표의 경로 수와 사건 수를 혼용하지 않는다.','', '![시간별 목표·경로 지표](figures/time_metrics.png)']
scene_rows=[]
for i in range(10):
 for c in ('clean','strong'):
  u=next(q for q in runs if n(q,'scene_seed')==i and q['condition']==c and q['policy']=='unknown');e=next(q for q in runs if n(q,'scene_seed')==i and q['condition']==c and q['policy']=='entry');scene_rows.append([i,c,f"{u['valid_confirmed']} / {e['valid_confirmed']} / {u['reachable_denominator']}",f"{u['correct_blocked']} / {e['correct_blocked']} / {u['blocked_denominator']}",f"{u['invalid_paths']} / {e['invalid_paths']}",f"{u['unknown']} / {e['unknown']}"])
text+=['','## 장면별 최종 결과','',mdtable(['장면','조건','유효 unknown / entry / R','정차단 unknown / entry / B','무효 unknown / entry','미확인 unknown / entry'],scene_rows),'','장면 5의 B=0은 N/A이며 균형 점수의 장면 평균에서 몰래 제외하거나 0으로 바꾸지 않았다. 주 표는 모든 장면의 고정 분모를 합산한 점수다.','', '![같은 높이의 학교 지도 비교](figures/school_map_floor_comparison.png)','', '회색은 미관측, 흰색은 관측 FREE, 짙은색은 OCC다. 층을 구분한 동일 z 단면이며 여러 층을 한 XY 평면에 합치지 않았다.']
text+=['','## 앞단 검증과 해석의 한계','',f"완전 관측의 공통 판정기에서 유효 경로 {sum(n(q,'complete_valid') for q in gate)}/112, 올바른 차단 {sum(n(q,'complete_correct_blocked') for q in gate)}/106, 무효 경로 0이었다. 차단 6개는 완전 지도에서도 보수적 증명 부족으로 미확인이다. 드론 부분 지도에서의 낮은 확인률을 정답 경로 생성기 자체의 전면 실패로 해석하지 않는다.",'','정답의 차단 라벨은 독립 메시 기반 10cm 지지 로드맵 기준이다. 연속 공간의 모든 미세 우회를 배제한 해석적 증명은 아니다. 정책 제시 경로의 독립 검증과 기준 라벨 모순을 별도 감사했다: **'+str(len(qc['reference_contradictions']))+'건**. 완전 지도 합격, 부분 지도의 확인, 기준 라벨 모순은 각각 별개의 검사다.','',f"실제 관측 입력 {qc['actual_frames']:,}장, 80개 선택 프레임의 깊이/sigma/valid 재생 일치, 모든 정책 쌍의 첫 관측 일치, 동결 소스 {qc['sources_unchanged']}개 불변을 확인했다. 저장된 깊이 276장만으로 20초 지도를 다시 적분해 모든 배열이 비트 단위로 같았다.",'','확대 화재 75개 소스가 같은 3D 연기장과 깊이 저하에 연결돼 있다. 이는 무차원 확산장과 지정된 오차 분포이며 실제 화재 농도로 환산하지 않는다. 온도·가스·건물 붕괴의 동역학, 화염 자체의 깊이 산란, 실제 카메라 영상 처리는 제외했다. 카메라는 낙관적 가상 모델이고 완전 위치/즉시 통신을 가정한다.','', '![고정 위치의 이상·오염 깊이와 valid](figures/source_local_depth.png)','','구체적인 수정 범위와 실패 기록은 [FIXES](FIXES.md), 센서·기체·사람·분모 정의는 [METHOD](METHOD.md), 실행/재생은 [REPRODUCE](REPRODUCE.md)에 있다.','', '## 결과 파일','', '- [방식별 집계 CSV](data/summary_by_method.csv) · [40회 최종 CSV](data/runs_final.csv) · [쌍별 차이](data/paired_comparison.csv)\n- [목표·경로 상세](data/targets_final.csv) · [짧은 경로 표](data/paths_final_compact.csv) · [전이 집계](data/temporal_goal_summary.csv)\n- [정책 활성/대체 규칙](data/policy_activity.csv) · [실제 배정 차이](data/paired_assignment_audit.csv) · [운용/실패](data/operations.csv)\n- [시각별 지표](data/metrics_time_all.csv) · [깊이 유효율/오차](data/sensor_observations.csv) · [지도 수](data/map_counts.csv)\n- 재생 입력: `results/fire_closed_loop_20261107/main/scene*_seed0_{clean,strong}_{unknown,entry}/`의 깊이 NPZ·frames.csv·trajectories.csv·maps·replay_manifest.json\n- 선택 로그: 같은 실행 폴더의 candidates.jsonl, assignments.jsonl, decisions.jsonl. 후보 선택 뒤 관측 생성 순서는 selection_before_sensor에 명시한다.\n- 감사: `results/fire_closed_loop_20261107/audit/main_checks.json`, 기준/검증: `reference_final`, `reference_checks`, `gates_complete`.']
(report/'RESULTS.md').write_text('\n'.join(text)+'\n')
Path('RESULTS_fire_closed_loop.md').write_text('# 복층 화재 10장면 폐루프 실험\n\n40회(4대·180초·clean/strong·unknown/entry)를 실제 실행하고 감사했다.\n\n'+mdtable(['조건','정책','유효 경로/도달 가능','올바른 차단/차단','균형 점수','무효/검증 경로','미확인','판정/전체'],main_rows)+'\n\n[결과·한계](reports/fire_closed_loop_20261107/RESULTS.md) · [방법](reports/fire_closed_loop_20261107/METHOD.md) · [실행](reports/fire_closed_loop_20261107/REPRODUCE.md)\n')
status=Path('CURRENT_STATUS.md');marker='## 2026-11-07 복층 확대 화재 폐루프 실험 완료'
if marker not in status.read_text():status.write_text(status.read_text()+'\n\n'+marker+'\n\n10장면×clean/strong×unknown/entry의 40회, 4대·180초·4Hz를 완료했다. 112 도달 가능/106 차단 목표를 고정했고 소스 77개 불변과 실제 깊이 재생을 검사했다. 수치와 실패/대체 규칙은 [최종 결과](reports/fire_closed_loop_20261107/RESULTS.md), [방식별 CSV](reports/fire_closed_loop_20261107/data/summary_by_method.csv)에 있다. 원본 건물/이전 실패 기록은 보존했다.\n')
helpers={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path('scripts/explore')/n for n in ['post_fire_study.py','report_fire_study.py','analyze_fire_decisions.py','replay_fire_checkpoint.py','diagnose_fire_paths.py','plot_fire_sensor.py']]};(root/'postprocessing_source_manifest.json').write_text(json.dumps(helpers,indent=2)+'\n');print('RESULT REPORTS WRITTEN',flush=True)
