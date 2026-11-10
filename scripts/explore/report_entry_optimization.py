#!/usr/bin/env python3
"""Report every sealed validation outcome, without tuning on validation."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from covor.explore.runner import csv_write,write_json
csv.field_size_limit(sys.maxsize)
p=argparse.ArgumentParser();p.add_argument('--root',default='results/entry_optimization_20261109');p.add_argument('--report',default='reports/entry_optimization_20261109');a=p.parse_args();root=Path(a.root);report=Path(a.report);data=report/'data';rows=list(csv.DictReader(open(data/'validation_runs.csv')));targets=list(csv.DictReader(open(data/'validation_targets.csv')));winner=json.load(open(root/'winner_frozen.json'));checks=json.load(open(root/'audit/validation_checks.json'));groups=[]
def i(q,k):return int(float(q[k]))
def group(rs,seed,condition,label,split):
 V=sum(i(q,'valid_confirmed') for q in rs);R=sum(i(q,'reachable_denominator') for q in rs);C=sum(i(q,'correct_blocked') for q in rs);B=sum(i(q,'blocked_denominator') for q in rs);I=sum(i(q,'invalid_paths') for q in rs);P=sum(i(q,'verified_path_denominator') for q in rs);D=sum(i(q,'decided') for q in rs);G=sum(i(q,'total_goals') for q in rs)
 return dict(seed=seed,condition=condition,label=label,split=split,runs=len(rs),valid=V,reachable=R,correct_blocked=C,blocked=B,balanced=.5*(V/R+C/B) if R and B else '',invalid=I,checked_paths=P,wrong_ratio=I/P if P else 'N/A',proposed=sum(i(q,'proposed_paths') for q in rs),unverifiable=sum(i(q,'unverifiable_paths') for q in rs),false_blocked=sum(i(q,'false_blocked') for q in rs),unknown=G-D,decided=D,goals=G,decision_rate=D/G,captures=sum(i(q,'captures') for q in rs),travel_m=sum(float(q['travel_distance']) for q in rs),failed_drones=sum(i(q,'failed_drones') for q in rs),fallback_seconds=sum(float(q['fallback_seconds']) for q in rs))
for seed in ['2','3','pooled']:
 for condition in ['clean','strong']:
  for label in ['unknown','legacy','optimized']:
   if seed=='pooled' and label=='legacy':continue
   for split in ['all','reserved_geometry','development_geometry_new_seed']:
    rs=[q for q in rows if (seed=='pooled' or q['run_seed']==seed) and q['condition']==condition and q['label']==label and (split=='all' or q['split']==split)]
    if rs:groups.append(group(rs,seed,condition,label,split))
csv_write(data/'validation_summary.csv',groups)
pairs=[]
for q in rows:
 if q['label']!='optimized':continue
 b=next(x for x in rows if x['label']=='unknown' and x['scene_seed']==q['scene_seed'] and x['run_seed']==q['run_seed'] and x['condition']==q['condition'])
 pairs.append(dict(scene=q['scene_seed'],seed=q['run_seed'],condition=q['condition'],split=q['split'],valid_delta=i(q,'valid_confirmed')-i(b,'valid_confirmed'),blocked_delta=i(q,'correct_blocked')-i(b,'correct_blocked'),invalid_delta=i(q,'invalid_paths')-i(b,'invalid_paths'),false_blocked_delta=i(q,'false_blocked')-i(b,'false_blocked'),failed_delta=i(q,'failed_drones')-i(b,'failed_drones'),unknown_delta=i(q,'unknown')-i(b,'unknown')))
csv_write(data/'validation_paired.csv',pairs)
floor=[]
for condition in ['clean','strong']:
 for label in ['unknown','optimized']:
  for level in [1,2,3]:
   rr=[q for q in targets if q['condition']==condition and q['label']==label and q['goal'].startswith(f'L{level}_')]
   if rr:floor.append(dict(condition=condition,label=label,floor=level,reachable=sum(q['reference_reachable']=='True' for q in rr),blocked=sum(q['reference_reachable']=='False' for q in rr),valid=sum(q['path_status']=='valid' for q in rr),correct_blocked=sum(q['reference_reachable']=='False' and q['verdict']=='blocked' for q in rr),invalid=sum(q['path_status']=='invalid' for q in rr),unknown=sum(q['verdict']=='unverified' for q in rr)))
csv_write(data/'validation_floors.csv',floor)
lengths=[]
for condition in ['clean','strong']:
 for label in ['unknown','optimized']:
  rr=[q for q in targets if q['condition']==condition and q['label']==label]
  lengths.append(dict(condition=condition,label=label,valid=sum(q['path_status']=='valid' for q in rr),single_node_valid=sum(q['valid_single_node']=='True' for q in rr),over_1m_valid=sum(q['valid_over_1m']=='True' for q in rr)))
csv_write(data/'validation_route_lengths.csv',lengths)
time_rows=list(csv.DictReader(open(data/'validation_time.csv')));curves=[]
for condition in ['clean','strong']:
 for label in ['unknown','optimized']:
  subset=[q for q in time_rows if q['condition']==condition and q['label']==label]
  for t in sorted({float(q['time']) for q in subset}):
   rs=[q for q in subset if float(q['time'])==t];R=sum(i(q,'reachable_denominator') for q in rs);B=sum(i(q,'blocked_denominator') for q in rs);G=sum(i(q,'total_goals') for q in rs)
   curves.append(dict(condition=condition,label=label,time=t,valid_rate=sum(i(q,'valid_confirmed') for q in rs)/R,blocked_rate=sum(i(q,'correct_blocked') for q in rs)/B,invalid=sum(i(q,'invalid_paths') for q in rs),decision_rate=sum(i(q,'decided') for q in rs)/G))
csv_write(data/'validation_time_curves.csv',curves)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,4,figsize=(17,7),constrained_layout=True)
for row,condition in enumerate(['clean','strong']):
 for label in ['unknown','optimized']:
  rr=[q for q in curves if q['condition']==condition and q['label']==label]
  for col,key in enumerate(['valid_rate','blocked_rate','invalid','decision_rate']):axs[row,col].plot([q['time'] for q in rr],[q[key] for q in rr],label=label)
 for col,title in enumerate(['Valid / reachable','Correct blocked / blocked','Invalid proposed routes','Decided / all goals']):axs[row,col].set_title(condition+': '+title);axs[row,col].set_xlabel('Simulation time [s]');axs[row,col].grid(alpha=.2);axs[row,col].legend()
(report/'figures').mkdir(exist_ok=True);fig.savefig(report/'figures/validation_time_metrics.png',dpi=150);plt.close(fig)
def table(header,rs):return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rs])
main=[]
for q in groups:
 if q['split']=='all':main.append([q['seed'],q['condition'],q['label'],f"{q['valid']}/{q['reachable']}",f"{q['correct_blocked']}/{q['blocked']}",f"{100*q['balanced']:.2f}%" if q['balanced']!='' else 'N/A',f"{q['invalid']}/{q['checked_paths']}",f"{q['unknown']}/{q['goals']}",f"{q['decided']}/{q['goals']}",q['false_blocked'],q['failed_drones']])
lines=['# Entry 목적함수 반복 개선: 실제 개발·검증 결과','',f"개발에서 선택한 설정은 **{winner['variant']}** ({winner['batch']})다. 설정과 수학 소스를 검증 전에 고정했다.4대·가상180초·4Hz·perfect_odometry·동일218목표다. 새로운 실행/잡음 seed2/3의100회 검증을 완료했다.4개 개발 장면과 이번 조정에 쓰지 않은6개 장면을 나눴다. 이6개도 과거 연구에서 이미 사용한 라이브러리이므로 완전히 처음 보는 건물이라고 부르지 않는다.",'',table(['시드','조건','정책','유효/R','정차단/B','균형','무효/검증 제안','미확인/G','판정/G','차단 오판','실패 기체'],main),'','유효는 실제 제시한 경로의 독립 메시 검증이다. 다른 참 경로로 대체하지 않았다. 균형 점수는 유효/R와 정차단/B의 평균이다. 무효 비율은 무효/(유효+무효로 검증된 제안)이며 분모0이면 N/A다. 검증 불가·전체 제안 수와 비용은 CSV에 따로 있다. 미확인과 충돌 실행은 제외하지 않았다.','', '## 두 정책의 최종 차이','']
for condition in ['clean','strong']:
 u=next(q for q in groups if q['seed']=='pooled' and q['condition']==condition and q['label']=='unknown' and q['split']=='all');e=next(q for q in groups if q['seed']=='pooled' and q['condition']==condition and q['label']=='optimized' and q['split']=='all')
 lines.append(f"- {condition}: 최적화entry−unknown 유효{e['valid']-u['valid']:+d}, 정차단{e['correct_blocked']-u['correct_blocked']:+d}, 무효 안내{e['invalid']-u['invalid']:+d}, 차단 오판{e['false_blocked']-u['false_blocked']:+d}, 균형{100*(e['balanced']-u['balanced']):+.2f}%p. 실패 기체는{u['failed_drones']}→{e['failed_drones']}다.")
lines+=['','같은 후보 생성·배정·비행/센서/지도/사람 판정에서 목적함수를 바꾼 비교다. 공통 후보yaw 보완64초 대조는 별도이며 이 표에 섞지 않았다. 일관된 개선인지 조건·시드·튜닝 제외 장면 결과를 모두 확인해야 한다.','', '## 층별 최종 경로','',table(['조건','정책','층','유효/R','정차단/B','무효','미확인'],[[q['condition'],q['label'],q['floor'],f"{q['valid']}/{q['reachable']}",f"{q['correct_blocked']}/{q['blocked']}",q['invalid'],q['unknown']] for q in floor]),'','## 실험과 감사','',f"총{checks['logical_captures']:,}개 논리 촬영, 센서 재생{checks['exact_sensor_replays']}개, 최종 제시 경로{checks['independently_revalidated_final_paths']}개를 독립 재검증했다. 정답 라벨과 유효 경로의 모순{len(checks['reference_contradictions'])}건이다. 고장 기체{checks['persistent_failed_drones']}대를 보존했고 재생성하지 않았다. 후보의 실제 깊이·참 오류·정답 라벨은 선택에 사용하지 않았다.",'','검증 결과로 가중을 다시 선택하지 않았다. 개발128회의 실패/열세 설정과 선택 경위는 각 라운드 보고서와 winner_frozen.json에 남겼다. 본 실험의 시간축은 오프라인 동기 시뮬레이션 시간이며 실시간 구동을 주장하지 않는다. 정확한 native 함수 호출 메모화는 수치를 바꾸지 않으며 CPU 부하가 다른 실행의 속도 향상을 주장하지 않는다.','', '## 재현과 산출물','','- [방법/명령/검증 분리](METHOD.md)\n- [최종 집계](data/validation_summary.csv) · [실행별](data/validation_runs.csv) · [목표·제시 경로별](data/validation_targets.csv)\n- [쌍별 차이](data/validation_paired.csv) · [층별](data/validation_floors.csv)\n- 개발: [라운드1](round01/RESULTS.md), [라운드2](round02/RESULTS.md), [라운드3](round03/RESULTS.md)\n- 원본/동결 소스·설정·시드·원시 입력: results/entry_optimization_20261109\n- 검증 audit/validation_checks.json, 선택 winner_frozen.json, 실행별 replay_manifest.json.','', '센서는 D455급 공개 사양을 참고한 낙관적 가상 깊이 모델과 통제된 저하 조건이다. 실제 카메라/화재 연기/열·가스·구조 안정성을 검증한 결과가 아니다. 정답은 지정한 몸 크기·자세의 기하학적 통과 가능성이다.']
lines+=['','## 한 점 경로와 시간별 변화','',table(['조건','정책','최종 유효','한 점 유효','1m 초과 유효'],[[q['condition'],q['label'],q['valid'],q['single_node_valid'],q['over_1m_valid']] for q in lengths]),'','입구와 겹치는 목표를 사후 제거하지 않고 보조 길이 지표로 공개했다. 잠깐 얻은 경로와 최종 경로를 구분하며 목표별 전이는 validation_transitions.csv에 있다.','', '![시간별 유효 경로·차단·무효·판정률](figures/validation_time_metrics.png)','']
subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_entry_outcomes.py')),'--report',str(report)],check=True)
lines+=['','[목표별 추가 성공·잃은 성공과 비교 기준별 분석](OUTCOME_ANALYSIS.md). 건물 단위의 쌍 재표집 구간은 data/paired_scene_cluster_intervals.csv에 있다.','']
(report/'RESULTS.md').write_text('\n'.join(lines)+'\n');write_json(root/'final_report_manifest.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [report/'RESULTS.md',data/'validation_summary.csv',data/'validation_targets.csv',root/'audit/validation_checks.json']});print('FINAL OPTIMIZATION REPORT COMPLETE',flush=True)
