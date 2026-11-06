#!/usr/bin/env python3
"""Collect the executed decision/path study, without changing archived results."""
import argparse,csv,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/smoke"))
sys.path.insert(0,str(ROOT))
from common import write_json,digest


def rows(path):return list(csv.DictReader(open(path)))


def table(data,keys):
    def fmt(x):
        try:
            n=float(x)
            return str(int(n)) if n.is_integer() else f"{n:.4f}"
        except (ValueError,TypeError):return str(x)
    return "\n".join(["| "+" | ".join(keys)+" |","|"+"---|"*len(keys)]+
                     ["| "+" | ".join(fmt(r.get(k,"")) for k in keys)+" |" for r in data])


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--miluv",default="results/decision_paths_20261103_miluv")
    args=ap.parse_args();src=Path(args.input).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    assert json.load(open(src/"complete.json"))["complete"]
    for name in ("cases.json","cases_geometry.json","references.json","reference_provenance.json","candidates.json",
                 "goals_paths.csv","summary.csv","area_metrics.csv","replay.json","complete.json"):
        shutil.copy2(src/name,out/name)
    for directory in ("selection","reference_paths"):
        shutil.copytree(src/directory,out/directory,dirs_exist_ok=True)
    shutil.copy2(ROOT/"configs/decision_path_validation.json",out/"validation_cases.json")
    shutil.copy2(ROOT/"reports/smoke_20261103/environment.json",out/"environment.json")
    miluv=Path(args.miluv)
    if (miluv/"summary.csv").exists():
        (out/"miluv").mkdir(exist_ok=True)
        for p in miluv.glob("*.json"):shutil.copy2(p,out/"miluv"/p.name)
        shutil.copy2(miluv/"summary.csv",out/"miluv/summary.csv")
    summary=rows(src/"summary.csv");area=rows(src/"area_metrics.csv")
    refs=json.load(open(src/"references.json"));reference_counts={s:sum(r["reference_state"]==s for r in refs) for s in ("reachable","blocked","undetermined")}
    selections=[json.load(open(p)) for p in sorted((src/"selection").glob("*.json"))]
    picks=[]
    for log in selections:
        for policy,cid in log["picks"].items():
            score=next(s for s in log["scores"] if s["id"]==cid)
            picks.append(dict(condition=log["condition"],policy=policy,candidate=cid,robot=score["robot"],
                 move_cost=score["move_cost"],priority=score["priority"],unknown_voxels=score["unknown_voxels"],
                 fallback=log["reason"]["fallback"] if policy=="decision" else False))
    text=f'''# 목표·제시 경로 검증과 한 번의 재관측 — 실행 결과

기준 브랜치 `paper/conference-2026`, 기반 커밋 `dffbb8e`.
기존 `smoke_20261103` 산출물을 재사용한 후속 연구다. 2026-11-03에 시작해 다음 날까지 수행했다.
**결론: 같은 한 번의 관측으로 미관측량 우선보다 유효 경로를 더 많이 확인하면서 잘못 안내한 경로도 줄였다는 결과는 얻지 못했다.**
새 확인 그래프는 모든 주 비교에서 통과 가능 경로를 내지 않았다. 오류율을 0으로 보고하지 않고
검증 완료 경로가 0개인 오류율을 N/A, 판정률 0/22, 미확인 22개로 기록했다.
기존 판정기를 유지하면 재관측이 무효 안내를 취소하는 조건과 오히려 추가하는 조건이 모두 있었다.

## 1. 고정한 사례와 독립 기준

평가 단위는 `corridor915f × south entry × arrival patch`다. 22개 사례 ID, 입구·도착 영역,
층 높이, 공통 변환(identity), 인체 설정은 `cases.json` / `cases_geometry.json`에 있다.
목표는 이전 임무 좌표를 중심으로 한 0.16×0.16 m 도착 구역이다. 좌표·층 스냅은 없다.
도착 발판 높이는 −0.05…0.35 m로 고정했다. 입구는 이전 (−2.35,−19.55) m 셀의 0.08 m 사각형이다.
실제 방 내부·문 상태의 독립 주석은 없으므로 이를 임의로 방·닫힌 문이라고 이름 붙이지 않았다.

기준은 기존 `gt_voxel_fs.npz`와 `2026. 9. 15.floorfix.obj`의 공통 좌표계다.
기준 격자에서 얻은 경로를 **별도 메시 검사**로 확인한 경우만 기준 도달 가능으로 확정했다.
같은 판정기를 GT에 실행했다는 이유만으로 정답으로 삼지 않았다.
기준 발판은 아래 방향 메시 ray, 몸은 메시의 수직·수평 chord와 이동 구간 샘플로 검증하고,
메시와 충돌하지 않아도 기준 복셀의 관측이 부족하면 검증 불가로 보류했다.
기준 상태는 후보 선택·후속 관측 적분 전에 별도 준비 단계에서 고정했다.

기준 사례 수: **도달 가능 {reference_counts['reachable']}, 차단 {reference_counts['blocked']}, 기준 판정 불가 {reference_counts['undetermined']}**.
`references.json`에 제외 사유와 첫 실패 좌표, `reference_paths/`에 기준 witness가 있다.
북쪽 6개 목표는 기존 격자에서 도달 경로가 있었지만 메시 발판을 확인하지 못해 제외했다.
나머지 6개 측면 목표도 미스캔·범위 밖 우회 가능성을 배제하지 못해 차단 정답으로 쓰지 않았다.
실제 복도에서 독립 주석된 유일 문 차단·대체 문 우회·필수 자세 구간의 유형은 부족하다.
이들은 `validation_cases.json`의 별도 분석용 작은 지도에서 검증했으며 복도 실측 성능 표본으로 합치지 않았다.

## 2. 확인 그래프·가능성 그래프와 구간 영향도

새 구현 `covor/entry/decision.py`는 원래 `TravCfg`와 `node_modes`의 발판·몸통·자세·폭 규칙을 공유한다.
인체 폭 0.70/0.45 m, 서기/숙이기/기기 1.90/1.40/0.90 m, 격자 0.10 m,
걸음 높이 0.18 m를 2행으로 양자화한 기존 규약이다. 대각선은 양쪽 모서리의 같은 높이대 노드를 요구한다.
확인 이동에는 원래 관측된 occupied 발판과 free 몸 공간만 쓴다. occupied closing으로 새로 생긴
발판이나 기존 0.5 m 틈 추정은 확인 이동으로 세지 않는다. 기존 정리·틈 판정은 별도 비교군에 보존했다.

가능성 그래프는 관측된 이웃 발판에서 XY 0.5 m, 높이 ±2행 안의 후보만 만든다.
발판 높이가 free로 관측됐거나 몸 공간이 occupied이면 후보로 만들지 않는다.
연속 미확인 이동 길이는 0.5 m 이하이며 관측으로 확인된 발판·몸 공간에 닿을 때만 길이를 초기화한다.
직렬 구간 하나를 열어도 나머지 불확실성은 그대로 남고 최종 통과 가능으로 승격하지 않는다.
확인된 입구가 없어도 가까운 발판에 근거한 가능성은 가설로 탐색할 수 있으나 확인 판정에는 쓰지 않는다.

미확인 노드는 3D 연결 성분으로 묶고, 1.5×1.5×0.3 m를 넘는 성분만 고정 세계 타일로 나눈다.
ID는 최소 복셀 좌표 순서로 결정한다. 작은 문턱의 여러 복셀은 한 구간이 된다.
모든 가능 연결을 유지한 상태와 **그 구간만 닫은 상태**에서 도달 목표 집합을 비교한다.
중요도는 차이가 나는 목표 수다. 다른 경로가 남으면 그 목표는 중요도에 포함되지 않는다.
`selection/*.json`에는 범위·영향 목표·중요도가, 원시 `hypothesis_paths/*.json`에는 열린/닫힌 경로가,
`graphs/*.npz`에는 상태를 재생할 그래프 마스크·구간 ID가 있다.

차단은 단순 무경로가 아니다. 더 관대한 모든 높이·무제한 미확인 column relaxation에서도
범위 밖 탈출이 없고 바닥·천장·내부 관측과 장애물/높이 단절 증거가 있어야 한다.
이 보수적 증명을 못 얻으면 미확인이다. 불완전한 실제 복도에서는 차단을 확정하지 못했다.

## 3. 후보와 관측 배정

이전 24개 보류 프레임 카탈로그를 그대로 사용했다. 모든 방식은 같은 후보·같은 이동 제한 6 m다.
현재 위치는 각 드론의 **마지막 초기 관측 포즈**이며 이동 비용은 후보까지의 유클리드 거리다.
이동 비용은 실제 비행 거리나 충돌 없는 접근 경로가 아니다.
카메라 시야·0.3…5 m 거리·현재 occupied의 정확한 voxel DDA 가림을 적용했다.
발판 상면과 몸 공간 관측 비율을 따로 구하고 둘 중 작은 값을 구간 관측 가능성으로 쓴다.
미관측 셀을 통과한 가시성은 추정으로 표시했다. 구간당 최대 512개 셀을 결정적으로 표본화한 비율이다.
같은 목표가 여러 구간에 있으면 관측 가능성의 최댓값 하나만 합산한다.
점수 동점은 이동 비용→후보 ID 순이며, 양의 점수가 없으면 nearest로 대체한다.

**이는 오프라인 재생 실험이다.** 후보 포즈와 초기 지도는 기존 전체 구간의 배치 최적화 결과다.
후보 시점보다 뒤의 데이터가 포즈 최적화에 사용됐으므로 인과적인 온라인 드론 계획 성능으로 보고하지 않는다.
카탈로그에는 과거 기록 위치도 포함된다. 실제로 그 시점으로 돌아가 비행했다는 의미가 아니다.
초기 프로토타입의 첫 포즈 기준 비용은 배정 전에 마지막 초기 포즈 기준으로 바로잡았고
중간 파일은 원시 폴더의 `prototype_firstpose_cost/`에 남겼다.

{table(picks,['condition','policy','candidate','robot','move_cost','priority','unknown_voxels','fallback'])}

선택 함수에는 현재 labels·후보 추정 포즈·보정값·임무 영역만 들어간다.
참 지도와 실제 후보 깊이는 선택 API에 없으며, 모든 선택 로그를 확정한 뒤 평가 단계에서 읽는다.
기준 사례 큐레이션은 별도 단계이고 그 판정·witness를 선택 함수에 전달하지 않는다.
후보 포즈에 참 포즈는 쓰지 않았다. 참 포즈는 선택 후 실제 스캔 관측을 생성할 때만 사용했다.

## 4. 실제 실행과 재사용

복도 무처리·균일 3강도·비균일 3강도, seed 0, 각 초기 300쌍을 재사용했다.
초기 지도 NPZ와 전체 log-odds `.ot`, 융합 포즈, 304개 기존 렌더/SGBM 캐시가 입력이다.
조건별 연기 깊이와 모든 보류 프레임의 영상은 이전에 전부 저장된 상태가 아니었다.
기존 캐시에 있는 무처리 스테레오·깊이는 재사용했고 없는 선택 프레임만 참 포즈에서 생성했다.
연기 조건의 지도에는 연기 영상에서 다시 계산한 SGBM 깊이만 넣었다.
초기 `.ot`를 방식별로 독립 복제하고, 선택한 스테레오 한 쌍만 원래 OccupancyBuilder로 적분했다.
참 포즈 렌더·추정 포즈 적분 규약, depth-only 가중과 지도 설정을 유지했다.

`observations/`에 선택 프레임 ID·좌우 영상·깊이·포즈·점군·시각이 있다.
`maps/`에 전후 3D 격자, `replay.json`에 초기 입력 순서·선택 후보·비용·스냅샷 경로가 있다.
원시 파일 위치는 `{src}`다. 반복 재관측·동시 배정·학습·발표용 시각화는 수행하지 않았다.

## 5. 목표·경로 지표 — 주 결과

모든 주 비교(7조건×초기/nearest/unknown/decision)는 새 확인 그래프를 동일하게 썼다.
각 실행에서 기준 도달 가능한 목표 분모는 10개, 전체 평가 목표 분모는 22개다.
기준 불가 12개는 유효 경로 확인율의 분모에서 제외하고 별도 집계했다.

| 지표 | 모든 주 비교 결과 |
|---|---|
| 유효 경로 확인율 | 0/10 = 0.0% |
| 전체 제안 경로 | 0 |
| 잘못 안내한 경로 비율 | N/A (0/0, 검증 완료 경로 없음) |
| 판정률 | 0/22 = 0.0% |
| 미확인 | 22/22 |
| 올바른 차단 | 0/0 (기준 차단 사례 없음) |
| 도달 가능한 목표를 차단으로 오판 | 0/10 |
| 미확인→유효 경로 / 미확인→올바른 차단 | 모두 0 |

통과 안내를 보류해서 오류가 적어진 것을 배정의 성공으로 해석하지 않았다.
관측된 발판과 몸 공간의 연속성이 부족해 한 번의 추가 관측으로 목표 경로를 확인하지 못했다.
중요도가 양수였던 조건도 단일 프레임이 필요한 발판·몸 공간 증거를 모두 복구하지는 못했다.
일부 조건은 대체 규칙을 사용했으므로 제안 점수의 성공 사례로 세지 않는다.

기존 정리 지도 판정기를 그대로 쓴 목표·경로 비교(틈 추정 끔):

{table([r for r in summary if r['engine']=='legacy_confirmed'],['condition','policy','n_proposed','n_valid_paths','n_invalid_paths','wrong_path_rate','n_decided','n_unverified','new_invalid_guidance','invalid_guidance_revoked'])}

이는 새 확인 그래프와 섞어 순위를 매기지 않는 별도 비교군이다.
기존 틈 규칙을 켠 결과는 `summary.csv`의 `engine=legacy_gap`에 별도로 있다.
무효 경로는 모든 경우 실제 제시한 순서 있는 경로를 검사한 결과이며, 목적지로 가는
더 좋은 경로를 기준 지도에서 찾아 대체하지 않았다.

### 면적 지표와 결론이 달라진 대표 사례

무처리 초기 지도는 기존 면적 재현율 0.081168(653/8045), false-reach 0.000935였다.
이전 목표 판정에서는 남쪽 3개 목표를 통과 가능으로 제시했으나, 실제 경로의 첫 발판 Z는 0.0 m다.
그 위치의 기준 메시 바닥은 약 0.0501 m이고, 제시 경로의 중심 다리 선분이 바닥을 관통했다.
따라서 3개 모두 strict geometric 검사에서 `mesh_body_collision`이다.
5 cm 정도의 지도 높이 편향도 실제로 제시한 경로 검증에서는 드러난다.
발판 근접 확인 허용오차 0.075 m는 지지면 존재 검사에만 쓰며 몸의 고형물 관통을 허용하지 않는다.
평가 도중 발을 GT 바닥으로 옮기거나 경로를 재계획해 이 실패를 지우지 않았다.
이 엄격한 XYZ 결과는 지도 해상도·표면 양자화에 민감하므로, 기하학적 오차 원인과 함께 읽어야 한다.

약한 비균일 연기에서는 decision의 한 프레임이 기존 확인 판정기의 무효 안내 4개를 취소했다.
반면 약한 균일·강한 균일 조건에서는 무효 안내가 각각 2개·1개 새로 생겼다.
영역이 커지거나 목표 통과 판정 수가 늘었다고 유효 경로 확인이 늘어난 것은 아니었다.

## 6. 기존 면적 지표 — 별도 보존

아래는 변경 전 공개 판정기의 정리·틈·입구 스냅을 유지한 값이다. 새 목표·경로 지표가 아니다.
재현한 초기 7조건의 면적 값은 이전 CSV와 정확히 일치했다.

{table(area,['condition','policy','reach_recall','reach_iou','false_reach_rate','n_map_reach','n_false_reach'])}

분모: recall은 GT 도달 컬럼 8045개, IoU는 지도/GT 도달 컬럼 합집합,
false-reach는 GT에서 어느 자세로도 설 수 없는 컬럼 수다. 목표 단위 오류 비율과 혼용하지 않는다.

## 7. MILUV: 독립 경로 정답 없는 기록 재생

무처리·중간 비균일 연기, 기존 3대×48쌍의 같은 초기 프레임·포즈를 사용했다.
이전 MILUV 결과에는 초기 전체 `.ot`와 프레임별 연기 깊이가 없어서 해당 단계만 재계산하고
기존 NPZ labels와 정확히 일치함을 확인한 뒤 새 출력에 전체 증거를 저장했다.
실제 기록된 IR 프레임만 후보로 사용했고, 두 조건의 선택을 확정한 뒤 후보 영상을 읽었다.
임무 영역은 clean 현재 지도에서 한 번 정한 가상 영역이다. 독립적인 구조대원 진입점·정답이 아니다.
따라서 도달 정확도나 유효 경로율은 보고하지 않고 지도 겹침·관측 비율·판정 변화만 기록했다.

{table(rows(miluv/'summary.csv'),['condition','policy','candidate_id','move_cost','n_passable','n_unverified','n_occupied','n_free','occupied_iou_clean']) if (miluv/'summary.csv').exists() else 'MILUV 결과 파일 없음; 실행 완료로 기재하지 않음.'}

## 8. 검증·파일·재실행

분석용 지도에서 공통 문턱 3목표, 대체 경로, 직렬 문턱, 관측된 벽/빈 발판 제외,
벽 가림, 후보의 미래 깊이/참 지도 미접근, 동일 입력 결정성, 막힌 유일 문,
우회로가 있지만 제시 경로는 벽을 통과하는 사례, 한 프레임 후 잘못된 안내 취소,
폭·자세·잘못된 층·기준 불완전·분모 0 처리를 테스트했다. 로그는 `test_*.log`다.
`audit.json`은 이전 81개 보고 산출물 해시 보존, HEAD 대비 기본 출력 배열 일치,
초기 면적 7조건 일치, 보류 관측 한 번 예산과 선택 입력 분리를 검사한다.

기존 파일 변경은 `covor/entry/traverse.py`에 **기본값 False인 `return_paths` 옵션**으로
predecessor를 저장하는 부분이다. 기본 탐색 순서·판정·면적 출력은 유지한다.
새 파일: `covor/entry/{{decision,assignment,path_eval}}.py`, `scripts/reobserve/`,
`tests/test_decision_paths.py`, `configs/decision_path_validation.json`.

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/src/gs25058/miniconda3/envs/covor/bin/python
O=results/decision_paths_20261103
$PY tests/test_decision_paths.py
$PY tests/test_entry_traverse.py
$PY scripts/reobserve/run.py --out $O --stage prepare
$PY scripts/reobserve/run.py --out $O --stage select --max-move 6
$PY scripts/reobserve/run.py --out $O --stage evaluate
$PY scripts/reobserve/miluv_replay.py --out results/decision_paths_20261103_miluv
$PY scripts/reobserve/audit.py --input $O --out reports/decision_paths_20261103
$PY scripts/reobserve/report.py --input $O --out reports/decision_paths_20261103
```

수치 검증 해상도: 3D 경로 중간점 간격 ≤0.025 m, 몸 chord 간격 0.05 m,
발판 면 근접 허용오차 0.075 m. 메시·복셀의 불완전성과 그보다 미세한 형상 미검출 가능성이 있다.
이는 지정한 인체 모형과 자세의 기하학적 통과 가능성 평가이며 실제 연기·화재·구조 활동의 안전 판정이 아니다.

## 9. 선행연구에서 가져온 관점과 추가한 검증

- [Balaguer 등, RoboCup 지도 평가](https://doi.org/10.1007/s10514-009-9141-z),
  [저자 공개본](https://robotics.ucmerced.edu/sites/g/files/ufvvjh1576/f/page/documents/ar2009.pdf):
  로봇·구조대원의 활용, 탐색·주석 등 다양한 지도 평가 관점을 참고했다.
  이번에는 기존 형상/면적 지표와 실제 목표·제시 경로의 기하학적 성공을 분리했다.
- [Borges 등](https://doi.org/10.1016/j.eswa.2019.06.067),
  [Aerial README](https://github.com/dvbrgs/Aerial),
  [experiments.py의 main_experiment](https://github.com/dvbrgs/Aerial/blob/master/experiments.py):
  도달/비도달 keypoint를 나누고, 찾은 경로를 정답 라벨에서 검사하며 비도달 문제에서
  경로를 잘못 내는 것을 평가하는 관점을 참고했다. 이번에는 3D 발판·자세·몸 부피,
  제시한 경로의 중간 구간, 첫 실패 원인, 검증 불가와 변화 지표를 추가했다.
- [Nikoohemat 등](https://doi.org/10.1016/j.autcon.2020.103109),
  [저자 기관 기록](https://research.utwente.nl/en/publications/indoor-3d-reconstruction-from-point-clouds-for-optimal-routing-in/):
  문·층계·가구 장애물과 실내 연결망을 경로 계획 대상으로 삼는 관점을 참고했다.
  이번에는 스캔의 누락과 실제 막힌 경로의 독립 검증을 구분하고, 없는 문 상태를 만들지 않았다.
- [Geifman·El-Yaniv, Selective Classification](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html):
  보류를 통한 오류 감소와 판정 범위의 교환 관계를 참고했다. 이번에는 보류를 미확인으로
  명시하고 오류율과 판정률·제안 수·검증 불가 수를 함께 보고했다. 신경망 분류나 확률적 위험 보장은 도입하지 않았다.
'''
    (out/"RESULTS.md").write_text(text)
    write_json(out/"artifacts.json",{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))
             for p in sorted(out.rglob("*")) if p.is_file() and p.name!="artifacts.json"})
    print(out/"RESULTS.md")


if __name__=="__main__":main()
