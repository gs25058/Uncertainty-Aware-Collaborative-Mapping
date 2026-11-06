# 짧은 경로 생성/평가 분리와 한 번 재관측의 사후 상한 검사

## 고정 사례와 평가기 양성 대조

이전에 메시 검증으로 도달 가능한 10개 목표 중 가까운 `south_0`를 실행 전에 고정했다.
입구 (-2.35, -19.55), 목표 중심 (-2.35, -19.25), 직선 거리 0.30 m다.
기존 입구/목표 영역·층 허용 범위·인체 크기·평가 접촉 허용오차를 그대로 사용했다.

양성 대조는 경로 그래프에서 꺼낸 경로를 재활용하지 않았다. 고정한 네 XY 위치에서 메시
삼각형의 무게중심 좌표로 발판 높이를 별도 계산했고, 숙이기 1.40 m·옆으로 폭 0.45 m를 고정했다.
연속 몸통 원판을 포함하는 보수적인 swept AABB와 메시 삼각형의 정확한 클리핑으로 몸통 공간도
독립 확인했다. 지면은 Z=0.05006477 m, 발은 5 mm 위 Z=0.05506477 m였다.
평가기 결과는 **valid**다. 이 경로는 `positive_control.json`에만 저장했고
지도 생성이나 시스템 탐색의 입력으로 넣지 않았다. 평가기의 좌표·접촉·충돌 문턱을 변경하지 않았다.
평가기는 기존 메시 발판·자세별 몸 공간·경로 사이 swept-body 검사를 그대로 수행한다.

## 1. 같은 관측에서 시스템이 스스로 생성한 경로

같은 초기 3대×100프레임, seed 0, 무처리 이상 깊이를 참 포즈에서 생성했다. 같은 깊이를
참 포즈와 기존 추정 포즈에 각각 적분했다. 기존 이상 깊이 대조 지도 두 개와 라벨 배열이
정확히 일치함도 확인했다. 표면층은 관측 평면으로만 생성했고 메시 발판 주석을 넣지 않았다.
탐색 입력은 현재 지도·표면층·고정 입구/목표뿐이다.

| arm | verdict | path_status | n_path_points | first_failure |
|---|---|---|---|---|
| true | passable | valid | 6 | none |
| estimated | passable | invalid | 5 | foot_height |

참 포즈에서 시스템이 스스로 만든 경로는 유효했다. 추정 포즈에서는 경로를 만들었으나 평가에서
거절됐다. **경로 생성 실패가 아니라 잘못된 높이의 경로 제안**이라는 차이가 중요하다.
첫 끊김은 `foot_height`다. 입구의 지지 표는 3개,
순 증거 1.602691로 지지 확인 조건을 만족했지만, 표면 높이
-0.054987 m, 발 높이 -0.049987 m였다.
메시 지면 0.050065 m보다 발이 약 10 cm 아래에서 시작해 몸이 바닥과 충돌했다.

`stage1_trace.csv`/`trace_*.json`은 발판 증거 → 발 높이 → 가능한 자세 → 몸 공간 → 다음 연결을
차례로 기록한다. `stage1_body_modes.csv`는 3자세×2폭의 free/occupied/unknown 셀 수와
처음 막힌 복셀, 독립 메시 몸 검증을 제공한다. 성공 경로는 `generated_true.json`,
거절 경로와 평가기의 첫 실패 위치는 `generated_estimated.json`에 있다.

잘못된 입구 발판에 기여한 초기 관측:

| robot | index | weight | surface_z | camera_dz | rotation_error_deg |
|---|---|---|---|---|---|
| ifo001 | 336 | 0.8055 | -0.0454 | -0.0676 | 1.4396 |
| ifo001 | 492 | 0.4963 | -0.0581 | -0.0601 | 0.6248 |
| ifo001 | 504 | 0.3009 | -0.0755 | -0.0529 | 1.8419 |

이 관측은 모두 첫 드론 ifo001에서 왔다. 따라서 다음 우선순위는 포즈/공통 좌표계의 수직
일관성이다. 다만 다른 두 드론에 상수 Z 보정만 넣는 것으로 충분하다고 볼 수 없다.
기준 드론 자신의 국소 고도 및 자세 오차도 함께 확인해야 한다. 본 실행에서 새로운 높이 보정을
적용하거나 참 높이를 추정 지도에 주입하지 않았다.

## 2. 한 번 관측으로 바뀔 수 있는 단일 문턱 사례

참 포즈 경로의 성립을 확인한 뒤, 사전에 고정한 Y=[-19.5,-19.4) m, Z=[-0.05,0.15] m
바닥 띠에서 초기 깊이 끝점만 통제해 보류했다. 이 띠를 실제로 관측한 첫 초기 프레임
**ifo001 index 336**는 그대로 남겼다. 나머지 초기 영상에서는 해당 끝점 픽셀을
invalid 처리하고, 몸 공간을 포함한 다른 광선은 기존 파이프라인에 정상 적분했다.
이는 자연스럽게 발생한 연기 관측 사례가 아닌 **깊이 관측 누락을 통제한 사례**다.
GT 라벨로 확인 영역을 채우거나 관측 표 수를 수동 변경한 사례가 아니다.

초기 상태는 `unverified`, 입구는 확인됨, 가능한 경로의 미확인 국소 구간은 1개,
문턱 밖 미확인 경로 노드는 0개다. 대표 발판의 관측 표는 1개, 증거는 0.836442다.
프레임별 가림 픽셀 수와 순서는 `threshold_input_order.csv`, 검증은 `threshold_case.json`에 있다.
최소 2프레임·순 증거 1.2·기존 몸/걸음 조건은 바꾸지 않았다.

### 후보 선택을 먼저 확정

기존 24개 드론/프레임 ID를 유지했다. 이 진단은 완벽한 위치추정 조건이므로 후보와 현재 위치에도
시뮬레이터 참 포즈를 사용했다. 이는 **오프라인 perfect-localization 대조**이며 현장 온라인 선택
성능으로 해석하지 않는다. 이동 비용은 각 드론의 마지막 초기 위치에서 후보까지의 직선 거리이고
공통 제한은 6 m다. 24개 중 제한 안 후보는 13개다.

현재 지도와 카메라 시야만으로 `selection.json`을 먼저 저장했다. 후보 깊이·사후 결과·참 형상은
선택 함수의 입력이 아니다. 이후 24개 후보의 이상 깊이를 각각 생성해 **같은 초기 전체 log-odds
지도와 표면층의 독립 복제본**에 한 번씩 적분했다. 후보 결과를 점수 조정이나 재선택에 쓰지 않았다.
선택 파일의 해시가 사후 검사 뒤에도 바뀌지 않음을 확인했다.

### 사후 상한과 정책 결과

성공 후보는 전체 24개 중 **2개: [2, 3]**이고,
이동 제한 안에서도 **2/13개: [2, 3]**다.
이들은 같은 목표 하나를 각각 확인한 대체 관측이지 서로 다른 목표가 아니다.
따라서 이 사례에서 한 번의 관측 예산 자체가 부족했던 것은 아니다.

| policy | selected_id | move_cost | after | path_status | path_length_m | newly_known_voxels | diagnosis |
|---|---|---|---|---|---|---|---|
| nearest | 23 | 0.2029 | unverified | not_proposed | None | 3 | selection_missed_feasible_one_view |
| unknown | 4 | 4.5411 | unverified | not_proposed | None | 71 | selection_missed_feasible_one_view |
| decision_impact | 3 | 3.7762 | passable | valid | 1.4414 | 90 | succeeded |
| decision_readiness | 2 | 3.9658 | passable | valid | 1.5828 | 87 | succeeded |

모든 방식의 추가 관측 수는 1이다. 중요 구간 우선(`decision_impact`)과 단일 프레임 증거 대조
(`decision_readiness`)는 성공 후보를 골랐다. 최근접과 미관측 공간 우선은 미확인에 머물렀다.
이 사례에서는 **제안 정책이 성공 후보를 놓친 문제가 관찰되지 않았다.**
원래 단순 중요도 점수는 후보 2와 3이 동점이어서 이동 비용이 낮은 3을 골랐고,
증거 대조는 2를 골랐다. 사후 성공 여부는 이 선택에 쓰지 않았다.
미관측 공간 점수는 현재 지도에서 예상한 가시 셀 수이며 실제 깊이 반환을 보장하는 정보 이득은
아니다. 후보 4의 예상 미관측 셀은 5184개였지만 목표 구간의 예상 관측 점수는 0이었다.
따라서 이 한 사례를 광범위한 지도 완성도 비교나 정책의 일반적 우위로 확대하지 않는다.

성공 경로가 실제로 건넌 문턱 발판:

| candidate_id | world_ijk | before_votes | before_evidence | after_votes | after_evidence |
|---|---|---|---|---|---|
| 2 | [-20, -195, 0] | 1.0 | 0.7680 | 2.0 | 1.6198 |
| 3 | [-18, -195, 0] | 1.0 | 0.7456 | 2.0 | 1.2175 |

대표 복셀 (-24,-195,0) 하나의 증거만 보면 성공을 놓친다. 후보 2/3은 같은 문턱 구간의 다른
발판을 두 번째로 관측해, 각각 약 1.58/1.44 m의 유효한 우회 경로를 제시했다. 목표와 입구는
동일하며 평가 과정에서 기준 경로로 대체하지 않았다. `candidate_paths/`에 실제 제시 경로의
XYZ·자세·폭, 검증과 첫 실패 위치를 저장했다.

## 범위와 다음 수정 방향

이 결과는 한 목표·한 시드·무처리 이상 깊이·참 포즈의 통제 실험이다. 앞서 SGBM/추정 포즈에서
0/10이었던 결과를 대체하지 않으며, 실제 연기·화재에서의 성능이나 여러 목표의 평균 우위를
입증하지 않는다. 참 포즈와 이상 깊이는 원인 분리용 진단 상한이다.

이번 목표의 기준 상태는 도달 가능이므로 올바른 차단 확인은 0개이며 차단 판단 능력은 검증하지
않았다. 성공 정책의 무효 경로는 0/1, 경로를 제시하지 않은 정책의 잘못 안내 비율은 N/A다.
따라서 이번 사례로 잘못된 안내 감소를 주장하지 않는다.

현재 우선순위는 **추정 포즈의 높이/자세와 공통 좌표계의 일관성을 개선하고 같은 짧은 경로를
다시 검증하는 것**이다. 평가기가 양성 경로를 거절하거나, 완벽한 센서/포즈에서도 표면층과
몸 검사 사이가 항상 끊기는 상황은 아니었다. 이번에는 평가 문턱이나 재관측 점수를 튜닝하지 않았다.

## 재실행·산출물

수정: `covor/entry/decision.py`에 단일 목표를 찾으면 멈추는 선택적 Dijkstra 인자를 추가했다.
기본 전체 도달 탐색은 그대로이며, 목표 경로가 완전 탐색과 같음을 테스트했다.
추가: `scripts/reobserve/short_path_probe.py`, `tests/test_short_path_probe.py`.
이외 지도/평가기 기본 규칙은 변경하지 않았다. 기존 보고서 산출물 191개의 해시를 확인했다.
새 단위 검증과 기존 회귀 테스트는 총 38건 통과했다.

`candidate_outcomes.csv`: 24개 전체 결과. `policy_comparison.csv`/`policy_summary.csv`: 정책별 비교.
`selection.json`: 선택 전 점수·후보·입력 해시. `successful_threshold_crossings.json`: 실제 건넌 발판.
`candidate_maps/`, `candidate_observations/`, `candidate_paths/`: 모든 후보의 재생 자료.
큰 입력/지도/경로는 `/src/gs25058/cr_RNE/covor_slam/results/short_path_probe_20261104`에 있고, 설정·상세 CSV·로그는 보고서 옆에도 복사했다.

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/src/gs25058/miniconda3/envs/covor/bin/python
O=results/short_path_probe_20261104_replay
for S in prepare positive ideal diagnose threshold certify select all-candidates contributors report; do
  $PY scripts/reobserve/short_path_probe.py --out $O --stage $S
done
$PY tests/test_short_path_probe.py
$PY tests/test_decision_paths.py
$PY tests/test_support_surfaces.py
$PY tests/test_entry_traverse.py
```
