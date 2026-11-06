# 현재 성능 저하의 원인 진단 — 2026-11-04

현재 `corridor915f`, C_3drone anchor-free, seed 0, 초기 3대×100프레임에 대한 진단이다.
기존 실험의 지도·포즈·후보·경로는 보존했다. 운영 알고리즘이나 판정 문턱은 바꾸지 않았다.
원인 분리를 위해 참 포즈/이상 깊이 oracle과 제시 경로의 높이·자세 민감도를 별도 실행했다.
이 결과는 진단 대조이며 기존 성능 결과나 후보 선택의 입력으로 사용하지 않는다.

## 1. 확인된 두 개의 독립 병목

**지도 정합 오차가 기존 면적 성능을 크게 떨어뜨리고, 점유 복셀을 발판으로 직접 쓰는 방식이 새 확인 경로를 끊는다.**
무처리 SGBM+추정 포즈의 기존 면적 recall은 0.081168, IoU는 0.080797이다.
같은 SGBM 관측을 참 포즈에 적분한 대조에서는 recall 0.967060, IoU 0.911434였다.
반면 참 포즈와 이상적 메시 깊이를 모두 써도 새 확인 그래프의 목표 연결은 0/22다.
따라서 포즈 정합만 고쳐도 새 확인 지표가 자동으로 회복된다고 볼 수 없다.

| condition | reach_recall | reach_iou | false_reach_rate | entry_confirmed_nodes | entry_component_columns | confirmed_goals | possible_goals |
|---|---|---|---|---|---|---|---|
| sgbm_true_pose | 0.9671 | 0.9114 | 0.0147 | 0 | 0 | 0 | 15 |
| ideal_true_pose | 0.9741 | 0.9502 | 0.0075 | 1 | 1 | 0 | 16 |
| ideal_estimated_pose | 0.1500 | 0.1485 | 0.0022 | 0 | 0 | 0 | 7 |

표의 recall/IoU/false-reach는 기존 cleanup·gap 규약의 **면적 지표**, confirmed_goals는 raw 관측 증거를
요구하는 새 그래프의 목표 연결 수다. oracle은 같은 300프레임·격자·가중식을 사용했다.
참 포즈 대조는 모든 드론의 자세와 동료 마스킹 좌표도 참 좌표로 일관되게 바꾼 상한 대조다.
오도달률도 달라지므로 이 값을 실제 시스템의 달성 성능으로 채택하지 않는다.

## 2. 후단이 드론 사이의 상대 고도를 크게 바꾼다

| robot | initial_rmse | initial_mean_dz | raw_rmse | mean_dz | individual_aligned_rmse | common_aligned_rmse |
|---|---|---|---|---|---|---|
| ifo001 | 0.1230 | -0.0001 | 0.2758 | 0.0260 | 0.1212 | 0.6293 |
| ifo002 | 0.1028 | 0.0001 | 0.6299 | 0.5097 | 0.0933 | 0.1742 |
| ifo003 | 0.0733 | 0.0002 | 1.4616 | 1.3890 | 0.0850 | 0.7532 |

최적화 전 카메라 초기값의 평균 Z 오차는 약 0이었지만, 최적화 후 ifo002는 +0.510 m,
ifo003은 +1.389 m로 변했다. ifo001은 +0.026 m다. 초기값은 기존 전 궤적 GT 정렬을 사용한 것이므로
초기값 성능을 실제 배치 가능한 VIO 기준선으로 해석하지 않는다. 여기서는 최적화가 상대 높이를
얼마나 이동했는지만 분리해 확인했다.

드론별로 다시 정렬한 ATE 0.121/0.093/0.085 m는 이 상수 정합 오차를 상당 부분 흡수한다.
공통 yaw+이동 변환 하나로 정렬해도 각 RMSE는 0.629/0.174/0.753 m다.
따라서 지도 전체의 공통 원점 이동만으로 해결되는 오차가 아니다.

왜 그런 해가 나오는가에 대해서는 **긴 복도의 약한 수직 거리 기하와 높이 제약 부재**가 유력하다.
실제 GT 몸 중심 거리의 수직 민감도 |Δz|/거리 중앙값은 약 0.0092–0.0160이다.
드론 간 거리는 약 8.4–24.9 m인 데 비해 고도 차이는 작고, 중력 prior는 자세의 기울기를 관측하며
Z 위치를 직접 고정하지 않는다. 현재 조건은 앵커·높이 팩터도 없다.
이는 측정한 기하와 설정에서의 해석이며 전체 그래프의 관측성 rank 검정이나 Jacobian 결함 검정을
대체하지는 않는다. 포즈 평균 Z 이동 자체와 oracle 지도 차이는 직접 확인한 사실이다.

## 3. 완벽한 센서에서도 free 증거가 바닥 표면을 지운다

추적한 세계 복셀은 [-25, -196, 0], 중심 [-2.45, -19.55, 0.05] m다.
메시 바닥은 Z=0.050065 m에 있고 GT는 occupied지만,
**참 포즈+이상 깊이로 만든 지도도 이 셀을 free로 분류**했다.

- 점유 종단 광선 31개: log-odds 합 +19.8792
- free 통과 광선 69개: log-odds 합 -21.1771
- 프레임 순서대로 재생한 최종 log-odds -1.2979 → free

원인은 같은 0.10 m 복셀에 얇은 바닥 표면과 그 위의 공기가 함께 들어가기 때문이다.
다른 바닥 지점으로 향하는 광선이 이 복셀의 공기 부분을 지나면서 free 증거를 더한다.
현재 점유지도는 부피 점유를 위한 합산 구조이고, 새 판정기는 그 셀이 occupied일 때만 발판으로 인정한다.
즉, **바닥을 보았다는 증거가 있는데도 발판 판정에는 남지 않는 표현의 불일치**가 있다.
새 가능성 그래프는 free 발판을 후보로 넣지 않으므로 이렇게 지워진 바닥에서 확정·가능 경로가 함께 끊긴다.
이는 SGBM 오차만으로 설명되지 않는다.

무처리 지도에서 지정 입구의 확인 연결 성분은 16셀이다. 중간 균일·중간 비균일·강한 비균일 조건은
입구의 확인 노드 자체가 0개다. 다음 표는 raw 지도와 cleanup을 추가한 진단을 함께 보인다.
cleanup으로 늘어난 지지를 관측으로 확인된 발판으로 채택하지는 않았다.

| condition | entry_confirmed_nodes | entry_component_columns | goals_with_local_confirmed_nodes | confirmed_goals | possible_goals |
|---|---|---|---|---|---|
| clean | 1 | 16 | 4 | 0 | 4 |
| clean_cleanup_diagnostic | 1 | 52 | 5 | 1 | 5 |
| uniform_weak | 1 | 9 | 4 | 0 | 4 |
| uniform_weak_cleanup_diagnostic | 1 | 9 | 5 | 0 | 5 |
| uniform_medium | 0 | 0 | 4 | 0 | 0 |
| uniform_medium_cleanup_diagnostic | 0 | 0 | 4 | 0 | 0 |
| uniform_strong | 1 | 1 | 5 | 0 | 6 |
| uniform_strong_cleanup_diagnostic | 1 | 1 | 5 | 0 | 6 |
| heterogeneous_weak | 1 | 13 | 4 | 0 | 5 |
| heterogeneous_weak_cleanup_diagnostic | 1 | 13 | 4 | 0 | 5 |
| heterogeneous_medium | 0 | 0 | 4 | 0 | 0 |
| heterogeneous_medium_cleanup_diagnostic | 0 | 0 | 4 | 0 | 0 |
| heterogeneous_strong | 0 | 0 | 4 | 0 | 0 |
| heterogeneous_strong_cleanup_diagnostic | 0 | 0 | 4 | 0 | 0 |

## 4. 재관측이 필요한 증거를 거의 추가하지 못한다

양의 중요도가 있는 실제 구간은 조건별 0–2개뿐이고, 7조건 중 4조건은 nearest 대체를 사용했다.
선택된 중요 구간의 몸 공간 가시성은 약 94–98%지만 주요 발판 가시성은 3–18%다.
주요 발판들은 시야각 안에는 들어왔다. 낮은 비율의 상당 부분은 관측된 셀에 의한 가림이다.
이 발판 높이 가설들은 약 −0.20…0.00 m로, 입구의 실제 바닥 약 +0.05 m보다 아래에 있다.
현재 지도의 바닥 높이 오차와 이웃 ±2행 추론이 아래쪽 발판 가설을 만들고,
이 가설을 전부 같은 관측 대상으로 평균 내는 것도 점수의 물리적 의미를 약하게 한다.

| condition | segment | n_foot_nodes | n_support_in_frustum | expected_support_visibility | expected_body_visibility | actual_foot_endpoint_hits | foot_unknown_before | foot_unknown_after | newly_confirmed_nodes |
|---|---|---|---|---|---|---|---|---|---|
| heterogeneous_weak | 662 | 102 | 102 | 0.1078 | 0.9746 | 7 | 97 | 88 | 9 |
| heterogeneous_weak | 997 | 119 | 1 | 0.0084 | 0.3359 | 0 | 117 | 117 | 0 |
| uniform_strong | 638 | 101 | 101 | 0.0297 | 0.9355 | 4 | 100 | 97 | 2 |
| uniform_weak | 619 | 45 | 45 | 0.1778 | 0.9805 | 2 | 41 | 41 | 3 |

예를 들어 약한 균일 조건은 미확인 발판 41개가 관측 뒤에도 41개였고,
약한 비균일 조건의 주요 구간도 97→88개로 일부만 줄었다. 단일 프레임은 국소 증거를 개선했지만
입구부터 목표까지의 모든 연결을 확인하지는 못했다. 중요도는 조건부 연결 의존도이지
그 프레임을 추가하면 유효 경로가 완성될 확률이 아니다.
현재 후보 점수에는 실제 스테레오 성공률이나 갱신 후 log-odds 문턱 통과까지의 예측도 없다.

## 5. 경로 검증의 100% 무효는 대부분 같은 출발점 실패다

기존 판정기의 제안 204건은 각각 독립적인 204개 장애물 통과 실패가 아니다.
132건은 첫 경로점의 몸 충돌, 72건은 입구/층 조건 실패로 끝났으며 여러 목표가 같은 시작 자세를 공유했다.
무처리 대표 경로는 발 Z=0.0 m를 제시하지만 실제 바닥은 약 0.0501 m다.
높이만 +0.05 m 옮기면 이번에는 약 Z=1.95 m의 입구 상부 구조물에 서 있는 몸이 충돌했다.
같은 XY 경로에 +0.05 m와 숙이기·옆으로 모드를 적용한 **진단**에서는 첫 목표 경로가 유효해졌다.

| shift_z | posture_at_least | status | reason |
|---|---|---|---|
| 0 | 0 | invalid | mesh_body_collision |
| 0 | 1 | invalid | mesh_body_collision |
| 0 | 2 | invalid | mesh_body_collision |
| 0.0500 | 0 | invalid | mesh_body_collision |
| 0.0500 | 1 | valid | mesh_support_and_swept_body_verified |
| 0.0500 | 2 | valid | mesh_support_and_swept_body_verified |
| 0.1000 | 0 | invalid | mesh_body_collision |
| 0.1000 | 1 | valid | mesh_support_and_swept_body_verified |
| 0.1000 | 2 | valid | mesh_support_and_swept_body_verified |

이는 바닥 높이와 자세 선택이 함께 틀린 사례다. 단순히 허용오차를 키우거나 좌표를 일괄 이동하는
것으로 모든 경로를 고칠 수 없다. 높이/자세 변경은 사후 민감도 검사이며 원래 경로를 대체하거나
기존 성공률에 넣지 않았다. 현재 실험에서는 이 공통 입구 실패가 다른 후보 정책의 효과를 가린다.

## 6. 평가 표본의 한계는 별도다

22개 목표 중 독립 메시 검증으로 도달 가능한 것은 10개, 기준 판정 불가는 12개였고
확정된 차단 목표는 0개다. 따라서 잘못된 경로를 취소하는 능력이나 차단 판정 정확도까지
일반화할 표본이 부족하다. 이 문제는 지도의 물리적 실패 원인과 구분해야 한다.

## 7. 수정 우선순위

1. **후단의 드론 간 상대 높이부터 확인한다.** 로봇별 정렬 ATE와 함께 공통 프레임 오차·평균 Z 오차를
   회귀 지표로 삼고, 초기/최종 상대 높이·거리 Jacobian·게이지·높이 관련 제약을 검사한다.
   관측 가능한 시작/지면 정보를 어떤 제약으로 쓸지는 실제 운용 조건에 맞춰 별도 실험해야 한다.
2. **발판/바닥 증거를 부피 점유와 분리한다.** 표면 높이·지지 증거와 불확실성을 유지하는 표현을 검토한다.
   free 바닥을 임의로 occupied로 바꾸거나 미확인을 통과 가능으로 완화하는 처방은 진단에서 검증하지 않았다.
3. **입구의 실제 바닥·상부 여유와 제시 자세를 일관되게 만든다.** 발판 높이, 접촉 허용오차,
   몸 충돌의 의미를 정의하고 양자화된 XYZ가 현장 표면과 어느 정도 일치해야 하는지 검증한다.
4. **그다음 재관측 정책을 비교한다.** 입구와 일부 목표의 유효 경로가 존재하고 한 국소 구간의
   미확인만 남은 고정 사례에서, 한 프레임으로 실제 발판 증거가 늘어나는 후보를 포함한다.
   경로 의존도 점수와 예상 관측 후 확정 이득은 다른 양이므로 각각 검증한다.

## 재실행과 보존

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/src/gs25058/miniconda3/envs/covor/bin/python
O=results/diagnosis_20261104
$PY scripts/reobserve/diagnose.py --out $O --stage graphs
$PY scripts/reobserve/diagnose.py --out $O --stage poses
$PY scripts/reobserve/diagnose.py --out $O --stage oracle
$PY scripts/reobserve/diagnose.py --out $O --stage paths
$PY scripts/reobserve/diagnose.py --out $O --stage candidates
$PY scripts/reobserve/diagnose.py --out $O --stage floor_trace
$PY scripts/reobserve/diagnose.py --out $O --stage report
```

oracle 3D 격자는 `/src/gs25058/cr_RNE/covor_slam/results/diagnosis_20261104`에, 진단 CSV·JSON은 이 보고서 옆에 있다.
운영 파라미터·후보 선택·기존 성능 CSV는 이번 진단으로 변경하지 않았다.
새 코드는 `scripts/reobserve/diagnose.py`다.
