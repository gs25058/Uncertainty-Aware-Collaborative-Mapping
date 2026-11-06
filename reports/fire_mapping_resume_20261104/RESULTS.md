# 사람용 3D 지도 목적에 맞춘 중단 실험 재검토·재개 결과

**적합하다고 판단한 실험을 끝까지 실행했다. 그러나 이번 설정에서 유효한 입구→목표 경로는
모든 방식이 0/10개였고, 주 방법으로 새 구성을 채택할 근거는 확보하지 못했다.**
원래 포즈의 표면 높이·몸 검사 조건은 두 조건 모두 국소 유효 구성 1/86개를 확인했지만,
제시한 국소 구성 6개 중 5개는 무효였다. 이는 경로 성공이 아니다.
상대 높이 등록은 면적 도달을 크게 늘리는 대신 오도달도 늘렸다.
포즈+깊이 가중도 이 장면에서는 무가중보다 3D IoU와 완성도가 낮고 exact false-free가 높았다.

## 판단과 실제 실행

목표는 모의 화재 센서 조건에서 여러 드론으로 **사람이 활용할 수 있는 3D 지도**를 만드는 것이다.
가중치나 특정 재관측 점수의 우수성을 전제하지 않았다. 동일한 `corridor915f`, seed 0,
3대×100 초기 스테레오 쌍, 고정 입구·22개 목표를 사용했다.

| 중단/미완료 실험 | 판단 | 이번 실행 |
|---|---|---|
| 부피 점유와 발판 표면 증거 분리 | 사람용 지도에 직접 관련 | 원래 포즈에서 기존 판정과 비교 완료 |
| 관측 발 높이·몸 폭·머리 공간 검증 | 사람의 기하학적 이동 판단에 직접 관련 | 높이 등록 없이 분리 검증 완료 |
| 겹치는 바닥을 이용한 상대 Z 등록 | 공통 바닥 가정이 있는 선택적 정합 대조 | 별도 조건으로만 완료 |
| 한 번의 추가 관측 배정 | 지도 개선 수단으로 관련 | 원래 포즈/등록 포즈 각각 동일 후보·예산으로 완료 |
| 무가중/깊이/포즈+깊이 가중 | 효과가 있을 때 채택할 구성 요소 | 같은 300쌍 입력으로 별도 대조 완료 |

무처리와 **약한 비균일 모의 연기 beta=0.15**를 실행했다. 나머지 연기 강도·MILUV·VIO 재실행,
추가 장면·시드·동시 배정·신경망은 이번 재개 범위에 넣지 않았다.
이번 실험은 같은 진단 장면에서 한 탐색적 비교이며 독립 장면의 일반화 검증은 아니다.

## 재개 전에 바로잡은 입력 오류

중단된 `repair.py`는 이미 정렬된 연기 영상을 `StereoDepth.depth`로 다시 정렬하고 있었다.
기존 정식 `run_corridor.py`/`run_miluv.py`의 `_depth_rectified` 처리에는 이 오류가 없다.
드론별 첫 프레임 감사에서 유효 마스크가 1109/533/9788픽셀 달랐다.
해당 중단 실험의 연기 깊이·표면층·등록·지도·선택은 재사용하지 않고 새로 만들었다.
깨끗한 입력과 검증된 깨끗한 표면/등록 결과는 해시와 함께 재사용했다.
수정 캐시는 입력·설정 해시와 `rectified_sgbm_v1` 표시를 요구한다. 회귀 테스트도 추가했다.
새 depth-only 재생의 두 조건 지도는 **기존 정식 지도와 분류 배열이 정확히 일치**했다.
따라서 기존 결과는 보존되며, 중단된 연기 높이 보정 거절 결과는 유효한 결론으로 사용하지 않는다.

## 1. 지도 표현과 목표 도달 — 분리된 비교

`raw`: 기존 점유 복셀 발판. `support`: 독립 표면 증거 + 기존 몸/격자 높이 판정.
`surface_geometry`: 원래 포즈 + 표면 높이/정확한 몸 원판-복셀 모서리 검사.
`registered`: 상대 Z 등록 + 표면 증거 + 기존 몸 판정.
`geometry`: 상대 Z 등록 + 표면 높이/몸 검사. 모든 조건의 진입점·목표·사람 크기는 같다.

| condition | variant | support_cells | entry_confirmed_nodes | entry_component_columns | valid_path_confirmed | n_proposed | n_invalid_paths | reach_recall | reach_iou | false_reach_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | raw | 14525 | 1 | 16 | 0 | 0 | 0 | 0.0812 | 0.0808 | 0.0009 |
| clean | support | 18811 | 0 | 0 | 0 | 0 | 0 | 0.0812 | 0.0808 | 0.0009 |
| clean | surface_geometry | 18811 | 0 | 0 | 0 | 0 | 0 | 0.0812 | 0.0808 | 0.0009 |
| clean | registered | 19029 | 0 | 0 | 0 | 0 | 0 | 0.8077 | 0.6759 | 0.0411 |
| clean | geometry | 19029 | 0 | 0 | 0 | 0 | 0 | 0.8077 | 0.6759 | 0.0411 |
| heterogeneous_weak | raw | 15159 | 1 | 13 | 0 | 0 | 0 | 0.0799 | 0.0795 | 0.0010 |
| heterogeneous_weak | support | 18645 | 0 | 0 | 0 | 0 | 0 | 0.0799 | 0.0795 | 0.0010 |
| heterogeneous_weak | surface_geometry | 18645 | 0 | 0 | 0 | 0 | 0 | 0.0799 | 0.0795 | 0.0010 |
| heterogeneous_weak | registered | 18927 | 0 | 0 | 0 | 0 | 0 | 0.8073 | 0.6920 | 0.0353 |
| heterogeneous_weak | geometry | 18927 | 0 | 0 | 0 | 0 | 0 | 0.8073 | 0.6920 | 0.0353 |

목표 유효 경로 확인율의 분모는 기준 메시에서 독립적으로 도달 경로를 확인한 10개다.
22개 중 12개는 기준 판정 불가, 기준 차단은 0개다. 차단 정확도는 이 표본에서 검증할 수 없다.
기존 면적 재현율의 분모는 기준 도달 열 8045개, 면적 false-reach의 분모는 기준에서 국소 몸이
들어가지 못하는 열 39582개다. 기존 cleanup/gap 규약의 면적 지표를 보존했다.
이 면적 증가를 유효한 제시 경로 증가로 해석하지 않는다.

## 2. 사람용 지도의 국소 유효성 — 경로 지표와 구분

고정된 메시 검증 지지점 15148개에 대해 같은 XY 열과 높이 오차
7.5 cm 이내에서 지지 회복률을 평가했다. 기준 복셀 후보 17780개 중
메시에서 지지를 검증하지 못한 2632개는 분모에서 제외했다.
예측 지지점 자체도 기준 메시의 수직 광선으로 유효/무효/검증 불가를 구분했다.
표면 precision의 분모는 valid+invalid이며 검증 불가 수를 함께 제공한다.

방법별 결과를 읽기 전에 같은 기준 지지점에서 128개 국소 검사 위치를 고정했다.
각 위치의 기준 몸 공간은 메시에서 3자세×2폭을 별도 검사했다. 평가 대상은 각 방법이
실제로 선언한 발 높이·자세·폭이며, 기준의 더 좋은 자세로 바꿔 검증하지 않는다.
XY는 같은 열이며 층 대응 높이 범위 ±0.15 m를 고정했다. 전역 목표 스냅은 바꾸지 않았다.
**이 표는 국소적으로 사람이 설 수 있는 구성의 유효성이고, 입구에서 도달한 경로 수가 아니다.**

| condition | variant | n_surface_claims | surface_valid | surface_invalid | surface_unverifiable | surface_recall | n_reference_local_feasible | n_local_proposed | n_valid_local | n_invalid_local | n_unverifiable_local | local_confirmation_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | raw | 14525 | 2629 | 6383 | 5513 | 0.1371 | 86 | 4 | 0 | 4 | 0 | 0.0 |
| clean | support | 18811 | 3098 | 9525 | 6188 | 0.1487 | 86 | 7 | 1 | 4 | 2 | 0.0116 |
| clean | surface_geometry | 18811 | 3098 | 9525 | 6188 | 0.1487 | 86 | 6 | 1 | 5 | 0 | 0.0116 |
| clean | registered | 19029 | 5542 | 2309 | 11178 | 0.2746 | 86 | 46 | 6 | 38 | 2 | 0.0698 |
| clean | geometry | 19029 | 5542 | 2309 | 11178 | 0.2746 | 86 | 43 | 1 | 42 | 0 | 0.0116 |
| heterogeneous_weak | raw | 15159 | 2644 | 6509 | 6006 | 0.1360 | 86 | 5 | 0 | 5 | 0 | 0.0 |
| heterogeneous_weak | support | 18645 | 2876 | 9151 | 6618 | 0.1393 | 86 | 8 | 1 | 7 | 0 | 0.0116 |
| heterogeneous_weak | surface_geometry | 18645 | 2876 | 9151 | 6618 | 0.1393 | 86 | 6 | 1 | 5 | 0 | 0.0116 |
| heterogeneous_weak | registered | 18927 | 5216 | 2133 | 11578 | 0.2584 | 86 | 45 | 4 | 41 | 0 | 0.0465 |
| heterogeneous_weak | geometry | 18927 | 5216 | 2133 | 11578 | 0.2584 | 86 | 41 | 2 | 39 | 0 | 0.0233 |

같은 검사 위치의 변화:

| condition | before | after | gained_valid | lost_valid | invalid_to_valid | invalid_to_no_proposal | added_invalid |
|---|---|---|---|---|---|---|---|
| clean | raw | support | 1 | 0 | 0 | 1 | 1 |
| clean | support | surface_geometry | 1 | 1 | 1 | 1 | 3 |
| clean | surface_geometry | geometry | 1 | 1 | 0 | 0 | 37 |
| heterogeneous_weak | raw | support | 1 | 0 | 0 | 1 | 3 |
| heterogeneous_weak | support | surface_geometry | 1 | 1 | 1 | 2 | 1 |
| heterogeneous_weak | surface_geometry | geometry | 2 | 1 | 0 | 0 | 34 |

첫 대표 사례는 clean의 probe 43 (added_invalid)이다. raw의 null / observed_support_or_body_missing에서 support의 {"xyz": [-1.9500000000000002, -17.55, 0.0], "posture": 0, "width": 0, "observed_support": true} / mesh_body_collision로 바뀌었다. 상세 좌표와 첫 실패 위치는 `representative_cases.json`에 있다.

## 3. 추가 관측 한 번 — 실제 적분과 경로 재검사

같은 24개 후보와 드론별 마지막 초기 위치에서의 직선 이동 비용 ≤6 m를 사용했다.
실제 비행 경로 계획이나 온라인 실험이 아닌 **기록 시점 재생**이다. 후보 포즈는 기존 전체
궤적 최적화 결과이며 온라인 의사결정 성능으로 해석하지 않는다.
기존 융합은 전 궤적 GT 정렬 초기화도 포함한다. 이번 결과는 이 포즈 조건에서의 지도·판정
구성 요소 비교이며, 사전 인프라 없는 현장 위치추정 성능의 검증이 아니다.
두 포즈 조건의 모든 선택 로그를 먼저 확정하고, 선택한 영상·깊이와 기준 메시를 평가에 읽었다.
각 방식은 동일 초기 전체 log-odds/표면층의 독립 복제본에 관측 한 번만 더했다.

`nearest`: 최근접, `unknown`: 미관측 공간, `decision_impact`: 기존 구간 중요도/가시성,
`decision`: 단일 프레임 증거와 직렬 구간 가시성을 요구하는 보수적 대조다.
후자를 필수 제안 방법으로 채택하지 않았다. 최소 2프레임·순 표면 증거 1.2 규칙 때문에
전혀 처음 보는 발판은 한 번으로 확인되지 않는다는 구현상의 상한이 있다.

| condition | variant | policy | candidate_id | move_cost | additional_observations | valid_path_confirmed | n_proposed | n_invalid_paths | n_unverified | wrong_path_rate | unverified_to_valid_path | new_invalid_guidance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | surface_geometry | initial |  | 0 | 0 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | surface_geometry | nearest | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | surface_geometry | unknown | 13 | 4.6717 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | surface_geometry | decision | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | surface_geometry | decision_impact | 3 | 3.7331 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | surface_geometry | initial |  | 0 | 0 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | surface_geometry | nearest | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | surface_geometry | unknown | 13 | 4.6717 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | surface_geometry | decision | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | surface_geometry | decision_impact | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | geometry | initial |  | 0 | 0 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | geometry | nearest | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | geometry | unknown | 13 | 4.6717 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | geometry | decision | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| clean | geometry | decision_impact | 3 | 3.7331 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | geometry | initial |  | 0 | 0 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | geometry | nearest | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | geometry | unknown | 13 | 4.6717 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | geometry | decision | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |
| heterogeneous_weak | geometry | decision_impact | 23 | 0.1943 | 1 | 0 | 0 | 0 | 22 | N/A | 0 | 0 |

잘못 안내한 경로 비율은 invalid/(valid+invalid), 분모 0은 N/A다.
제안 0건에서 오류 0건은 성능 개선이 아니다. 판정률은 (통과 가능+차단)/22이며 CSV에 있다.
보류→유효 경로, 보류→올바른 차단, 새 오안내, 기존 통과 취소, 오안내 취소와 유효 안내 손실도
집계 CSV에 저장했다. 이 결과로 재관측 우위를 주장할 수 있는지는 실제 목표·경로 수로 판단해야 한다.

## 4. 선택적 가중치 비교

| condition | arm | n_observations | voxel_iou | voxel_iou_1vox | voxel_n_false_free | voxel_n_gt_occ_domain | voxel_false_free_rate | known_completeness | area_reach_recall | area_false_reach_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | uniform | 300 | 0.0819 | 0.1833 | 26557 | 80655 | 0.3293 | 0.5651 | 0.0604 | 0.0004 |
| clean | depth_only | 300 | 0.0735 | 0.1695 | 26485 | 80655 | 0.3284 | 0.5592 | 0.0812 | 0.0009 |
| clean | pose_depth | 300 | 0.0615 | 0.1371 | 26809 | 80655 | 0.3324 | 0.5387 | 0.1009 | 0.0012 |
| heterogeneous_weak | uniform | 300 | 0.0798 | 0.1813 | 26634 | 80655 | 0.3302 | 0.5647 | 0.0605 | 0.0003 |
| heterogeneous_weak | depth_only | 300 | 0.0712 | 0.1673 | 26596 | 80655 | 0.3298 | 0.5590 | 0.0799 | 0.0010 |
| heterogeneous_weak | pose_depth | 300 | 0.0597 | 0.1358 | 26938 | 80655 | 0.3340 | 0.5386 | 0.0977 | 0.0011 |

같은 초기 300쌍·원래 포즈·지도 해상도·깊이·동료 제외를 사용했다. 지지층과 후보 선택을 바꾸는
실험이 아닌 **부피 점유 갱신 가중치의 구성 요소 대조**다. 포즈+깊이 가중의 이득을 가정하지 않았다.
3D IoU/false-free는 고정된 전체 기준 known 복셀 영역에서 계산했다. false-free 분모는
`voxel_n_gt_occ_domain`, 관측 완성도 분모는 `n_gt_known`이다. 미관측을 늘려 얻은 낮은 오류율도
같이 보이도록 완성도를 보고했다. 1복셀 허용 IoU도 따로 표시했다.
이 대조에서 목표 경로 검증을 다시 수행한 것은 아니며, 면적 지표만으로 사람 경로 유효성을 주장하지 않는다.

## 해석과 채택 범위

연구 목적에 맞는 실험은 완료했지만 모든 구성 요소를 주 방법으로 채택한 것은 아니다.
표면 지지와 몸 공간 검증은 계속 검토할 직접 관련 구성 요소다. 두 프레임 규칙은 검증 중인
보수적 설정이고 필수 안전 정의가 아니다. 상대 Z 등록은 한 층의 겹치는 바닥이라는 가정과
보정 공분산 미반영 문제가 있으므로 면적 재현율만 보고 채택하지 않는다.
불확실성 가중과 새 점수도 실제 지도 활용도·오안내·완성도·비용의 이득으로 판단한다.
임계값, 목표와 기준 상태를 성능에 맞춰 조정하지 않았다.

실험은 단일 스캔 장면·한 시드·약한 연기만 다룬다. 이번 재개에서는 드론 1/2/3대 비교나
동시 배정을 새로 검증하지 않았으므로 협업 규모에 따른 효율 개선을 추가로 주장하지 않는다.
현재 연기 모델은 적외선 영상의 거리 의존 감쇠·대비 저하 근사이며, 실제 화재·온도·가스 위험
검증이나 실제 연기 농도로의 환산이 아니다.

## 산출물과 재실행

판단 근거: `experiment_review.json`, 설정·입력 해시: `configuration.json`.
목표/제시 경로: `ablation_goals_paths.csv`, `policy*_goals_paths.csv`, `paths/`, `policy_paths*/`.
국소 지도: `local_geometry_details.csv`, `local_cases.json`, `representative_cases.json`.
선택: `selection*/`, 프레임 ID·드론 포즈·입력 순서·전후 지도: `replay*.json`, `maps*/`.
가중치: `weighting_summary.csv`, `weighting_frames.csv`, `weight_maps/`.
큰 지도와 경로는 `/src/gs25058/cr_RNE/covor_slam/results/fire_mapping_resume_20261104`에 있고, 집계·설정·선택 로그는 이 보고서 옆에도 복사했다.
기존 보고서 152개와 중단 산출물 444개의 해시 보존을 확인했다.
기존 지도 7개와 원본 설정/포즈/기준 입력 5개의 저장된 해시도 확인했다.
새 테스트 10건·기존 목표/경로 테스트 15건·기존 면적 탐색 테스트 11건이 통과했다.

이번 재개에서 수정/추가한 코드는 `scripts/reobserve/repair.py`(정렬 오류 수정·독립 대조·선택과
평가 분리·같은 후보 결과 재사용), `scripts/reobserve/reviewed.py`(가중치·국소 형상 평가·보고),
`tests/test_support_surfaces.py`(정렬 회귀), `configs/fire_mapping_resume.json`이다.
원래 융합·점유 갱신·면적 탐색의 기본 동작은 변경하지 않았다.

```bash
export MPLCONFIGDIR=/tmp/covor_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/src/gs25058/miniconda3/envs/covor/bin/python
O=results/fire_mapping_resume_20261104_replay
$PY scripts/reobserve/reviewed.py --out $O --stage prepare
$PY scripts/reobserve/repair.py --out $O --stage surfaces
$PY scripts/reobserve/repair.py --out $O --stage register
$PY scripts/reobserve/repair.py --out $O --stage integrate
for V in surface_geometry geometry; do
  $PY scripts/reobserve/repair.py --out $O --stage select --policy-variant $V
done
for V in surface_geometry geometry; do
  $PY scripts/reobserve/repair.py --out $O --stage policy-evaluate --policy-variant $V
done
$PY scripts/reobserve/repair.py --out $O --stage assess --variants raw,support,surface_geometry,registered,geometry
$PY scripts/reobserve/reviewed.py --out $O --stage weights
$PY scripts/reobserve/reviewed.py --out $O --stage local
$PY scripts/reobserve/reviewed.py --out $O --stage report
$PY tests/test_support_surfaces.py
$PY tests/test_decision_paths.py
$PY tests/test_entry_traverse.py
```
