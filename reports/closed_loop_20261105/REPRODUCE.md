# 재실행과 저장 형식

작업 트리: `/src/gs25058/cr_RNE/covor_slam`, 기반 브랜치 `paper/conference-2026`, 기존 commit `dffbb8e`. 사용자 추가 파일 `다중드론_관측배정_이동계획_구현지침_20261105.md`를 적용했다. 모든 기존 자료는 유지했다. 새 장면은 이동 중 실제 위치에서 새 영상을 생성해야 하므로 MILUV의 옛 프레임이나 corridor915f의 옛 지도를 새 건물 관측으로 재사용하지 않았다. 기존 렌더러·모의 연기·SGBM·점유·표면·진입 그래프·경로 검증 코드를 연결해 재사용했다.

현재 서버 명령:

```bash
cd /src/gs25058/cr_RNE/covor_slam
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
bash scripts/explore/build_backend.sh
$COVOR_PY scripts/explore/run_closed_loop.py --config configs/closed_loop_v4_20261106.json --stage truth --out results/closed_loop_replay/truth
$COVOR_PY scripts/explore/run_closed_loop.py --config configs/closed_loop_v4_20261106.json --stage single --duration 24 --out results/closed_loop_replay/single
$COVOR_PY scripts/explore/validate_crossing.py --out results/closed_loop_replay/crossing
$COVOR_PY scripts/explore/run_closed_loop.py --config configs/closed_loop_v4_20261106.json --stage pair --duration 24 --out results/closed_loop_replay/pair
$COVOR_PY scripts/explore/run_matrix.py --config configs/closed_loop_v4_20261106.json --out results/closed_loop_replay/final --workers 8
$COVOR_PY scripts/explore/run_matrix.py --config configs/closed_loop_pose_v4_20261106.json --out results/closed_loop_replay/pose --workers 4
$COVOR_PY scripts/explore/analyze.py --root results/closed_loop_replay/final --out results/closed_loop_replay/analysis
$COVOR_PY scripts/explore/summarize.py --root results/closed_loop_replay/final --out results/closed_loop_replay/tables
$COVOR_PY scripts/explore/replay_map.py --run results/closed_loop_replay/final/scene29_seed0_smoke_unknown --out results/closed_loop_replay/map_check --sgbm
```

완성된 폐루프 실행은 `summary.json`이 이미 있으면 덮어쓰기를 거절한다. matrix는 완료된 실행을 읽고 미완료 실행만 재개한다. 새 재실행에는 위처럼 새 경로를 사용한다.

구성 요소 테스트:

```bash
$COVOR_PY tests/test_closed_loop.py
$COVOR_PY tests/test_decision_paths.py
$COVOR_PY tests/test_short_path_probe.py
$COVOR_PY tests/test_support_surfaces.py
$COVOR_PY tests/test_entry_traverse.py
$COVOR_PY tests/test_entry_sgbm.py
$COVOR_PY tests/test_smoke_reobserve.py
```

초기 v1 결과는 `results/closed_loop_20261105/main`, 그 위치 오차 대조는 `pose_error`에 남겼다. v1에서 가치 함수 비교 전에 미관측 점수로 5개를 추리는 공통 사전 필터와 진입 가시성의 5 m 상수가 발견됐다. v2(`final`, `pose_final`)부터 최대 24개 전부에 같은 이동 검사를 하며 실제 센서 4.5 m 범위를 쓴다. 중간 정지 예약을 고친 v3는 `final_v3`, `pose_v3`다. 최종 v4(`final_v4`, `pose_v4`)는 배정 직전 진입 그래프와 활성 작업의 이득도 같은 최신 지도 버전으로 갱신한다. 보정 전 모든 실행은 별도 진단 자료로 남겼고, 최종 집계는 v4의 실패 실행도 전부 포함한다. v1의 정확한 소스는 `main/source_snapshot`, `pose_error/source_snapshot`에 별도 보존했다. 최초 전체 재실행 지도와 SGBM의 비트 단위 재현은 `audit/replay_checks.json`에 있다. 위치 오차 어댑터를 추가한 뒤 기본 정확 위치 모드의 8초 지도도 정확히 같음을 확인했다(`audit/pose_adapter_default_check.json`).

실행별 파일:

| 파일 | 의미 |
|---|---|
| `config.json`, `prior.json`, `pose_error.json` | 전체 설정, 공통으로 알려준 건물 외곽·목표 구역·입구·실외 대기 공간, 위치 오차 |
| `reference_targets.json` | 평가 전 고정한 독립 형상 정답; 계획 모듈에는 전달하지 않음 |
| `frames.csv` | 실제 촬영 시간, 고유 프레임 ID, 드론 ID, 해당 프레임 적분 후 map version, 깊이 유효율, 센서/지도 계산 시간 |
| `frames/*_left.png`, `*_right.png`, `*.npz` | 실제 입력 mono8 양안 영상과 다시 계산한 Z/sigma/valid, 렌더링 참 카메라 T, 동료 실제 위치 |
| `trajectories.csv` | 0.25초마다 실제 xyz, 추정 xyz, yaw, 실제 짐벌 pitch, 활성/고장 상태 |
| `maps/*.npz` | 8초마다 및 마지막의 점유 labels·ijk·res·version·발판 support/height/evidence/votes |
| `candidates.jsonl` | 당시 관측 지도만으로 생성한 후보 위치·yaw·pitch 프로그램, 예상 unknown, 목표별 entry 점수, 구간별 발판/몸 공간 가시성 |
| `assignment_scores.csv` | 실행 가능한 각 드론–후보의 실제 계획 비용, 한계 가치, 비용 차감 후 점수, 예약 대기 |
| `assignments.jsonl` | 확정한 드론–후보, 폴백 여부, FUEL 선분 계수용 endpoints/duration, 계획 시각, 종료·촬영 시각 |
| `events.csv` | 모든 후보 경로 검사 목록, 경로/예약 실패, 대기, 새 점유로 인한 정지 요청과 정지, 충돌 |
| `decisions.jsonl`, `target_paths.csv` | 4초마다 목표의 세 상태·순서 있는 실제 제시 경로·가능성 경로·구간 절단 대조·경로 검사 결과 |
| `metrics_time.csv`, `summary.json`, `transitions.csv` | 시간별/마지막 목표 지표와 초기→최종 변화 |
| `timing.csv` | 사람 판정, 후보/배정/이동 계획, 지도 추출, 기준 경로 평가의 외부 계산 시간 |

재생에는 `frames.csv`의 map_version 순서대로 프레임을 한 번씩 적분한다. `frames/*.npz:T`는 렌더링 참 카메라 포즈다. **위치 오차 실험의 지도 재생에는 `frames.csv:camera_pose`의 추정 카메라 포즈를 쓴다.** `camera_pose_truth`와 npz의 T는 센서 생성/평가용이다. 정확 위치 조건에서는 두 값이 같다. 실제 짐벌 영상 재생에는 `trajectories.csv:gimbal_pitch`를 사용한다. 배정 로그의 pitch 준비 보간 값보다 실제 주기적 짐벌 스캔이 우선이다. 최종 버전의 센서 시간은 render/smoke/stereo로도 분리했다. 원본 데이터·기존 map·실험 CSV를 변경하지 않았다.

같은 궤적의 센서 요인 대조 재실행: `scripts/explore/replay_ideal.py --run <원래 실행 폴더> --out <새 출력 폴더>`. 후보를 다시 선택하지 않고 기록된 포즈/시각/동료 가림을 고정한다. 센서 자체 표본 검증: `scripts/explore/check_stereo_quality.py --run <무처리 실행 폴더> --out <새 출력 폴더>`. 원본 SGBM/지도 재생은 `replay_map.py`가 참 장면을 읽지 않고 수행하며, `--sgbm`을 빼면 저장 깊이를 재사용한다.
