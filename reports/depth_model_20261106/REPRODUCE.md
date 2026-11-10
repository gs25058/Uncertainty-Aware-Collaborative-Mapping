# 재실행

기존 v4 기본 명령은 바꾸지 않았다. backend 필드가 없는 설정은 `legacy_sgbm`이다. 새 실행은 별도 설정·출력 경로를 쓴다. 기존 데이터와 실패 기록 및 사용자가 미리 stage한 PNG는 유지했다.

```bash
cd /src/gs25058/cr_RNE/covor_slam
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
$COVOR_PY tests/test_depth_observation.py
$COVOR_PY scripts/explore/validate_depth_model.py --out results/depth_replay/validation
$COVOR_PY scripts/explore/validate_depth_strata.py --out results/depth_replay/validation_strata
$COVOR_PY scripts/explore/replay_depth_backend.py --source results/closed_loop_20261105/final_v4/scene11_seed0_clean_unknown --out results/depth_replay/fixed_trajectory
$COVOR_PY scripts/explore/run_closed_loop.py --config configs/depth_model_20261106.json --scene 11 --seed 0 --condition clean --policy unknown --out results/depth_replay/pilot_unknown
$COVOR_PY scripts/explore/run_closed_loop.py --config configs/depth_model_20261106.json --scene 11 --seed 0 --condition clean --policy entry --out results/depth_replay/pilot_entry
$COVOR_PY scripts/explore/run_matrix.py --config configs/depth_model_20261106.json --out results/depth_replay/main --workers 16
$COVOR_PY scripts/explore/run_matrix.py --config configs/depth_model_1hz_20261106.json --out results/depth_replay/rate1 --workers 2
$COVOR_PY scripts/explore/summarize.py --root results/depth_replay/main --out reports/depth_replay/data
$COVOR_PY scripts/explore/summarize.py --root results/depth_replay/rate1 --out reports/depth_replay/rate1_data
$COVOR_PY scripts/explore/analyze.py --root results/depth_replay/main --out results/depth_replay/analysis
$COVOR_PY scripts/explore/verify_depth_study.py --root results/depth_replay/main --out results/depth_replay/main_checks.json --expected 32
$COVOR_PY scripts/explore/verify_depth_study.py --root results/depth_replay/rate1 --out results/depth_replay/rate_checks.json --expected 2
$COVOR_PY scripts/explore/replay_cached_depth.py --run results/depth_replay/main/scene11_seed0_clean_unknown --out results/depth_replay/cache_check --until 8
$COVOR_PY scripts/explore/archive_depth_study.py --root results/depth_replay --report reports/depth_replay
```

대표 그림은 아래처럼 별도로 생성한다. 깊이 그림에 사용하는 fixture 참 깊이는 검증 자료이고 정책 입력에는 저장하지 않는다. 본 실행 디렉터리에 PNG를 생성하지 않는다.

```bash
$COVOR_PY scripts/explore/plot_depth_model.py --root results/depth_model_20261106 --out reports/depth_model_20261106/figures
```

실제 실행한 결과의 `replay_full/`은 위 예시의 `fixed_trajectory/`에 해당한다. 새 경로에서 그림도 재생하려면 `--out results/depth_replay/replay_full`로 고정 궤적 재생을 실행하고 `plot_depth_model.py --root results/depth_replay --out reports/depth_replay/figures`를 쓴다. `archive_depth_study.py`는 source_manifest와 현재 코드 일치를 검사하고 주기/관측/면적 표 및 소스 사본을 저장한다. `summarize.py` 이후 이를 실행하면 새 backend의 replay_manifest에서 존재하지 않는 legacy left/right PNG 항목도 제거한다. 실제 재생 입력은 NPZ다.

현재 서버의 본 실행은 자원 범위 내에서 30개 worker를 사용했다. 최초 완료한 scene11/seed0/clean 두 실행은 같은 설정임을 확인하고 32회에 포함했다. worker 수는 동기식 시뮬레이션의 물리 시각을 바꾸지 않는다. 실제 외부 계산 시간은 별도 기록하고 실시간 성능으로 사용하지 않는다. 완료 summary가 있는 실행은 재사용하며, 새 재현에는 새 출력 경로를 권장한다. FUEL 원본 Polynomial 공유 라이브러리가 필요하면 기존 `scripts/explore/build_backend.sh`를 먼저 실행한다.

설정 `sensor.backend`를 `ideal_depth`로 바꾸면 같은 새 카메라의 가림·범위를 유지하고 확률 오차·bias·drop·outlier 없이 생성한다. `depth_error_model`은 고정 오차 프로필을 사용한다. `legacy_sgbm`은 기존 카메라 설정(`configs/closed_loop_v4_20261106.json`)과 함께 실행한다. 기존 SGBM의 크기/보정 옵션을 새 모델의 K로 해석하지 않는다.

주 출력 NPZ는 `frames/<drone_tick>.npz`: Z[m], sigma[m], valid, 실제 world←left-camera T, time, drone_id, 현재 동료 위치를 담는다. 참 깊이/오류/outlier/T투과율은 없다. `frames.csv`의 camera_pose가 적분에 쓴 추정 T이며 camera_pose_truth가 센서 생성 T다. 미래 포즈 최적화나 사후 정렬은 하지 않는다. 촬영마다 map_version이 1 증가하고 프레임은 한 번씩만 적분한다. `trajectories.csv`의 실제·추정 위치와 실제 짐벌 자세를 함께 저장한다.

`simulator_audit.csv`는 평가 전용 관측 분모와 오차 집계다. 정책은 이 파일을 읽지 않는다. `candidates.jsonl`, `assignment_scores.csv`, `assignments.jsonl`, `events.csv`, `decisions.jsonl`, `target_paths.csv`, `metrics_time.csv`, `maps/*.npz`는 기존 v4 형식이다. 조건 clean/weak/medium/strong을 구분한다. 후속 그림/재생을 위한 깊이만 저장하며 본 실행의 좌우 PNG와 영상은 만들지 않는다.

재생은 frames.csv의 map_version 순서와 추정 T를 사용한다. 새 시뮬레이터에서 관측 자체를 다시 생성할 때에는 저장한 실제 T를 직접 `RayDepthSensor.measure(T,time,drone_id,teammates)`에 넘긴다. Euler 자세로 변환했다가 되돌리는 반올림을 피한다. 생성 키에는 정책명과 호출 순서가 없다. 고정 궤적 대조는 기존 body trajectory의 0.25초 실제 위치/yaw/짐벌을 읽고 새 baseline으로 카메라 T를 계산한다. 두 backend의 카메라·시각·지도·사람 규칙은 동일하다.

명목 profile 파일은 `configs/sensors/d455_inspired_optimistic.json`이다. 실제 샘플 격자는 실험 설정의 sensor.width/height, 적분 주기는 sensor_period다. profile의 4Hz 값은 기본값이고 1Hz 대조는 실제 주기를 따로 명시한다. 오차 σ는 격자 크기와 독립이다. 새 관측 모델의 광선·오염·품질 수식은 [SENSOR_MODEL.md](SENSOR_MODEL.md)에 기록했다.
