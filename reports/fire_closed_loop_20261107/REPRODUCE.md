# 실행과 재생

서버 Python은 `/src/gs25058/miniconda3/envs/covor/bin/python`이다. 아래 명령은 저장소 루트에서 실행한다. 새 실행은 항상 새 출력 폴더를 쓴다. 이미 완료된 이번 결과를 덮어쓰지 않는다.

```bash
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# 기준 형상/목표 검사와 완전 관측 사람 경로 검사
$COVOR_PY scripts/explore/prepare_fire_reference.py --config configs/fire_closed_loop_20261107.json --scene 1 --out results/fire_check_new/reference1
$COVOR_PY scripts/explore/validate_fire_human.py --config configs/fire_closed_loop_20261107.json --scene 1 --out results/fire_check_new/scene1
# 공통 센서의 clean/strong 고정 위치 검사
$COVOR_PY scripts/explore/validate_fire_sensor.py --config configs/fire_closed_loop_20261107.json --out results/fire_sensor_new
# 작은 폐루프: 같은 설정에서 짧게 한 정책 실행
$COVOR_PY scripts/explore/run_closed_loop.py --config configs/fire_closed_loop_20261107.json --scene 1 --seed 0 --condition strong --policy entry --stage run --duration 16 --out results/fire_pilot_new
# 40회 고정 본 비교. workers는 하드웨어에 맞춰 낮춰도 시뮬레이션 설정은 같음
$COVOR_PY scripts/explore/run_matrix.py --config configs/fire_closed_loop_20261107.json --out results/fire_main_new --workers 8
$COVOR_PY scripts/explore/summarize.py --root results/fire_main_new --out reports/fire_main_new/data
```

이번 실행의 자동 gate/소스 동결/40회/집계 명령은 `scripts/explore/fire_study_driver.py`이며 인자는 `results/fire_closed_loop_20261107/execution.json`에 있다. driver는 기존 reference_checks, gates_complete, pilot의 합격 산출물을 확인한 뒤 시작한다. `source_snapshot`에 이번에 실행한 미커밋 변경분을 포함한 전체 소스를 보존한다. HEAD만 checkout하는 것으로 이번 소스를 재현했다고 간주하지 않는다.

```bash
# 현재 결과의 사후 감사/시간 그래프/경로 길이/센서/정책 활성도 표
$COVOR_PY scripts/explore/post_fire_study.py
# 실제 관측을 다시 생성하지 않고 저장된 깊이만 적분: 20초 지도와 비트 단위 비교
$COVOR_PY scripts/explore/replay_cached_depth.py \
  --run results/fire_closed_loop_20261107/main/scene1_seed0_clean_unknown \
  --until 20 --out results/fire_replay_new
```

실제 완료한 20초 체크포인트 재생은 `scripts/explore/replay_fire_checkpoint.py`다. `--run`, `--until 20`, `--out`을 받으며 `perfect_odometry`를 명시적으로 검사한 뒤 공개 NPZ의 포즈와 동료 위치만 사용한다. 결과 `results/fire_closed_loop_20261107/replay_checkpoint20/verification.json`에서 276장 적분 후 모든 지도 필드가 비트 단위로 같음을 확인했다.

재생 입력 순서는 각 실행의 `frames.csv`(map_version 순)와 `replay_manifest.json`이다. `frames/d{drone}_t{tick}.npz`에는 Z/sigma/valid/실제 카메라 T/촬영 시각/드론 ID/동료 위치가 있다. 지도 적분은 CSV의 추정 camera_pose를 사용한다. `trajectories.csv`는 매 0.25초 실제·추정 위치/yaw/실제 짐벌 pitch, `assignments.jsonl`은 FUEL 구간/지속시간/예약 대기다. `maps/t{tick:05d}.npz`, `maps/final.npz`에는 지도와 모든 지지층/관측 높이 필드가 들어간다.

정답/소스는 `reference_final`, `evaluation_cases.json`, `smoke_fields_v2`에 있으며 정책 입력과 분리된다. 실행 소스 SHA-256은 `main/source_manifest.json`, 설정은 `main/fixed_config.json`, 기반 커밋은 `57888219b8d9e4b02e078800b012b6a858040960`이다. 외부 FUEL/RACER/EGO-Swarm 커밋은 [EXTERNAL_AUDIT](../closed_loop_20261105/EXTERNAL_AUDIT.md)에 있다.
