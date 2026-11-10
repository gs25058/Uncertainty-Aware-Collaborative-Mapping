# 실행·재생 명령과 소스

**2026-11-08 재개 실행:** 현재 출력은 `results/fire_recon_20261108_resume`, 보고서는 이 디렉터리다. 아래 `20261107` 경로는 중단 전 입력·gate·자산과 최초 실행의 이력을 가리킨다. 현재 실행/재개 명령과 동결 소스 106개, 정확 복원 검증은 [RESUME.md](RESUME.md)에 있다. 현재 출력에 대한 감사·집계는 다음 명령으로 수행한다. 실행 중이면 완료를 기다리며, 별도 잠금으로 중복 후처리를 막는다.

```bash
bash scripts/explore/finish_resumable_recon.sh
```

최신 상태는 `results/fire_recon_20261108_resume/pipeline_status.json`이다. `main/complete.json`은 40개 최종 지도와 결과 및 소스 불변 검사가 끝난 뒤 생기고, `phase=complete`는 독립 센서 재생 감사와 보고서 생성까지 끝난 상태다. 중단 전 결과 폴더를 완성된 신규 결과로 읽지 않는다. 현재 소스/설정은 `main/source_manifest.json`, `main/source_snapshot/`, `configuration_manifest.json`; 원본 소스 98개는 이전 폴더에 그대로 있다.

```bash
cd /src/gs25058/cr_RNE/covor_slam
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
# 보완 장면과 원본 보존/발판 검증
$COVOR_PY tests/test_fire_recon_scene.py
$COVOR_PY scripts/explore/validate_fire_recon.py --config configs/fire_recon_20261107.preflight.json --scene 1 --out results/recon_check_new/complete
# 작은 실제 폐루프 두 정책: 새 경로를 사용
$COVOR_PY scripts/explore/run_fire_recon.py --config configs/fire_recon_20261107.preflight.json --scene 1 --seed 0 --condition clean --policy unknown --duration 16 --out results/recon_check_new/pilot_unknown
$COVOR_PY scripts/explore/run_fire_recon.py --config configs/fire_recon_20261107.preflight.json --scene 1 --seed 0 --condition clean --policy entry --duration 16 --out results/recon_check_new/pilot_entry
$COVOR_PY scripts/explore/replay_cached_depth.py --run results/recon_check_new/pilot_unknown --until 16 --out results/recon_check_new/map_replay
# 보완된 실제 GLB/PLY/충돌 형상/복셀을 별도 내보내기
$COVOR_PY scripts/explore/export_recon_assets.py --config configs/fire_recon_20261107.json --out results/recon_assets_new
```

실제 전체 인수/검증/실행 자동화는 `prepare_recon_gates.py`, `recon_study_driver.py`이며, 출력은 `results/fire_recon_20261107`다. 이 둘은 이번 실행 경로를 명시한 연구 실행기다. 다른 위치로 재현할 때는 아래의 범용 CLI에 각 gate 경로와 새 output을 전달한다. gate는 실제 geometry hash와 같은 입구/목표·사람 설정을 요구한다.

```bash
# 기존 검증 gate 사용, 새 출력에서 모든 40조건을 처음부터 실행하려면:
$COVOR_PY scripts/explore/run_recon_matrix.py \
  --config configs/fire_recon_20261107.json \
  --gates results/fire_recon_20261107/gates_complete \
  --out results/recon_main_new --workers 20
$COVOR_PY scripts/explore/summarize.py --root results/recon_main_new --out reports/recon_main_new/data
```

이번에는 동일한 2층 20회를 `reuse_recon_runs.py`로 포함한 뒤 위 matrix를 돌려 나머지 3층 20회를 실행한다. 재사용은 실제 동일 형상/숫자 설정/핵심 코드와 대표 깊이의 비트 동일을 모두 검사하고, 재사용 run과 실패 기체 수를 명시한다. 기존 source metadata는 복사하므로 후처리 replay_manifest 변경이 원본을 덮어쓰지 않는다. 복원 가능한 큰 frames/maps 디렉터리는 원본 경로를 참조하는 링크다. 다른 서버로 옮길 때에는 참조 대상도 함께 복사한다.

실제 주 결과의 후처리:

```bash
$COVOR_PY scripts/explore/summarize.py --root results/fire_recon_20261108_resume/main --out reports/fire_recon_20261108_resume/data
$COVOR_PY scripts/explore/post_recon_study_v2.py --root results/fire_recon_20261108_resume --report reports/fire_recon_20261108_resume
$COVOR_PY scripts/explore/analyze_fire_decisions.py --root results/fire_recon_20261108_resume/main --out reports/fire_recon_20261108_resume/data
$COVOR_PY scripts/explore/report_recon_study.py --root results/fire_recon_20261108_resume --report reports/fire_recon_20261108_resume
```

새 재실행 결과에서 감사하려면 고정 설정/fixed_config, source_manifest, 실행 완료 complete.json, 정답 및 gate 상태와 재사용 여부도 같은 root 구조에 보존한다. 소스 해시가 다른 결과를 현재 코드와 같다고 간주하지 않는다. 주 실행 시점의 source_snapshot을 사용한다. 오래된 source snapshot에는 이후 작성한 후처리 도구가 없을 수 있어 postprocessing_source_manifest와 현재 보관된 후처리 소스를 함께 보존한다.

`run_fire_recon.py`는 기존 runner의 world factory만 해당 프로세스에서 바꾼다. 기존 `run_closed_loop.py`/기본 FireWorld를 보완 장면의 실행 명령으로 혼동하지 않는다. 관측 가치/센서/지도/사람/이동 코드를 별도로 복사하지 않았다. 이번 3D 라이브러리는 direct-depth backend를 사용하며 GLB의 가상 재질에서 SGBM을 실행한 결과가 아니다.

재생 입력은 실행별 `frames.csv`의 map_version 순서, `frames/*.npz`의 Z/sigma/valid/T/time/drone_id/teammates, CSV의 추정 camera_pose, `trajectories.csv`의 실제/추정 위치·yaw·실제 gimbal_pitch다. 4Hz의 각 촬영은 실제 위치/시각의 새 관측이며 동일 깊이를 복제하지 않는다. 후보·배정·경로는 candidates/assignments/decisions JSONL과 target_paths.csv, 이벤트는 events.csv, 지도는 maps NPZ에 있다. replay_manifest는 NPZ만 가리키며 존재하지 않는 양안 PNG를 요구하지 않는다.

환경/외부 코드 커밋/라이브러리 해시: `intake/environment.json`, 기반 commit `57888219b8d9e4b02e078800b012b6a858040960`. D455-inspired 프로필은 `configs/sensors/d455_inspired_optimistic.json`, 범위/단위/가정은 [방법](METHOD.md)과 [기존 센서 모델](../depth_model_20261106/SENSOR_MODEL.md)에 있다. 요청된 별도 HTM의 동일성은 아직 확인되지 않았으며 실제 사용한 입력의 해시는 `intake/assets_manifest.json`에 있다.

## 유효 깊이 0개 프레임의 통계

원래 후처리는 유효 픽셀 0개 프레임의 빈 오차 칸을 float로 읽다 실패했다. 실행 중 동결한 소스와 raw CSV를 바꾸지 않고 `post_observation_audit.py`/`post_recon_study_v2.py`로 집계만 보완한다. 오차는 N/A, 풀링 가중치는 0이며 촬영 수/비용에는 포함한다. 원래 driver의 후처리 실패가 발생하면 `finish_recon_post.py`가 별도 로그를 남기며 감사·집계·보고만 이어간다. 정책 실행을 다시 돌리거나 실패 실행을 제외하지 않는다. 실제 후처리 코드 해시는 별도 postprocessing_fix_manifest/zero_valid_manifest에 기록한다.

## 최종 사후 진단

```bash
bash scripts/explore/finish_resumable_diagnostics.sh
```

학교는 방법별 결과를 보기 전 preflight에서 고정한 대표 장면이다. 선택한 기준 경로에서의 첫 부족 단계와 최종 무효 안내를 기록하며 정책에 결과를 되돌려 주지 않는다. 완료 기록은 `diagnostics_complete.json`, 전체 감사는 `audit/main_checks.json`이다. 원본 대비 모든 최종 지도 필드·궤적·주요 운용 지표를 비교한 결과는 `audit/geometry_repair_comparison.csv`다.
