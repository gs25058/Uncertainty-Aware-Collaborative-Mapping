# 소방관 수색 구역의 진입 판단 비교 — 자료 인수

작업 시작: 2026-11-07 23:17 KST. 작업 트리는 `paper/conference-2026`, 기반 commit `57888219b8d9e4b02e078800b012b6a858040960`이다. 실행 시점의 미커밋 수정과 기존 stage 파일을 보존한다.

## 실제 확인한 자료

- `paper/README.md`, `OCCUPANCY_PIPELINE.md`, 루트 `다중드론_관측배정_이동계획_구현지침_20261105.md`.
- `reports/closed_loop_20261105/{METHOD,RESULTS,FIXES,EXTERNAL_AUDIT,REPRODUCE}.md`: 최종 v4는 `data_v4`, `final_v4`다. 레거시 SGBM 조건을 새 깊이 모델과 혼동하지 않는다.
- `reports/depth_model_20261106/{SENSOR_MODEL,RESULTS,REPRODUCE}.md`, `configs/depth_model_20261106.json`, `configs/sensors/d455_inspired_optimistic.json`: 정상 동작을 가정한 직접 깊이 관측 모델과 통제된 저하 조건. 32+2회 결과가 이미 있다.
- 최신 복층 연결 소스 `covor/explore/{fire_buildings,fire_world,fire_reference,fire_smoke,scene,sensing,depth_observation,human,motion,policy,runner}.py`와 `covor/entry/{surfaces,decision,path_eval,traverse}.py`.
- 확대 화재 viewer는 `web/fire_buildings_3d.html`과 `web/fire_buildings_large_fire_3d.html`이며 두 파일 해시가 같다.
- 건물 10개 실제 자산: `results/fire_buildings_large_fire_20261107/{01_apartment,...,10_library}/`의 `scene.glb`, `scene.ply`, `collision_geometry.json`, `scene.json`, `gt_voxel.npz`, `validation.json`.

요청한 이름 `fire_buildings_3d(1).htm`은 작업 트리와 확인한 사용자 서버 경로에서 발견되지 않았다. 기존 viewer의 다운로드 복사본인지 별도 장면인지 사용자에게 확인을 요청했다. 이번 메시지 본문은 §2의 “후보 생성 절차와 후보 수 제한”까지 수신했다. 이후 조건을 임의로 받았다고 간주하지 않는다.

## 보존과 실행 소스

`results/fire_recon_20261107/intake/`에 git status, unstaged/staged binary patch, 116개 소스·설정·문서의 해시와 사본, 62개 viewer/메시/메타데이터/복셀의 해시, 시작 상태를 기록했다. 이 시점은 새 보완 어댑터 구현 전이다. 구현 후 작은 gate의 소스는 `preflight/source_manifest.json`과 `preflight/source_snapshot/`에 따로 동결했다.

이전 원본 장면 40회는 `results/fire_closed_loop_20261107/main/`에서 진행 중이다. 해당 77개 소스/설정은 그대로 유지한다. 원본 실행과 새 보완 장면의 정답·분모·출력을 혼합하지 않는다.

## 이미 확인한 앞단 결함과 분리한 보완

모든 3층 건물의 2층 계단 시작점에 하부 층계참이 없어 약 27cm의 무지지 구간이 생겼다. 기준 메시 광선 검사 10곳 모두 같은 결함을 확인했다. 원본에서는 3층 목표 46개 전부가 기준 도달 불가이므로 3층 경로 확보 성능을 검증할 수 없다.

학교 복사본에 하부 층계참 2개만 추가한 인과 대조에서 기준 도달 가능 14→20개, 3층 도달 가능 0→6개를 확인했다. 20개 양성 경로 모두 별도 메시 평가기에서 유효했다. 사람의 크기/걸음/판정 설정을 완화한 결과가 아니다. 대조: `results/fire_closed_loop_20261107/diagnostics/school_landing_control/`.

새 `covor/explore/fire_recon_scene.py`는 opt-in `repair_bottom_landings`를 제공한다. 원본 부품·화재·목표는 유지하고 누락된 하부 층계참만 추가한다. 정답 cache 키에는 실제 고체 꼭짓점/삼각형과 영역, 기준 검사 코드가 들어가므로 이전 장면의 정답을 재사용하지 않는다. 다른 geometry hash의 certified 라벨은 거절한다.

새 `scripts/explore/run_fire_recon.py`는 프로세스 안의 world factory만 교체하고 기존 `runner.run`을 그대로 호출한다. unknown/entry 값, 후보 수, 배정/이동/충돌, 센서/지도, 사람 판정과 경로 평가를 복사하거나 서로 다르게 바꾸지 않는다. 원본 장면과 기존 명령의 기본 동작은 유지한다.

## 작은 검증 명령

```bash
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
$COVOR_PY tests/test_fire_recon_scene.py
$COVOR_PY scripts/explore/validate_fire_recon.py --config configs/fire_recon_20261107.preflight.json --scene 1 --out results/fire_recon_new/preflight
$COVOR_PY scripts/explore/run_fire_recon.py --config configs/fire_recon_20261107.preflight.json --scene 1 --seed 0 --condition clean --policy unknown --duration 16 --out results/fire_recon_new/pilot_unknown
$COVOR_PY scripts/explore/run_fire_recon.py --config configs/fire_recon_20261107.preflight.json --scene 1 --seed 0 --condition clean --policy entry --duration 16 --out results/fire_recon_new/pilot_entry
```

`configs/fire_recon_20261107.preflight.json`은 작은 연결 검사용이다. 아직 전체 비교가 완료된 설정이나 결과로 표시하지 않는다. 기본은 기존 낙관적 깊이 모델, 4대·4Hz·perfect_odometry이며 소스 기반 연기장과 동일한 공통 사람 판정기를 재사용한다.

실제 자산 대조는 10개 모두 통과했다(`intake/asset_geometry_checks.json`). 보완 장면은 원본을 덮어쓰지 않고 `results/fire_recon_20261107/assets/`에 별도 GLB·고체 PLY·충돌 JSON·메타데이터·기준 복셀로 내보낸다. 최종 평가 정답은 이 자산의 장면 표시용 상태와 구분해 독립 검사한다. 작은 검증의 실제 수치는 [PREFLIGHT](PREFLIGHT.md)에 기록한다.
