# 재실행

저장소 루트에서 실행한다. Python은 `/src/gs25058/miniconda3/envs/covor/bin/python`이며 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`을 사용한다. 원본 결과와 체크포인트는 `results/entry_optimization_20261109/round03/precision_v4/`에 있다. 모든 출력은 새로운 경로를 사용한다.

```bash
PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

$PY tests/test_route_tasks.py -v
$PY tests/test_route_quality.py -v
$PY tests/test_candidate_views.py -v

$PY scripts/explore/diagnose_observed_support.py \
  --run results/fire_recon_20261108_resume/main/scene1_seed0_clean_unknown \
  --config configs/entry_current_20261109/pure/precision_v4.json \
  --out results/method_refinement_reproduce/diagnosis

$PY scripts/explore/probe_repair_method.py \
  --run results/entry_optimization_20261109/round03/precision_v4/scene1_seed0_clean_entry \
  --configs configs/method_refinement_20261110/precision_common.json \
    configs/method_refinement_20261110/repair_common.json \
  --out results/method_refinement_reproduce/common_pool.json

$PY scripts/explore/run_repair_branch.py \
  --parent results/entry_optimization_20261109/round03/precision_v4/scene1_seed0_clean_entry \
  --config configs/method_refinement_20261110/repair_common.json \
  --out results/method_refinement_reproduce/gate \
  --policy entry --additional 4

$PY scripts/explore/run_repair_study.py \
  --out results/method_refinement_reproduce/paired32 \
  --configs configs/method_refinement_20261110 --workers 24
```

실제 실행 소스는 `results/method_refinement_20261110/paired32/source_snapshot/`이고 상대 파일별 SHA256은 옆의 `source_manifest.json`에 있다. 실제 명령·입력·설정은 `fixed_spec.json`, `configs/*.json`에 저장된다. 이 스냅샷을 사용하면 이후 작업 트리의 선택적 판정·경로 비용 실험을 섞지 않고 재현할 수 있다.

고정 지도 재평가의 실제 소스는 `results/method_refinement_20261110/fixed_map_source/`, 해시는 `fixed_map_source_manifest.json`, 세 실행 명령 배열은 `fixed_map_commands.json`에 있다. 예:

```bash
$PY scripts/explore/replay_route_quality.py \
  --runs results/fire_recon_20261108_resume/main/scene4_seed0_clean_unknown \
  --out results/method_refinement_reproduce/fixed_map --complete-goal-cut

$PY scripts/explore/probe_vertical_views.py \
  --run results/entry_optimization_20261109/round03/precision_v4/scene1_seed0_clean_entry \
  --out results/method_refinement_reproduce/vertical_probe
```

실행 완료 후 집계:

```bash
$PY scripts/explore/report_method_refinement.py \
  --root results/method_refinement_20261110 \
  --out reports/method_refinement_20261110
```

원본 `transitions.csv`는 기존 runner의 임무 0초 대비 결과다. 이번 분기 비교에는 집계기가 만드는 `branch_transitions.csv`와 공개 `targets.csv`의 **180초 대비 변화**를 사용한다.

각 분기의 `branch_manifest.json`에 초기 확률 지도 체크포인트, 입력 해시, 실패 기체·예약·잔여 대기와 위치가 기록된다. `frames.csv`의 시각·map_version 순서로 `frames/*.npz`를 읽는다. 프레임은 중복 생성하지 않았고 원래 prefix는 변경하지 않는 hard link로 보존했다. `trajectories.csv`는 실제·추정 위치와 실제 yaw/pitch다. `assignments.jsonl`, `candidates.jsonl`, `assignment_scores.csv`, `events.csv`, `maps/*.npz`, `target_paths.csv`로 이동·선택·지도·경로를 재생할 수 있다. 초기 지도 복사본은 완료 집계 시 `maps/initial.npz`에 저장한다.

새 선택적 판정 기능은 `human.observed_route_quality_weight: 1.0`, `human.full_goal_cut_certificate: true`다. 기본 실행은 바뀌지 않는다. 두 기능은 주 폐루프 비교에서 사용하지 않는다. 새 관측 가치는 `utility.entry_objective: task_progress_v5`와 `task_progress.route_local_repairs/serial_repair_potential`로 선택한다. 공통 후보는 `candidates.body_reobserve_views/path_repair_views`와 `evidence_query_profile`로 선택한다.
