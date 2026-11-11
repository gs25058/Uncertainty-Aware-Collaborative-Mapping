# Completion-opportunity 분기 재현

이 실행은 중단된 `results/method_search_20261110/opportunity32/`을 덮어쓰지 않는다. 완료된 결과는 `opportunity32_completed/`에 있으며, 실행 소스는 이전에 고정된 `opportunity32/source_snapshot/`이다.

```bash
cd /src/gs25058/cr_RNE/covor_slam
PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SNAP=results/method_search_20261110/opportunity32/source_snapshot/scripts/explore/run_repair_branch.py
CFG=results/method_search_20261110/opportunity32/config.json
OUT=results/method_search_reproduce/opportunity32_completed

for CASE in scene0_clean scene0_strong scene1_clean scene1_strong \
            scene4_clean scene4_strong scene8_clean scene8_strong; do
  scene=${CASE%%_*}; condition=${CASE#*_}; scene_num=${scene#scene}
  parent="results/entry_optimization_20261109/round03/precision_v4/scene${scene_num}_seed0_${condition}_entry"
  "$PY" "$SNAP" --parent "$parent" --config "$CFG" \
    --out "$OUT/$CASE" --policy entry --additional 32
done
```

각 명령은 부모의 180초 완료 체크포인트에 기록된 확률 지도, 발판 통계, hit 최대값, 기체 상태, 예약, 미처리 촬영과 관측 이력을 복원한다. 새 결과는 212초까지의 새 센서 관측과 이동만 기록한다. `branch_manifest.json`의 부모 체크포인트 커밋 해시와 `source_manifest.json`을 함께 보관해야 한다.

완료된 각 사례의 `summary.json`과 `target_paths.csv`를 같은 시간의 부모 결과와 비교한다. 전체 목표영역 차단 인증을 사용할 때는 `reports/method_refinement_20261110/whole_goal_cut_summary.csv`의 보수적 수치를 쓴다. 원래의 짧은 재개 출력(`opportunity32/`)에는 최종 요약이 없으므로 논문 표에 사용하지 않는다.
