# 현재 완료 범위의 재현

대용량 입력·깊이·지도는 서버 results에 보존한다. 외부 FUEL native library와 환경은 기존 [외부 코드 감사](../../closed_loop_20261105/EXTERNAL_AUDIT.md) 및 [지도 입력 문서](../../../OCCUPANCY_PIPELINE.md)를 따른다. 실제 수학 소스는 동결 아카이브, 실행 설정은 configs/entry_current_20261109다.

```bash
export COVOR_PY=/src/gs25058/miniconda3/envs/covor/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p results/current_source_replay/control
tar -xzf reports/entry_optimization_20261109/current/sources/control.tar.gz -C results/current_source_replay/control
$COVOR_PY results/current_source_replay/control/scripts/explore/run_entry_trial.py \
  --config configs/entry_current_20261109/control/heading_rate.json \
  --out results/current_reproduction/heading_rate_scene1_clean_16s \
  --scene 1 --seed 0 --condition clean --policy entry --duration 16 \
  --memoize-motion --defer-dense
```

180초 재현은 새 출력 경로와 `--duration 180`을 사용한다. 같은 source archive를 유지해야 체크포인트를 재개할 수 있다. 현재 사용자 종료 범위 뒤의 전체 driver/validation을 자동 실행하지 않는다.

보고·감사는 새 관측 배정을 실행하지 않는다:

```bash
$COVOR_PY scripts/explore/report_current_entry_work.py
```

실행별 frames.csv가 촬영 순서·시각·지도 버전·실제/추정 포즈를, trajectories.csv가 실제 이동과 짐벌 방향을, assignments.jsonl/events.csv가 계획·감속·정지·실패를 보존한다.
