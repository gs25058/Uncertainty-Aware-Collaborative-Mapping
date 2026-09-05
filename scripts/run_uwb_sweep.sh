#!/usr/bin/env bash
# Parallel driver for the UWB noise sweep (PREREG_uwb_noise.md).
# Each worker runs ONE grid cell in its own process; cells already in the CSV are
# skipped by --plan, so this is safe to stop and restart.
#   bash scripts/run_uwb_sweep.sh [n_parallel]
set -u
N=${1:-20}
PY=/src/gs25058/miniconda3/envs/covor/bin/python
cd /src/gs25058/cr_RNE/covor_slam
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY scripts/sweep_uwb_noise.py --plan \
  | $PY -c 'import json,sys; [print(json.dumps(c)) for c in json.load(sys.stdin)]' \
  | xargs -d '\n' -P "$N" -I{} $PY scripts/sweep_uwb_noise.py --cell '{}'
# -d '\n' is required: without it xargs applies its own quote processing and eats
# the JSON's double quotes, so every worker dies on JSONDecodeError.
