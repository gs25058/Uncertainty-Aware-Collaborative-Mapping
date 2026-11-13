# Uncertainty-Aware Collaborative Mapping

Multi-robot visual-inertial odometry fused with UWB ranges in a GTSAM factor graph,
following CoVOR-SLAM (Lee et al., 2023), and an occupancy mapper that carries the
resulting pose uncertainty into the map. Evaluated on the MILUV dataset.

- **Fusion back-end** (`covor/`): per-robot SE(3) trajectories from a VINS-Fusion
  stereo+IMU front-end, joined by UWB anchor and inter-robot range factors, a laser
  height factor and a gravity prior. Solved with Levenberg-Marquardt.
- **Occupancy mapping** (`covor/occupancy.py`): stereo depth is integrated into a
  log-odds grid, with each observation weighted by the fused pose covariance.
- **Evaluation**: ATE RMSE against cleaned mocap ground truth.

## Branches

| branch | contents |
|--------|----------|
| `main` | stable fusion and occupancy code |
| `develop` | work in progress: entry-map, exploration and simulation experiments |
| `paper/conference-2026` | write-ups, pre-registrations, result reports and paper figures |

## Setup

```bash
conda create -n covor python=3.10
conda activate covor
pip install -r requirements.txt        # core
pip install -r requirements-dev.txt    # experiments on develop
```

The dataset is not included. Download MILUV
([doi:10.25452/figshare.plus.28386041](https://doi.org/10.25452/figshare.plus.28386041))
and point the code at it:

```bash
export MILUV_ROOT=/path/to/miluv        # must contain data/ and config/
export VINS_OUT_DIR=/path/to/vins/out   # VINS-Fusion output, <seq>/<robot>/vio.csv
```

Both default to `data/miluv` and `data/vins_out` inside the repository, so symlinks
there work as well.

## Usage

```bash
python run_covor.py default_3_zigzag_0
```

This builds the graph, optimizes it, prints per-robot ATE for the VIO-only and fused
trajectories, and writes plots and `*_summary.json` to `results/`.

Reference result on `default_3_zigzag_0` (rigid-aligned ATE RMSE, metres):

| robot  | VIO only | fused |
|--------|----------|-------|
| ifo001 | 0.171    | 0.066 |
| ifo002 | 0.202    | 0.111 |
| ifo003 | 0.242    | 0.087 |

Occupancy maps:

```bash
python scripts/fuse_and_dump.py --seq default_3_zigzag_0   # fused poses + covariance
python scripts/build_occupancy.py                          # weighted occupancy grid
python scripts/export_occupancy_web.py                     # browser viewer
```

Tests:

```bash
python scripts/test_jacobians.py        # analytic vs finite-difference Jacobians
python -m unittest discover -s tests    # experiment modules
```

Some experiment tests load an external observation backend; set
`EXTERNAL_OBSERVATION_DIR` to its location.

### Legacy ORB-SLAM3 front-end

An ORB-SLAM3 monocular front-end is still available (`Cfg(frontend="orb")`).
Build the driver against an ORB-SLAM3 checkout (`ORB_SLAM3_DIR`, default
`../ORB_SLAM3`) with `scripts/build_driver.sh`, then run
`scripts/prep_vo.py` and `scripts/run_vo.sh` per robot. Keyframes for two sequences
are included in `vo_output/`.

## Layout

```
run_covor.py            end-to-end run: build, optimize, evaluate, plot
covor/
  data.py               MILUV loaders (VIO, mocap, ranges, anchors, height)
  factors.py            odometry, range, height and prior factors
  fusion.py             graph assembly, initialization, optimization
  evaluate.py           ATE and plotting
  occupancy.py          stereo depth and uncertainty-weighted occupancy grid
covor/entry/            entry-map construction and traversability (see its README)
covor/explore/          observation planning and fire-scene exploration
covor/synth/            mesh-based synthetic dataset generator (see its README)
covor/smoke.py, reobserve.py
scripts/                pipelines and experiment drivers, grouped by topic
tests/                  unit tests for the experiment modules
configs/                ORB-SLAM3 camera configs
orbslam_driver/         ORB-SLAM3 monocular driver source
vo_output/              sample ORB-SLAM3 keyframes
web/                    viewer templates
meshes/PROVENANCE.json  source of the scanned meshes (meshes themselves not tracked)
```

## Reference

- Lee et al., *Cooperative SLAM using Visual Odometry and Ranges for Multi-Robot
  Systems*, 2023.
- MILUV dataset, DECAR group.
