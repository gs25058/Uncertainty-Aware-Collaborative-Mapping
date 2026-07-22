# CoVOR-SLAM on MILUV

A from-scratch implementation of **CoVOR-SLAM** (Lee et al., 2023 — *Cooperative
SLAM using Visual Odometry and Ranges for Multi-Robot Systems*), run on the
**MILUV** dataset (DECAR group). It fuses each robot's monocular visual odometry
with UWB inter-agent and anchor ranges in a **Sim(3) factor graph**, recovering
metric, jointly-consistent multi-robot trajectories **without inter-agent loop
closure** — drift is corrected by ranges, not by visual place recognition.

- **Front-end:** ORB-SLAM3 monocular VO per robot (up-to-scale keyframes).
- **Back-end:** GTSAM factor graph in Python — 7-DoF Sim(3) states, VO odometry,
  scale drift, UWB range, and laser-height factors, solved with Levenberg–Marquardt.
- **Evaluation:** ATE RMSE against mocap ground truth, per robot.

## Pipeline

```
MILUV images ──► ORB-SLAM3 (monocular)  ──►  per-robot up-to-scale VO keyframes
MILUV uwb_range.csv ─────────────────────►  inter-agent + anchor ranges
                                              │
                                              ▼
                        Sim(3) factor graph (GTSAM, Python)
                        · Pose3 (metric) + scalar scale per keyframe  = 7-DoF Sim(3)
                        · VO odometry between-factors (scale-corrected)
                        · scale random-walk factors (scale drift)
                        · UWB inter-agent range factors   (native RangeFactorPose3)
                        · UWB anchor range factors        (native, anchor = pinned Pose3)
                        · laser height factors            (downward altimeter -> z)
                        · frame-alignment / prior factors (gauge)
                                              │
                                Levenberg-Marquardt
                                              ▼
                     metric multi-robot trajectories  ──► ATE vs mocap
```

## Results

On `default_3_zigzag_0` (3 robots, line-of-sight), per-robot ATE RMSE:

| robot  | keyframes | mono-VO (Sim3) | CoVOR fused (SE3) |
|--------|-----------|----------------|-------------------|
| ifo001 | 202       | 0.091 m        | 0.568 m           |
| ifo002 | 307       | 0.129 m        | 0.538 m           |
| ifo003 | 391       | 0.284 m        | 0.425 m           |

**Fusion does not beat VO on this sequence — and that result is the finding, not a
bug.** The pipeline was validated separately: fed *perfect* ranges (mocap
`gt_range`) with the moment-arm + height model, fused ATE matches VO and the
drifty robot ifo003 improves (0.284 → 0.130 m). The limiter here is the **data**,
not the code:

- **Real UWB noise (σ ≈ 0.22 m, outliers to ~0.9 m) exceeds the VO error of
  robots 1 & 2** (0.09–0.13 m), so range fusion can only pull them off a good
  solution. As a rule of thumb, **fusion helps a robot iff its VO error ≳ the UWB
  noise floor.** This LOS sequence is simply too easy for VO.
- **Anchor geometry is vertically weak** (6 anchors all ~1.7 m high → poor VDOP);
  the downward-laser **height factor** restores the unobservable z.

A harder, longer, or higher-speed multi-robot sequence — where VO actually drifts
past the UWB noise floor — is where fusion is expected to win.

## Getting started

### Prerequisites

- **MILUV dataset** (DECAR) — Figshare [doi:10.25452/figshare.plus.28386041](https://doi.org/10.25452/figshare.plus.28386041).
  Each sequence provides per-robot `uwb_range.csv` (with `range`, `gt_range`,
  `std`, `bias`), `mocap.csv`, `height.csv`, plus `config/uwb/{anchors,tags}.yaml`
  and `config/height/bias.yaml`.
- **ORB-SLAM3** built with Pangolin / OpenCV / Eigen / Boost (for the VO front-end).
- Two conda environments (see [Environments](#environments)).

Data locations are set by the path constants at the top of `covor/data.py`
(`DATA`, `UWB_CFG`, `HEIGHT_CFG`, `VO_DIR`) and `RESULTS` in `run_covor.py` —
edit these to match your machine.

### Run

```bash
conda activate slam_env        # C++ / VO side
# 1) front-end VO — per robot; runs ORB-SLAM3 monocular
python scripts/prep_vo.py default_3_zigzag_0 ifo001   # generates config + image list
bash   scripts/run_vo.sh  default_3_zigzag_0 ifo001   # repeat for ifo002, ifo003
#   (or: bash scripts/run_vo_all.sh default_3_zigzag_0   -- all 3 robots, with retries)

conda activate covor           # Python / GTSAM fusion side
# 2) fusion + evaluation
python run_covor.py default_3_zigzag_0
# -> results/<seq>_trajectories.png, _errors.png, _summary.json
```

Pre-computed VO keyframes for `default_3_zigzag_0` and `default_3_random_0` ship
in `vo_output/`, so you can run step 2 without building ORB-SLAM3.

### Sanity checks

```bash
python scripts/selftest_fusion.py   # synthetic end-to-end factor-graph test
python scripts/test_jacobians.py    # analytic vs finite-difference Jacobians (~1e-7)
```

## How it works

### Sim(3) states without `gtsam.Similarity3`
The paper optimizes 7-DoF Sim(3) states. GTSAM's Python `Similarity3` does not
expose the Lie-group tangent ops (`retract` / `localCoordinates`) needed to use it
as an optimization variable, so each keyframe state is a metric `Pose3`
(translation = metric world position) **plus a scalar scale variable** — the same
7 DoF. The scale enters the VO odometry factor (scaling the up-to-scale relative
translation), and a random-walk factor on scale models scale drift, matching the
paper's per-keyframe Sim(3) formulation.

### Range / height factors and Jacobians
Range residuals `||p_a − target|| − z` use GTSAM's **native C++ `RangeFactorPose3`**
(analytic Jacobian, no Python callback → fast); anchors are fixed points modelled
as strongly-priored `Pose3` variables so the same native factor applies. The
custom analytic builders in `factors.py` (unit-tested against finite differences
*and* the native factor in `scripts/test_jacobians.py`) drive the online-bias
mode. The height factor is a unary `p_z − h` with analytic Jacobian
`dr/dξ = [0,0,0, e_z^T R]`. All analytic Jacobians match finite differences to
~1e-7 or better.

### All three robots are anchored
Each robot's own `uwb_range.csv` holds *its own* anchor ranges;
`data.load_ranges` unions all robots' CSVs and de-duplicates, so every robot gets
its full set of anchor measurements. (Reading only ifo001, as an earlier version
did, silently dropped 2/3 of the anchors and left robots 2 & 3 unanchored.)

### Assumptions
- **Antenna offsets** are treated as pre-compensated (paper Sec. II-B), so range
  residuals use the camera translation directly.
- **Frame initialization** (the paper's `SL1L2` / initial scale from ArUco / known
  GT scale) is seeded by aligning each robot's VO to mocap (Umeyama Sim(3)). The
  optimization then relies on VO odometry + ranges; mocap is used only for the
  initial guess and for evaluation.
- Monocular ORB-SLAM3 can lose tracking on drone footage and reset (new atlas
  map); odometry factors are skipped across large keyframe time gaps (map breaks).

## Repository layout

```
run_covor.py                   end-to-end: build → optimize → evaluate → figures
covor/
  data.py                      MILUV loaders (VO, mocap, ranges, anchors, tags, height)
  factors.py                   factor builders (odometry, scale, range, height, priors)
  fusion.py                    graph assembly, Umeyama init, LM optimization
  evaluate.py                  ATE RMSE + trajectory / error plots
configs/                       per-robot ORB-SLAM3 camera configs (ifo001/2/3)
orbslam_driver/
  mono_miluv.cc                ORB-SLAM3 monocular front-end for MILUV
scripts/
  build_driver.sh             compile the driver against the built ORB-SLAM3
  prep_vo.py                  generate ORB-SLAM3 config + image list per robot
  run_vo.sh / run_vo_all.sh   run VO for one robot / all three (with retries)
  selftest_fusion.py          synthetic end-to-end test of the factor graph
  test_jacobians.py           analytic-vs-finite-difference Jacobian unit tests
  sweep.py                    UWB weighting grid sweep
  sweep_height.py             VO / UWB / height / UWB+height ablation + grid
vo_output/                     pre-computed VO keyframes (sample inputs for fusion)
```

## Environments

- **`slam_env`** (conda): ORB-SLAM3 + Pangolin / OpenCV / Eigen / Boost — C++ build
  and VO front-end.
- **`covor`** (conda, py3.10): `gtsam` 4.2 + numpy / scipy / pandas / matplotlib —
  the fusion back-end.

## Reference

- **CoVOR-SLAM** — Lee et al., *Cooperative SLAM using Visual Odometry and Ranges
  for Multi-Robot Systems* (2023).
- **MILUV dataset** — DECAR group. Figshare [doi:10.25452/figshare.plus.28386041](https://doi.org/10.25452/figshare.plus.28386041).
