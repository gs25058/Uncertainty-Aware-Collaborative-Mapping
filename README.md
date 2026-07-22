# CoVOR-SLAM on MILUV

An implementation of **CoVOR-SLAM** (Lee et al., 2023 — *Cooperative SLAM using
Visual Odometry and Ranges for Multi-Robot Systems*) run on the **MILUV** dataset
(DECAR group). It fuses per-robot monocular visual odometry with UWB inter-agent
and anchor range measurements in a Sim(3) factor graph to recover accurate,
metric, jointly-consistent multi-robot trajectories — without inter-agent loop
closure.

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

### Range/height factors and Jacobians
Range residuals `||p_a - target|| - z` use GTSAM's **native C++ `RangeFactorPose3`**
(analytic Jacobian, no Python callback -> fast); anchors are fixed points modelled
as strongly-priored `Pose3` variables so the same native factor applies. The
custom analytic builders in `factors.py` (unit-tested against finite differences
*and* the native factor in `scripts/test_jacobians.py`) are used for the
online-bias mode. The height factor is a unary `p_z - h` with analytic Jacobian
`dr/dxi = [0,0,0, e_z^T R]`. All analytic Jacobians match finite differences to
~1e-7 or better.

### All three robots are anchored
Each robot's own `uwb_range.csv` holds *its own* anchor ranges; reading only
ifo001 (as an earlier version did) dropped 2/3 of the anchor measurements and
left only robot 1 anchored. `data.load_ranges` now unions all robots' csvs and
de-duplicates, so every robot gets its ~5k anchor ranges.

### Why Pose3 + scalar instead of `gtsam.Similarity3`
The paper optimizes 7-DoF Sim(3) states. GTSAM's Python `Similarity3` does not
expose the Lie-group tangent ops (`retract`/`localCoordinates`) needed to use it
as an optimization variable, so each keyframe state is parameterized as a metric
`Pose3` (translation = metric world position) **plus a scalar scale variable**.
Together these are the same 7 DoF; the scale enters the VO odometry factor
(scaling the up-to-scale relative translation) and a random-walk factor on scale
models scale drift — matching the paper's per-keyframe Sim(3) formulation. All
factors are `gtsam.CustomFactor` with finite-difference Jacobians on the tangent.

### Assumptions / notes
- **Antenna offsets** are treated as pre-compensated (as the paper states in
  Sec. II-B), so range residuals use the camera translation directly.
- **Frame initialization** (`SL1L2` / initial scale in the paper — from ArUco /
  known GT scale) is seeded here by aligning each robot's VO to mocap (Umeyama
  Sim(3)). The optimization then relies on VO odometry + ranges; mocap is used
  only for the initial guess and for evaluation.
- Monocular ORB-SLAM3 can lose tracking on drone footage and reset (new atlas
  map); odometry factors are skipped across large keyframe time gaps (map breaks).

## Layout
```
orbslam_driver/mono_miluv.cc   ORB-SLAM3 monocular front-end for MILUV
scripts/build_driver.sh        compile the driver against the built ORB-SLAM3
scripts/prep_vo.py             generate ORB-SLAM3 config + image list per robot
scripts/run_vo.sh              run VO for one robot
scripts/selftest_fusion.py     synthetic end-to-end test of the factor graph
covor/data.py                  MILUV loaders (VO, mocap, ranges, anchors, tags, height)
covor/factors.py               factor builders (odometry, scale, range, height, priors)
covor/fusion.py                graph assembly, Umeyama init, LM optimization
covor/evaluate.py              ATE RMSE + trajectory/error plots
run_covor.py                   end-to-end: build → optimize → evaluate → figures
scripts/test_jacobians.py      analytic-vs-finite-difference Jacobian unit tests
scripts/sweep.py               UWB weighting grid sweep -> sweep_results.csv
scripts/sweep_height.py        VO / UWB / height / UWB+height ablation + grid
```

## Run
```bash
conda activate slam_env        # C++ / VO side
# 1) front-end VO (per robot; runs ORB-SLAM3 monocular)
python scripts/prep_vo.py default_3_zigzag_0 ifo001
bash   scripts/run_vo.sh default_3_zigzag_0 ifo001     # repeat for ifo002, ifo003

conda activate covor           # Python / GTSAM fusion side
# 2) fusion + evaluation
python run_covor.py default_3_zigzag_0
# -> results/<seq>_trajectories.png, _errors.png, _summary.json
```

## Environments
- `slam_env` (conda): ORB-SLAM3 + Pangolin/OpenCV/Eigen/Boost (C++ build & VO).
- `covor` (conda, py3.10): `gtsam` 4.2 + numpy/scipy/pandas/matplotlib (fusion).
