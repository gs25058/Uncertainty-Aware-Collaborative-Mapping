# `covor.synth` — synthetic observations on a scanned room, with occupancy GT

MILUV provides no occupancy ground truth (`RESULTS_SUMMARY.md` §10), so IoU,
precision/recall and false-free can only be given in absolute terms on a
simulated space. This package renders a scanned room mesh into the **same
observation formats the real pipeline already consumes**, so `covor/fusion.py`
and `covor/occupancy.py` run on it **unmodified**:

| what the real run reads | what this package writes |
|---|---|
| `<DATA>/<seq>/<robot>/mocap.csv` | GT body pose (initial guess only) |
| `<DATA>/<seq>/<robot>/uwb_range.csv` | synthetic inter-agent ranges |
| `<DATA>/<seq>/timeshift.yaml` | a non-zero clock offset (exercises A-8) |
| `<VINS_DIR>/<seq>/<robot>/vio.csv` | raw **local-frame** VIO (A-2) |
| MILUV stereo images | rendered depth + `sigma_Z` (same formula) |

Nothing under `covor/` outside this package is edited. `dataset.install_paths`
redirects `covor.data.DATA` / `.VINS_DIR` at run time rather than copying a fake
sequence into the MILUV tree, so real and synthetic data can never be confused.

## Two traps that will cost you an afternoon

1. **`import gtsam` before `import open3d`.** open3d ships its own libtbb; if it
   is loaded first, `import gtsam` dies with an undefined TBB symbol. Every
   entry point here imports gtsam first.
2. **The sequence name must end in a MILUV anchor-constellation key** (`_0`).
   `CoVOR.__init__` calls `load_anchors(seq)` unconditionally. No anchor ranges
   are generated and `anchor_robots=()` is passed, so the constellation is never
   used — the name only keeps the loader from raising.

## Reproduce

```bash
P=/src/gs25058/miniconda3/envs/covor/bin/python     # needs trimesh + open3d

# A. mesh -> GT voxels (writes results/synth_<name>/)
$P scripts/synth/build_gt_voxel.py --obj <scan.obj> --name room909

# B. synthetic observations for one seed
$P scripts/synth/make_dataset.py --name room909 --seed 0

# C. integrity gate  (Part D must not run unless this passes)
$P scripts/synth/gate.py --name room909 --stride 4
$P scripts/synth/gate.py --name room909 --stride 8 --corrupt nobodycam  # neg. control
$P scripts/synth/diagnose_gate.py --name room909                        # if it fails
$P tests/test_synth_pipeline.py                                         # regression

# D. the experiment (one cell; --regen redraws that seed's noise)
$P scripts/synth/run_experiment.py --name room909 --seed 0 --cond C_3drone --regen
```

## Parameters

Everything already fixed by `RESULTS_SUMMARY.md` appendix B is **imported, not
redefined**: `covor.fusion.Cfg` and `covor.occupancy.OccCfg` supply the fusion
and mapping constants. What follows is only what a simulator has to invent —
each value is either measured on MILUV (marked *m*) or a stated design choice
(marked *d*). None was chosen by looking at a result.

### Geometry / GT (`mesh_gt.py`, `scripts/synth/build_gt_voxel.py`)

| parameter | value | why |
|---|---|---|
| grid | `i = floor(p/res)`, centre `(i+0.5)·res` | *identical to `covor.occupancy`*; asserted in `tests/test_synth_pipeline.py` |
| `res` | 0.10 m | appendix B |
| voxelisation | exact triangle/box SAT, 13 axes | `trimesh.voxelized` / open3d each anchor their own grid — the GT must not inherit a half-voxel offset from the thing it is judging |
| GT free | enclosure ≥ 0.90 of 128 rays, ∩ floor–ceiling band, 2-voxel closing, seed-connected | *d*; a scan is not watertight, so flood fill leaks (measured: 313,374 of 313,423 cells became one component). Sweep reported in `mesh_config.json` |
| GT unknown | everything else | excluded from every metric — never counted as free |

### Trajectories (`trajectory.py`)

| parameter | value | why |
|---|---|---|
| rate | 20 Hz raw → 10 Hz graph nodes | `Cfg.vins_stride = 2` |
| duration | 90 s | task brief (60–120 s) |
| speed | 0.35 m/s | *m*: MILUV p95 speed is 0.80 m/s; a mapping pass is flown slower |
| heights | 1.10 / 1.25 / 1.40 m ± 0.08 | *d*: inside the brief's 1.0–1.5 m band, one per robot, so 3 drones add vertical diversity |
| clearance | 0.35 m from any non-free cell | = `OccCfg.dyn_radius` |
| yaw sweep | ±50°, 12 s period | *d*: a lawnmower pass with a fixed heading never sees the side walls |
| zones | equal-area split along x | *d*: disjoint coverage, so map (ii) differs from map (i) |

Legality is by construction (BFS through the clearance mask) **and** re-checked
per sample; `plan_all` raises rather than returning a path that clips geometry.

### Front-end error (`vio.py`)

Not a plain random walk. Accumulating white relative-rotation noise at
`sigma_odo_rot` gives a 21° yaw drift over 90 s, three times worse than the real
VINS front-end (−4.5…+0.2 °/min, §10) — which would inflate the apparent value
of UWB fusion. The process is split into the pieces the real system has:

| term | value | why |
|---|---|---|
| yaw | drift at +1.5 / −2.5 / +4.0 °/min + a walk 0.3× that size | *m*: the measured VINS range; yaw is the DoF gravity does not observe |
| roll/pitch | OU, stationary σ = `Cfg.sigma_tilt` = 0.0114 rad, τ = 2 s | *m*: bounded, because gravity observes it |
| white rotation | σ solved so the node-to-node residual equals `Cfg.sigma_odo_rot` | calibrated by measurement inside `generate`, not by an analytic guess |
| position | accumulated white noise at `Cfg.sigma_odo_trans` per node | *m*: drift 0.12 m / 90 s, the right order |

`generate` returns the **achieved** statistics (relative residual, drift rate,
raw ATE) so the model is checked in the report rather than asserted.

### UWB (`uwb.py`)

| parameter | value | why |
|---|---|---|
| rate | 1.4 Hz per tag pair | *m*: MILUV 1.37–1.43 Hz |
| σ | 0.05 m | = `Cfg.range_sigma_floor` |
| bias | +0.004 m | *m*: MILUV inter-agent bias +0.0035 |
| moment arm | MILUV `tags.yaml` | appendix B `use_moment_arm=True` |
| NLOS | off by default; `--nlos` adds +0.30 ± 0.15 m where the mesh blocks the LOS | *d* |
| anchors | none generated | the proposal is anchor-free (§1, §2.4) |

The `std` column carries the **true** σ. On real MILUV that column is optimistic
(A-7); here the simulator knows the truth, so a mis-weighting cannot be blamed
on the csv.

### Rendering (`render.py`)

Intrinsics, rectification and back-projection all come from the unmodified
`covor.occupancy.StereoDepth`; the renderer only supplies `Z`. Rays are built in
the rectified-left frame with `d_z ≡ 1`, so open3d's `t_hit` **is** the depth —
the same quantity `f·B/disparity` produces. `mode="ideal"` uses mesh depth with
`sigma_Z = Z²/(f·B)·Δd`; `mode="sgbm"` renders a textured stereo pair and pushes
it through the real SGBM. A ray that hits nothing is **invalid, not max-range**,
so neither path ever carves free space it did not measure.
