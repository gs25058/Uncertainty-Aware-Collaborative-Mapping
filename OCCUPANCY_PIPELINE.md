# Uncertainty-Aware Occupancy Mapping on CoVOR-SLAM Poses

This is the research-proposal pipeline (§3③, §4) built on top of the CoVOR-SLAM
UWB-fused poses: it turns the fused trajectory + its registration covariance into
a **navigation occupancy map** whose evidence is **weighted by uncertainty**. It
is the point where the project moves from reproduction to the proposal's novelty:
the pose covariance Σ that CoVOR computes and normally discards is forwarded into
the map, and combined with stereo depth uncertainty σ_Z to weight every log-odds
update.

Everything is pure Python (OpenCV SGBM + `octomap-python` `updateNode` with a
float log-odds argument); no C++ of OctoMap/OpenCV is modified, matching the
proposal's §7.3 tooling table.

## Modules

| file | role | proposal step |
|---|---|---|
| `scripts/fuse_and_dump.py` | run UWB fusion, extract fused pose `T` + `tr(Σ_pos)` per keyframe → `vo_output/occ_*.npz` | §2.5, Phase 2-4 |
| `covor/occupancy.py` · `StereoDepth` | rectified SGBM stereo → depth `Z`, uncertainty `σ_Z`, back-projected `P_cam` | §4.2–4.3, Phase 1-2/1-3 |
| `covor/occupancy.py` · `OccupancyBuilder` | ray traversal + **weighted log-odds** into an OctoMap tree; classify; `.bt` export | §4.4–4.8, Phase 1-5 / 3-2 |
| `scripts/build_occupancy.py` | driver: 1 / 2 / 3-drone map + top-down and slice viz | Phase 1-6 / 3-3 |
| `scripts/compare_weighting.py` | ablation: uncertainty-weighted vs uniform (standard OctoMap) | Phase 3-7 |
| `scripts/gt_pose_control.py` | control: same map from mocap GT poses; scores objects and floating voxels against GT | diagnostic |
| `scripts/export_occupancy_web.py` + `web/occupancy_viewer.template.html` | classified voxels → self-contained WebGL2 3D viewer (single HTML file) | Phase 1-6 / 3-3 viz |
| `tests/test_ray_traversal.py` | regression: per-cell log-odds vs the reference DDA | diagnostic |

`OccupancyBuilder` is the reusable `slam_to_occupancy` module (Phase 1-6): the
same instance ingests any number of `(pose, Σ, depth)` observations from one or
several drones into one shared tree, so the 1 vs 2 vs 3-drone comparison
(Phase 3-3) is just which `.npz` files are fed in.

## The weighted log-odds update (the novelty, §4.5–4.6)

For each beam from sensor origin `t` to the back-projected world point `P_world =
R·P_cam + t` (§3③), free evidence is added to the pass-through voxels and occupied
evidence to the endpoint voxel, each scaled by a per-observation weight:

```
l(c) ← clamp( l(c) + w · l_meas ),   l_meas = l_occ (endpoint) | l_free (pass-through)
w    = exp(−tr(Σ_pos)/α) · exp(−σ_Z²/β)                        # ∈ (0,1]
```

- `tr(Σ_pos)` — trace of the 3×3 position block of the 6×6 marginal covariance of
  the fused pose (registration uncertainty; large ⇒ down-weight).
- `σ_Z = Z²/(f·B)·Δd` — stereo depth uncertainty (grows with Z²; far ⇒ down-weight).
- `α, β` — the proposal's "experimentally determined" scale constants, set to the
  data's uncertainty scale so a median-quality observation keeps `w ≈ 0.85` and
  outliers collapse toward 0 (see `OccCfg` for the derivation).
- **Safety asymmetry** `|l_free| < |l_occ|` (`l_free=−0.40`, `l_occ=+0.85`): free
  space is asserted conservatively, occupied readily ⇒ fewer false-free. This
  holds only if the traversal contributes `l_free` to a cell **exactly once per
  beam** — see "Ray traversal" below.
- Weighting toggles off (`OccCfg.weighted=False` ⇒ `w≡1`) to recover a standard
  OctoMap update, making the ablation structural.

`unknown` is never conflated with `free`: pixels with no valid disparity cast no
ray (`Z=NaN`), and cells below the free threshold stay unknown (§4.8).

### Ray traversal

`_dda_batch` is a vectorized Amanatides & Woo traversal: all beams share the
camera origin, so tMax/tDelta are set up once and the active ray set marches in
lockstep, shrinking as beams reach their endpoint. One beam contributes to one
cell exactly once; different beams still accumulate (log-odds additivity, §4.7).
`_voxel_traverse` is the single-ray reference kept for `tests/test_ray_traversal.py`,
which asserts per-cell log-odds **values** (not just cell counts) against it.

### Classification thresholds (§4.8)

`tau_occ = l_occ = 0.85` is a definition, not a tuned value: one observation
contributes `w·l_occ ≤ l_occ` whatever `w` is, so `l > tau_occ` is unreachable
from a single observation — and 0.85 is the *smallest* threshold with that
property. Demoted cells become **unknown, not free**, and §4.8 forbids treating
unknown as traversable, so nothing gains traversability.

This subsumes a separate "trust occupied evidence only within 2–3 m" rule: with
`f·B = 21.7 px·m`, `w_depth = exp(−σ_Z²/β)` is already 0.94 / 0.71 / 0.34 / 0.07
at `Z` = 2 / 3 / 4 / 5 m, so clearing `tau_occ` takes roughly 2 / 2 / 4 / 17
observations at those ranges — a graded range dependence rather than a hard cutoff.

### Dynamic teammates

The drones see each other constantly, and a beam ending on a teammate is a
correct observation of a *moving* object that a static map should not keep.
`integrate_frame(..., teammates=...)` drops the endpoint evidence for beams
ending within `OccCfg.dyn_radius = 0.35 m` of another robot, keeping the free
evidence along the beam (that space really was traversed). Knowing where the
teammates are is possible only because the CoVOR fusion puts every robot in one
frame — a side benefit of §4.7, not an extra sensor.

### Export is terminal

OctoMap's `writeBinary` converts the tree to its maximum-likelihood estimate and
prunes it, so `write_bt()` destroys log-odds and merges cells. Classify and plot
first; `classify_points()` raises if called afterwards.

## How to run

```bash
PY=/src/gs25058/miniconda3/envs/covor/bin/python   # has gtsam + cv2 + octomap

# 1) fuse + dump poses & Sigma (once)
$PY scripts/fuse_and_dump.py

# 2a) single-drone occupancy (Phase 1)
$PY scripts/build_occupancy.py --drones ifo001 --weighted --stride 2

# 2b) uniform-vs-weighted ablation (Phase 3-7)
$PY scripts/compare_weighting.py --drones ifo001 --stride 2

# 2c) 3-drone collaborative map (Phase 3-3)
$PY scripts/build_occupancy.py --drones ifo001,ifo002,ifo003 --weighted --stride 3

# 2d) interactive 3D viewer (one self-contained HTML file, no server, no CDN)
$PY scripts/export_occupancy_web.py --drones ifo001,ifo002,ifo003 --stride 3
#    -> web/occupancy_viewer.html  (open in any WebGL2 browser)

# 3) diagnostics
$PY tests/test_ray_traversal.py                      # traversal regression
$PY scripts/gt_pose_control.py --robot ifo001        # pose-error vs mapping-error
```

## Results (`default_3_zigzag_0`, res 0.10 m)

| map | keyframes | occupied | free |
|---|---|---|---|
| ifo001, weighted | 190 | 22,055 | 495,241 |
| ifo001, uniform (standard OctoMap) | 190 | 62,327 | 501,973 |
| 3 drones, weighted | 464 | 40,780 | 721,556 |
| ifo001, **GT poses** (control) | 190 | 24,626 | 282,484 |

Ablation at the `z = 0.8 m` slice: of 8,497 cells the uniform map calls *free*,
473 (5.6 %) are held unknown/occupied by the weighted map, and they stay
spatially concentrated in the far / high-drift region — the false-free that §4.6
targets.

MILUV has no occupancy ground truth, but it does pin down the two things that
were visibly wrong, so both are scored directly:

- **objects** — `config/apriltags/apriltags.yaml` gives the 3D positions of the
  13 elevated tag stands, the room's only real objects (`experiments.csv` marks
  this sequence `obstacles_bool = false`; the obstacle sequences are
  `obstacles_1_random3_0b` and the `cirObstacles_*` set).
- **floating cubes** — occupied voxels strictly inside the room and above the
  floor, and how many coincide with a teammate's position.

| | objects w/ ≥3 voxels | object voxels | floating inside | of which on a teammate |
|---|---|---|---|---|
| fused poses, before the fixes | 10/13 | 272 | 1,101 | 36 % |
| fused poses, after | 9/13 | 100 | 736 | — |
| **GT poses, before** | 13/13 | 584 | 130 | 79 % |
| **GT poses, after** | 13/13 | 419 | **29** | 43 % |

## Faithfulness checklist (proposal §4 / §7.3)

| element | proposal | ours | verdict |
|---|---|---|---|
| Input pose | fused SE(3) `T^k_n` (UWB-aligned, not single-drone VO) | `covor.fusion` fused `Pose3` | **match** |
| **Σ forwarded** | §2.5 ★ registration covariance passed to mapping, not discarded | `gtsam.Marginals` → `tr(Σ_pos)` per keyframe | **match** |
| Depth | stereo SGBM, `Z=f·B/d`, 5 m cutoff, invalid⇒unknown | `StereoDepth` (MILUV infra1/infra2, B=0.05 m) | **match** |
| Depth uncertainty | `σ_Z=Z²/(f·B)·Δd` | same | **match** |
| Back-projection | `P_cam=[(u−cᵤ)Z/f, …]`, 4×4 downsample | same | **match** |
| World transform | `P_world=R·P_cam+t` (§3③) | same (pose = infra1 camera frame) | **match** |
| Ray casting | free = pass-through, occupied = endpoint, unknown behind | `_dda_batch`, vectorized Amanatides & Woo (§4.4 Bresenham-equivalent), regression-tested against the single-ray reference | **match** |
| **Weighted log-odds** | §4.5 ★ `w=exp(−trΣ/α)exp(−σ_Z²/β)`, `updateNode` direct | same, `updateNode(cell, w·l_meas)` | **match** |
| **Safety asymmetry** | §4.6 ★ `|l_free|<|l_occ|` | `l_free=−0.40 < l_occ=0.85` | **match** |
| Multi-drone accumulation | §4.7 log-odds additive into one tree | same builder, all robots | **match** |
| Classification | occ / free / unknown, unknown≠free | `tau_occ = l_occ` (one observation can never suffice), `tau_free = 0` | **match** |
| Ablation | Phase 3-7 uniform vs weighted | `weighted` flag + `compare_weighting.py` | **match** |
| Optimiser / tooling | GTSAM + OpenCV + OctoMap-python, no C++ edits | same | **match** |

### Justified deviations from the proposal's *intended* stack

- **Front-end is ORB-SLAM3 mono VO, not VINS-Fusion.** The proposal assumes a
  VINS front-end whose IMU gives metric scale and observes roll/pitch (gravity).
  We inherit the CoVOR reproduction's mono front-end (Sim(3) → Pose3+scale). One
  concrete consequence: map-break segments have a **free rotational gauge** (UWB
  ranges constrain only position), so their marginal covariance is regularized
  with a weak prior in `fuse_and_dump.py`; those segments correctly get a large
  `tr(Σ_pos)` and hence low weight `w`. This is the intended behaviour, reached
  through a different (documented) route.
- **α, β** are set to this data's uncertainty scale (the proposal calls them
  experimentally determined). They are not tuned to any accuracy metric — MILUV
  has no occupancy GT (§4.2 of the proposal), so this stage demonstrates that the
  pipeline runs as designed and that the weighting changes the map in the
  safety-conservative direction, rather than reporting an IoU/false-free number
  (which requires the simulator stage).
- **The room has almost nothing in it.** `experiments.csv` marks
  `default_3_zigzag_0` as `obstacles_bool = false`; the only real objects are the
  13 AprilTag stands, and they sit at the room perimeter, 0.6–3.8 m from the
  trajectory. Combined with `f·B = 21.7 px·m` (σ_Z = 0.42 m at 3 m, 1.15 m at
  5 m), this sequence is a weak demonstration of *object* mapping. The obstacle
  sequences (`obstacles_1_random3_0b`, already downloaded, and the
  `cirObstacles_*` set) are the right target for that.

## What this stage shows

On `default_3_zigzag_0`, a single drone yields a navigation occupancy map — free
interior, occupied walls, unknown beyond — from CoVOR fused poses + stereo. The
uncertainty weighting **changes the map in the safety direction**: in the
`z=0.8 m` slice, 5.6 % of the cells that the standard (uniform) map declares
*free* are held *unknown/occupied* by the weighted map, and those suppressed cells
are spatially concentrated in the far / high-drift regions — exactly the
false-free that §4.6 targets. Quantitative IoU / false-free evaluation against GT
is the simulator stage (proposal §4.2, Phase 1-6/3-3), not MILUV.

## Defects found and fixed (2026-07-23)

The first version of this pipeline produced a map whose room boundary looked
right but whose interior objects were missing and which was littered with
floating occupied voxels. Diagnosed by measurement, in this order.

1. **Free evidence was counted 2–4× per cell.** `integrate_frame` sampled each
   beam at `res/2` in *arc length* instead of doing a voxel traversal. Cells
   containing several samples of the same beam accumulated several times the
   intended `l_free` (measured 2× on axis-aligned rays, up to 4× on diagonals),
   so effective `l_free` was −0.8…−1.6 against `l_occ = +0.85` — the §4.6 safety
   asymmetry was silently **inverted**. Arc-length sampling also *skipped* cells a
   beam only clips (18 of 53 found on a body diagonal), biasing the carving by ray
   direction. Fixed with `_dda_batch`; the object voxel count rose 24 % (fused)
   and 30 % (GT), with the largest gains on the stands nearest the trajectory,
   which are seen from the most angles.
2. **`tau_occ = 0` made one stereo mismatch a permanent cube.** 75 % of the cells
   classified occupied were reachable by a single observation. Fixed by
   `tau_occ = l_occ` (see above); floating voxels 2,393 → 1,242 (fused) and
   225 → 98 (GT).
3. **Teammate observations were mapped as static structure.** 79 % of the
   floating voxels remaining in the GT map sat within 0.4 m of another drone.
   Fixed by `OccCfg.dyn_radius`; floating voxels 98 → 35 (GT) with the object
   count untouched. On fused poses this changes nothing (1,242 → 1,236), because
   the exclusion sphere is placed from an imprecise fused teammate position and
   the observer's own rotation error already puts the endpoint elsewhere.
4. **`build_occupancy.py` reported post-pruning leaf counts.** `write_bt` is
   destructive (max-likelihood + prune) and ran before `visualize`. Occupied
   structure barely prunes (1–3 % undercount, so the conclusions above are
   unaffected) but free space prunes heavily — the same run reported 172,802 free
   from this driver against 495,241 when classified first.

### Diagnosis: which failures are the map's, and which are the pose's

`scripts/gt_pose_control.py` rebuilds the map from mocap GT poses with everything
downstream unchanged, which splits the two. Against the fused poses (median
position error 0.19–0.26 m, median **rotation** error 21–34° after the best
constant camera–body offset, since UWB ranges constrain position only and the
mono front-end has no gravity reference):

- **Objects** are recovered by GT poses at 13/13 stands (median 38 voxels each)
  but only 9–10/13 under fused poses, worst on the *nearest* stands. Part
  mapping bug (fixed above), part pose error.
- **Floating cubes** are overwhelmingly a pose artefact: 736 remain inside the
  room under fused poses against 29 under GT poses.

So the remaining gap between this map and a clean one is the front-end, not the
mapping stage — consistent with the deviation recorded below.

### Not changed, and why

- **SGBM settings** already include the left–right consistency check
  (`disp12MaxDiff=1`), `uniquenessRatio=10` and speckle filtering
  (`speckleWindowSize=100`, `speckleRange=2`). Left as-is: the task is the
  proposal's design, not a stereo bake-off.
- **A hard "occupied only within 2–3 m" cutoff** is unnecessary — `w_depth`
  plus `tau_occ` already impose a graded version of it (see above).
- **Spreading the endpoint evidence over a σ_Z-wide Gaussian along the beam**
  would be a more complete uncertainty-aware sensor model than weighting alone.
  It is a design change beyond the current §4.5 formula and is *not* implemented;
  flagged as the natural next step.

---

## Which uncertainty term actually carries the weighting (2026-08-01, post-VINS)

The weight is a product of two terms, `w = exp(−trΣ/α) · exp(−σ_Z²/β)`. After the
front-end moved to VINS-Fusion and the fusion layer became SE(3), we measured what
each term contributes. **They swapped roles**, and the honest reading is that the
pose term is now dormant *by design* while the depth term carries the novelty.

### The pose term went quiet — because the poses got good

| era | median tr(Σ_pos) | spread | `w_pose` at α = 0.3 |
|---|---|---|---|
| ORB-SLAM3 mono | 0.05 m² (and 100–216 m² on gauge-free segments) | ~4000× | 0.85 … ≈0 |
| VINS + SE(3) + gravity prior | **0.0017 m²** | **2.6×** | 0.991 … 0.997 |

The old spread was not richness, it was **damage**: map breaks left segments with a
free rotational gauge, whose covariance blew up to 100–216 m². The weighting was
largely detecting broken poses. Remove the breaks (VINS), remove the scale gauge
(SE(3)), pin roll/pitch (gravity prior), and the uncertainty becomes **uniform along
the whole trajectory** — which is a *result*, not a regression.

Two measurements keep this honest rather than convenient:

- **Calibration.** NEES = 9.7 against an ideal 3, so Σ is 1.8× overconfident in σ.
  Mild by SLAM standards, and — importantly — a *uniform* scale error is absorbed
  entirely by α and changes no ranking.
- **Discrimination.** This is what actually matters, and it is weak. Binned by
  tr(Σ), the real error is **U-shaped** (lowest-Σ decile 0.074 m ≈ highest-Σ decile
  0.093 m). Controlling for attitude (Spearman within yaw strata, which is immune to
  the lever-arm model) recovers a positive relation on two robots (+0.15…+0.60) but
  **not on ifo001 (−0.41…+0.26)**. Median across 18 strata: +0.26. Real signal,
  weak and inconsistent.

**α stays at 0.3, deliberately.** Lowering it to "wake up" the pose term would
amplify a signal we have just shown to be unreliable, and choosing α so the ablation
shows a difference would make the ablation circular. At α = 0.3 the pose term sleeps
harmlessly when it has nothing to say, and still discriminates hard where it does:
in the 1/2/3-drone study (§4.9) a drone without UWB stays odometry-only, tr(Σ)
diverges along its chain past 0.1–1.0 m², and `w_pose` drops 0.72 → 0.036.

### The depth term carries it

`σ_Z = Z²/(f·B)` with `f·B = 21.7`, β = 0.5:

| range | σ_Z | `w_depth` |
|---|---|---|
| 1 m | 0.05 | 0.99 |
| median | 0.67 | 0.41 |
| 5 m | 1.17 | **0.065** |

**15× spread**, from a closed-form physical model rather than an estimator's
self-assessment.

### Narrative

The earlier ablation result — suppression concentrated in **far-range** and
**high-drift** regions — was already the sum of both terms: far-range is the depth
term, high-drift is the pose term. VINS removed the drift, so only the depth half
remains. Stated plainly:

> **We propagate both pose and depth uncertainty into the map. The better the
> front-end, the more the depth term dominates — and the pose term's value shows up
> not within a single well-constrained run, but between configurations where some
> agents are constrained and others are not.**

The pose term's job is to tell constrained regions from unconstrained ones. In a run
where all three drones are UWB-constrained throughout, there is nothing to tell
apart. That is an **absence of the condition, not a failure of the method**, and its
proper venue is the 1/2/3-drone comparison.

---

## GT-pose control on the VINS front-end (2026-08-06)

One stereo pass feeds three builders — GT poses (w_pose forced to 1), fused
weighted, fused uniform — at identical `OccCfg`/`DepthCfg`/stride/keyframe set, so
only the pose source and the weight differ. 3 drones, stride 4, 1632 frames.
**Predictions were written into the run script before measuring.**

### Mocap loading fix — the measuring instrument was contaminated

`load_mocap` sampled the **raw** csv nearest-in-time. MILUV's own loader
(`miluv/utils.py:130-188`) does not: it drops all-zero rows, drops any sample whose
rotation differs from the last good one by >1 rad **together with its predecessor**,
then fits csaps splines at smooth=0.9999. `covor.data` now ports those exact rules
(`_mocap_splines`, `mocap_pose_at`) and every mocap consumer routes through it —
`covor/fusion.py`, `scripts/gt_pose_control.py`, `run/eval_traj.py`, `run/plot_all.py`.

| | rows dropped |
|---|---|
| zigzag / ifo001 | 79 / 29,605 (0.27 %) |
| zigzag / ifo002 | 199 / 29,524 (0.67 %) |
| zigzag / ifo003 | **0** |
| obstacles / ifo001 | **0** |

ifo003 dropping nothing independently confirms the earlier finding that its mocap
carries no glitches, while ifo001/ifo002 do. This is **error removal, not
improvement** — the outliers we had been excluding by hand ("mocap dropout spikes",
gross >45° at 0–0.8 %) were exactly these rows.

What moved, and what did not:

| quantity | raw mocap | cleaned |
|---|---|---|
| VINS position RMSE (4 tracks) | 0.171 / 0.202 / 0.242 / 0.104 | **unchanged** |
| VINS tilt median | 0.90 / 0.84 / 0.63 / 0.69° | 0.87 / 0.82 / 0.60 / 0.68° |
| VINS tilt **max** | 171.6 / 173.6 / 8.9 / 24.1° | **27.1 / 10.3 / 8.8 / 24.2°** |
| gross rotation outliers >45° | 0.28 / 0.82 / 0 / 0 % | **0 / 0 / 0 / 0 %** |
| NEES (Σ overconfidence) | 9.7 → 1.8× | **9.7 → 1.8× unchanged** |
| GT-control floating cubes | 116 | **52** |

The headline numbers were robust to the contamination; the tails and the GT map were
not. `sigma_odo_trans/rot` and `sigma_tilt` were re-derived on cleaned mocap with no
hand filter and landed within 8 % of the hand-filtered values (0.0041 / 0.0124 /
0.0114 vs 0.0045 / 0.0131 / 0.0119), which is the check that the loader replaces the
hand filter rather than stacking with it. The `Cfg` defaults now carry the
loader-derived values.

### Frame fix: mocap is the MARKER pose, not the IMU pose

`gt_pose_control.py` composed `T_wc = T_w_marker @ inv(T_cam_imu)`, treating the
Vicon rigid body as the px4 IMU. They are different frames. `fit_marker_to_imu`
now estimates the missing rotation `B` from the **raw VINS front-end only** (never
the fused poses, so the reference is not calibrated against what it judges):

| robot | B | check: refit from fused |
|---|---|---|
| ifo001 | 3.27° (pitch +2.54, roll −2.06) | differs by 0.135° |
| ifo002 | 1.09° | 0.006° |
| ifo003 | 2.43° | 0.141° |

Ordinary mounting misalignment, and **non-circular** — refitting from the fused
poses moves it by ≤0.14°. Applying it cuts the fused-vs-GT rotation gap from
2.94–4.14° to **1.93–2.85°**, and what remains is just the fused rotation error
itself (0.9° tilt + 1.3–2.0° yaw), i.e. `B` did its whole job.

**The translation lever arm is NOT identifiable and was not applied.** The fit
residual (0.19–0.25 m, set by VINS drift) is larger than the lever arm itself, and
refitting from fused instead of raw VINS moves it by 0.062–0.195 m — more than its
own magnitude. Pinning it needs an independent observation; PnP on the AprilTag
stands would do it, but MILUV ships tag positions without detections.

### P1 — confirmed; see the milestone table below.

### ★ Milestone: fused poses now map as well as ground-truth poses ★

| map | stands ≥3 vox | object voxels | floating cubes | total occupied |
|---|---|---|---|---|
| GT control (B applied, cleaned mocap) | 13/13 | 600 | **52** | 65,471 |
| fused, weighted | **13/13** | 575 | **48** | 64,225 |
| fused, uniform | 13/13 | 663 | 53 | 96,794 |

ORB era: 9–10/13 stands and **1,101** floating cubes against GT's 130. Fused now
recovers **every** stand, floating cubes fell **1,101 → 48 (23×)**, and — the point —
that is **statistically the same as the GT-pose control's 52**. The thread that
opened when 21–34° rotation error was diagnosed as the map's limiting factor closes
here: with the SE(3)/VINS front-end, replacing the estimated poses with mocap buys
essentially nothing at this voxel resolution.

> **Correction.** An earlier revision of this document read the same comparison as
> "the GT control is no longer a valid upper bound", because fused (48) beat GT
> (116). That was **mostly a defect in how mocap was being loaded**, not a property
> of the reference: sampling raw mocap nearest-in-time let tracker glitches cast
> whole point clouds from wrong poses. With MILUV's own gap/outlier rules applied
> the GT control drops to 52 and the ordering is normal again. The residual
> lever-arm uncertainty (0.6–1.4 voxel) is still real, but it was the smaller effect.

### The occupied −41 % question

**Cell-level: AMBIGUOUS, and the sweep shows exactly why.** Of 40,014 removed cells,
26.1 % become free and 73.9 % unknown. Their GT status against tolerance:

| tol [voxel] | 0.25–0.75 | 1.00 | 1.25 | 1.50 | 2.00 | 3.00 |
|---|---|---|---|---|---|---|
| GT-occupied | 18.7 % | 42.0 % | 50.3 % | 66.4 % | 73.0 % | 87.5 % |
| GT-free | 41.4 % | 40.6 % | 36.9 % | 27.3 % | 22.9 % | 11.4 % |
| GT-unknown | 39.9 % | 17.4 % | 12.8 % | 6.3 % | 4.1 % | 1.1 % |

The GT-occupied share crosses 50 % at **1.24 voxel** — and the reference's own
residual offset is **0.6–1.4 voxel**, which brackets that crossover. The verdict is
therefore *undetermined by this data*, not merely unclear. No single threshold is
reported. Of the 10,426 cells that became FREE — the safety-critical direction —
the split at 1 voxel is 50.2 % GT-occupied vs 49.4 % GT-free, a coin flip. Range
concentration is in the predicted direction but weak: removed cells sit at median
**3.00 m** vs **2.56 m** for kept (σ_Z 0.41 vs 0.30 m).

**Aggregate: settled, and it answers the actual question.** These need no map-to-map
registration, so neither the lever arm nor the mocap glitches touch them:

| | weighted | uniform | GT |
|---|---|---|---|
| total occupied | 64,225 (**−1.9 %**) | 96,794 (**+47.8 %**) | 65,471 |
| object voxels | 575 (**−4.2 %**) | 663 (+10.5 %) | 600 |
| stands recovered | 13/13 | 13/13 | 13/13 |

Uniform over-declares occupied volume by **48 %** against GT; weighted sits within
**1.9 %** — **25× closer**, and 2.5× closer on object volume, while losing no
object. **The cells weighting removes are not the real obstacles.** That is the
question that was asked, and it is answered on registration-independent grounds; the
per-cell adjudication is a bonus, not the basis. **β is NOT adjusted.**

## §4.9 collaboration-gain experiment — DESIGN, implemented but NOT run

### The main axis is ANCHOR-FREE — the anchors are inherited, not designed

The proposal is anchor-free in three places: §1 lists it as a constraint ("사전
인프라 설치 불가능한 재난 현장 상정"), §2.4's objective has exactly three terms —
odometry, inter-agent ranges, prior — and states "anchor-free — 고정 앵커 항 없음",
and §2.2 calls the inter-drone range "the *only* observation linking them". §4.9's
causal chain is

> drones ↑ ⇒ UWB constraints **C(N,2)** ↑ ⇒ Σ ↓ ⇒ w ↑ ⇒ occupancy quality ↑

and C(N,2) counts **inter-agent pairs** (1 drone → 0, 2 → 1, 3 → 3). Anchors appear
nowhere in it. The anchors this pipeline uses are a **CoVOR-reproduction
inheritance** — the CoVOR paper uses them; this study does not. So the main ladder
drops them and they survive only as a side condition.

This is also what unblocks the pose term. With all three drones anchored, tr(Σ) is
0.0010–0.0026 m² and uniform, so `w_pose` sits at 0.991–0.997 and does nothing.
Anchor-free, the only absolute reference is the gauge prior, tr(Σ) grows along each
odometry chain, and the conditions separate — which is why §4.9 is the pose term's
proper venue.

**Expect the absolute numbers to get worse** (today's 6–8 cm fused error rests on
the anchors, and the anchor bias +0.087 m is what sets its floor). That is a return
to the proposal's target system, not a regression; condition D preserves continuity
with every number measured so far.

### Conditions — implemented as `Cfg.inter_pairs` / `Cfg.anchor_robots`

These change the **graph**, not the output filter. Verified by build counts:

| cond | inter_pairs | anchor_robots | inter factors | anchor factors | C(N,2) | role |
|---|---|---|---|---|---|---|
| **A** | `()` | `()` | 0 | 0 | 0 | 1 drone, VIO only — lower bound |
| **B** | `((0,1),)` | `()` | 3,226 | 0 | 1 | first pair |
| **C** | `((0,1),(0,2),(1,2))` | `()` | 9,712 | 0 | 3 | **proposal's target system** |
| D | `None` (all) | `None` (all) | 9,712 | 14,578 | 3 | current pipeline — continuity |
| E | `()` | `None` | 0 | 14,578 | 0 | anchors only — isolates infrastructure |
| GT | — | — | — | — | — | upper bound (after the frame fix above) |

### Separating coverage from pose quality — required for the causal claim

A→B→C changes two things at once: the number of UWB constraints *and* the number of
cameras contributing to the map. "More cameras ⇒ better map" is trivially true and
proves nothing about Σ. So:

- **A1 — quality at fixed coverage (the actual §4.9 test).** Build every map from
  **ifo001's observations only**, varying only the poses that conditions A–E produce.
  Any map difference is then attributable to pose quality alone, i.e. to Σ.
- **A2 — coverage gain (the proposal's headline figure).** Build from 1 / 2 / 3
  cameras. Deliberately confounded; reported as coverage, not as causal evidence.

### ⚠ Open decision — gauge and alignment convention

Anchor-free leaves **yaw + position (4 DoF)** free; VINS fixes roll/pitch. The
proposal (§2.3-다) fixes it with a prior on **drone 1's first pose only**. The code
currently priors **every** robot's first pose at σ_trans = 0.3 m, from a
mocap-seeded initial guess — harmless when anchors dominate, but anchor-free that
prior becomes a 0.3 m absolute reference on all three drones, comparable to the
tr(Σ) we are trying to measure. Options:

- **G1 (proposal-faithful, recommended)** — tight prior on robot 1 only; robots 2,3
  enter through inter pairs. Condition A has no pairs, so each robot is then its own
  gauge, which is the honest meaning of "1 drone alone".
- **G2 (uniform)** — same weak prior everywhere, simplest, but injects three
  mocap-derived absolute references and weakens the anchor-free claim.

Evaluation alignment, to be applied identically in every condition:
- **per-robot 4-DoF (yaw + translation)** for each robot's own ATE;
- **one joint 4-DoF** for all three, with the gap between the two being the
  *inter-robot registration error* — precisely what UWB is supposed to reduce, and a
  cleaner collaboration metric than ATE alone.

The map must be expressed in the mocap frame to be scored against the GT map, so the
alignment choice feeds directly into the map metrics. **This needs sign-off before
the run** — it also touches the deferred initialisation item (mocap-seeded Umeyama).

### Measurement axes

| axis | metric | why |
|---|---|---|
| registration uncertainty | tr(Σ_pos) distribution per condition | direct evidence the pose term engages |
| weight response | `w_pose` distribution | does the tr(Σ) change actually move the weight (today: 0.991–0.997) |
| map quality | IoU of occupied vs GT; **false-free rate** | proposal's stated metrics |
| causal link | correlation of tr(Σ) with false-free rate | §4.9's stated verification method |
| coverage | mapped voxel count, occlusion recovery | A2 only |
| localisation | per-robot ATE + inter-robot registration error | ties map back to poses |

Registration-independent cross-checks (object recall, object-voxel count, total
occupied vs GT) are reported alongside IoU, since they survived the frame error that
still limits per-cell scoring.

### Notes

- Condition A's tr(Σ) **will diverge along the chain**. That is correct and intended
  here — do not confuse it with the ORB-era "regularisation artifact" pitfall, where
  divergence came from map breaks leaving a free rotational gauge.
- `default_3_zigzag_0` only; `obstacles_1_random3_0b` ships just `ifo001.bag`.
- Cost: one fusion run per condition (~4 min) + one occupancy build (~6 min for A1's
  single camera, ~17 min for A2's three). Roughly 1.5–2 h for the full grid.

