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

Rebuilt the control with the SE(3)/VINS poses. One stereo pass feeds three
builders — GT poses (w_pose forced to 1), fused weighted, fused uniform — with
identical `OccCfg`/`DepthCfg`/stride/keyframe set, so the only difference is the
pose source and the weight. 3 drones, stride 4, 1632 frames.

**Predictions were recorded before measuring** (in the run script's docstring):
P1 fused should now approach GT, since rotation error fell 21–34° → 0.6–0.9°;
P2 the cells weighting removes should be mostly not-occupied in GT and
concentrated at long range, since `w_depth` is the only term with real spread.

### P1 — confirmed, and then some

| map | stands ≥3 vox | object voxels | floating cubes inside room | total occupied |
|---|---|---|---|---|
| GT control | 13/13 | 549 | 116 | 70,869 |
| fused, weighted | **13/13** | 575 | **48** | 64,225 |
| fused, uniform | 13/13 | 663 | 53 | 96,794 |

ORB era: fused 9–10/13 stands and **1,101** floating cubes vs GT 130. Fused now
recovers **every** stand and floating cubes fell **1,101 → 48 (23×)**.

### ⚠ The GT control is no longer a valid upper bound

Fused shows *fewer* floating cubes than the GT control (48 vs 116). That cannot
mean the fused poses beat mocap. Measured cause: **the GT chain has a systematic
frame error**. `gt_pose_control.py` treats the mocap rigid-body pose as the px4-IMU
body pose, but they are not the same frame — the fused and GT camera poses differ by
**0.057–0.144 m (0.6–1.4 voxels at res 0.10) and 2.9–4.1°**, a constant offset, the
same marker↔IMU mount transform that `eval_traj.py` has to fit as its two-sided `B`.

So the front-end pose error is no longer the map's limiting factor — the *reference's*
frame error is. That is a real milestone, and it also means **the control must be
fixed before it can serve as the §5.2(A) upper bound**: compose
`T_wb = T_w_marker · B` with the marker→IMU rotation `B` and lever arm `l`, both
estimable from VINS-vs-mocap alone (front-end only, no fusion, so no GT leakage into
the estimator).

### P2 / the occupied −41% question — **AMBIGUOUS by the pre-registered rule**

Weighting removes 40,014 of 96,794 uniform-occupied cells (41.3 %): 26.1 % become
FREE, 73.9 % become UNKNOWN.

| GT status of removed cells | 0.5 voxel tol | 1.0 voxel | 2.0 voxel |
|---|---|---|---|
| GT-occupied | 19.4 % | 42.2 % | 71.7 % |
| GT-free | 42.5 % | 39.9 % | 22.8 % |
| GT-unknown | 38.1 % | 17.9 % | 5.6 % |

The verdict swings with tolerance, and the GT reference's own ~1-voxel frame error
sits exactly in the band that decides it. Range concentration is in the predicted
direction but weak: removed cells sit at median **3.00 m** from the camera vs
**2.56 m** for kept cells (σ_Z 0.41 vs 0.30 m) — a 1.17× ratio, not the sharp
far-field concentration P2 anticipated. Of the 10,426 cells that became FREE — the
safety-critical direction — 51.4 % are GT-occupied at 1-voxel tolerance.

**Per the rule fixed in advance, this is AMBIGUOUS, not an improvement claim.**

### What *is* clean: registration-independent aggregates

These need no map-to-map registration, so the frame error does not touch them:

| | fused weighted | fused uniform | GT |
|---|---|---|---|
| total occupied | 64,225 (**−9.4 %** vs GT) | 96,794 (+36.6 % vs GT) | 70,869 |
| object voxels (0.4 m radius on AprilTag GT) | 575 (**+4.7 %**) | 663 (+20.8 %) | 549 |

Uniform **over-declares** occupied volume by 37 % against the GT reference and
inflates the object stands by 21 %; weighted sits within 9 % and 5 %. Weighted is
**3.9× closer** to GT on total occupied and **4.4× closer** on object volume, while
losing no object (13/13). So the cells weighting removes are demonstrably not the
real obstacles, even though per-cell adjudication is currently blocked.

**Standing conclusion**: weighting moves the occupied geometry toward the GT
reference in aggregate and costs no object recall; whether the specific removed
cells are all spurious cannot be settled until the GT control's frame chain is
fixed. β is NOT adjusted on this evidence.

---

## §4.9 collaboration-gain experiment — DESIGN ONLY, not run

Proposal §5.2(A) fixes the conditions and says the implementation is "just change
the UWB pair list, `[(1,2)] → [(1,2),(1,3),(2,3)]`", with **IoU and false-free
rate** as the metrics, and §4.9 states the causal chain to be demonstrated:

> drones ↑ ⇒ UWB constraints C(N,2) ↑ ⇒ Σ ↓ ⇒ w ↑ ⇒ occupancy quality ↑

### ⚠ Blocker 1: the pair-list ablation is not implemented

`fuse_and_dump.py --drones` **only filters which robots get written out** — it does
not change the graph. The comment above it claims it performs the §4.9 ablation; it
does not. `Cfg` has global `use_ranges` / `use_anchor` / `use_inter` switches, but
nothing per-robot or per-pair. Needed: a `Cfg.inter_pairs` (tuple of robot-index
pairs, default all three) and a `Cfg.anchor_robots` (which robots get anchor
ranges), both applied in `CoVOR.build`'s range loop.

### ⚠ Blocker 2: the GT upper bound is not yet valid

See the section above — the control's marker↔IMU frame chain must be fixed first,
otherwise every IoU / false-free number is measured against a reference that is
itself displaced by ~1 voxel.

### Conditions

Proposal's table lists 1-drone as "VIO only / VINS-Fusion / drift as-is", i.e. no
ranges at all. But the 1→2 step then changes two things at once (anchors appear
*and* the first inter pair appears), while §4.9's causal claim is specifically about
C(N,2). Adding one rung separates them:

| # | condition | anchors | inter pairs | C(N,2) | role |
|---|---|---|---|---|---|
| 0 | VIO only | — | — | 0 | lower bound (proposal's "1대") |
| 1 | 1 drone + anchors | ifo001 | — | 0 | isolates infrastructure ranging |
| 2 | 2 drones | 1,2 | (1,2) | 1 | |
| 3 | 3 drones | 1,2,3 | (1,2),(1,3),(2,3) | 3 | main result |
| 4 | GT poses | — | — | — | upper bound (after the fix) |

### Two sub-experiments — coverage must not be confounded with pose quality

Comparing a 1-drone map to a 3-drone map changes *both* the number of cameras and
the pose accuracy. The §4.9 causal claim is about pose accuracy only, so:

- **A1 — quality at fixed coverage.** Always map with **ifo001's camera alone**, and
  vary only how many drones' UWB constrain ifo001's poses (conditions 0–4). This is
  the clean test of Σ → w → map quality.
- **A2 — coverage gain.** Map with 1 / 2 / 3 cameras (the proposal's headline
  figure). Deliberately confounded — it measures the whole collaboration benefit.

Both use identical `OccCfg`/`DepthCfg`/stride/keyframe set.

### Measurement axes

| axis | metric | why |
|---|---|---|
| registration uncertainty | tr(Σ_pos) distribution per condition | direct evidence the pose term engages; expect divergence to 0.1–1.0 m² for unconstrained robots, so `w_pose` swings 0.72 → 0.036 at α = 0.3 |
| map quality | **IoU** of occupied vs GT, **false-free rate** (cells the map calls free that GT calls occupied) | proposal's stated metrics |
| coverage | mapped voxel count; recovered volume behind occlusions | A2's point |
| localisation | ATE per condition | ties the map result back to the pose result |
| causal link | correlation of tr(Σ) with false-free rate | §4.9's stated verification method |

Registration-independent cross-checks (object recall, object-voxel count, total
occupied vs GT) should be reported alongside IoU, since they survived the frame
error that currently invalidates per-cell scoring.

### Sequence

`default_3_zigzag_0` only. `obstacles_1_random3_0b` has just `ifo001.bag` — no
ifo002/ifo003 bag or mocap — so it cannot support a multi-drone condition.

### Cost

Conditions 1–3 are one fusion run each (~4 min) plus one occupancy build each
(~17 min for 3 cameras, ~6 min for 1). Condition 0 needs no fusion. Roughly 1.5–2 h
for the full A1+A2 grid.
