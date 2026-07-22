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
  space is asserted conservatively, occupied readily ⇒ fewer false-free.
- Weighting toggles off (`OccCfg.weighted=False` ⇒ `w≡1`) to recover a standard
  OctoMap update, making the ablation structural.

`unknown` is never conflated with `free`: pixels with no valid disparity cast no
ray (`Z=NaN`), and cells below the free threshold stay unknown (§4.8).

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
```

## Faithfulness checklist (proposal §4 / §7.3)

| element | proposal | ours | verdict |
|---|---|---|---|
| Input pose | fused SE(3) `T^k_n` (UWB-aligned, not single-drone VO) | `covor.fusion` fused `Pose3` | **match** |
| **Σ forwarded** | §2.5 ★ registration covariance passed to mapping, not discarded | `gtsam.Marginals` → `tr(Σ_pos)` per keyframe | **match** |
| Depth | stereo SGBM, `Z=f·B/d`, 5 m cutoff, invalid⇒unknown | `StereoDepth` (MILUV infra1/infra2, B=0.05 m) | **match** |
| Depth uncertainty | `σ_Z=Z²/(f·B)·Δd` | same | **match** |
| Back-projection | `P_cam=[(u−cᵤ)Z/f, …]`, 4×4 downsample | same | **match** |
| World transform | `P_world=R·P_cam+t` (§3③) | same (pose = infra1 camera frame) | **match** |
| Ray casting | free = pass-through, occupied = endpoint, unknown behind | vectorized voxel traversal (§4.4 Bresenham-equivalent) | **match** |
| **Weighted log-odds** | §4.5 ★ `w=exp(−trΣ/α)exp(−σ_Z²/β)`, `updateNode` direct | same, `updateNode(cell, w·l_meas)` | **match** |
| **Safety asymmetry** | §4.6 ★ `|l_free|<|l_occ|` | `l_free=−0.40 < l_occ=0.85` | **match** |
| Multi-drone accumulation | §4.7 log-odds additive into one tree | same builder, all robots | **match** |
| Classification | occ / free / unknown, unknown≠free | `tau_occ`, `tau_free` thresholds | **match** |
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

## What this stage shows

On `default_3_zigzag_0`, a single drone yields a clean navigation occupancy map —
free interior, occupied walls, unknown beyond — from CoVOR fused poses + stereo.
The uncertainty weighting **changes the map in the safety direction**: in the
`z=0.8 m` slice, ~6–7 % of the cells that the standard (uniform) map declares
*free* are held *unknown/occupied* by the weighted map, and those suppressed cells
are spatially concentrated in the far / high-drift regions — exactly the
false-free that §4.6 targets. Quantitative IoU / false-free evaluation against GT
is the simulator stage (proposal §4.2, Phase 1-6/3-3), not MILUV.
