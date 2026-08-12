# Paper-Faithful CoVOR-SLAM Reproduction

This note documents a reproduction of **CoVOR-SLAM (Lee et al., 2023,
"CoVOR-SLAM: Cooperative SLAM using Visual Odometry and Ranges for Multi-Robot
Systems")** on the MILUV `default_3_zigzag_0` sequence. The single goal is
**faithfulness to the paper**, not ATE on this dataset. Where our setup departs
from the paper, the departure is stated explicitly and justified below.

Sequence: `default_3_zigzag_0` (3 drones, LOS UWB). Front-end: ORB-SLAM3 mono
(infra1 IR cam, 848×480). Back-end fusion: GTSAM factor graph, Levenberg–Marquardt.
Reproduction script: [`scripts/faithful_covor.py`](scripts/faithful_covor.py).

---

## 1. The front-end fix: continuous *drifting* VO

The paper's per-robot front-end is a **drifting but continuous Visual Odometry**.
What CoVOR-SLAM replaces with UWB ranges is the **inter-agent** loop closure; each
robot's front-end producing a single continuous trajectory is a *premise*, not
something the ranges provide. Our earlier baselines sat at two wrong extremes:

- **Loop closing fully ON** — ORB-SLAM3 `MONOCULAR` runs local BA + loop closure +
  map merging + place recognition. This is drift-*corrected* full SLAM, stronger
  than the paper's assumed front-end, leaving no room for ranges to help.
- **`loopClosing: 0`** — in ORB-SLAM3 this single flag disables loop closure **and
  map merging** (both gated by `mbActiveLC`). Drone footage loses tracking often,
  so with merging off the map fragments into ~10 pieces that never rejoin; saved
  trajectories covered only 4–15 % of each flight → no full-sequence ATE, no
  inter-robot time overlap, no fusion.

### Fix — a `loopCorrection` flag in the ORB-SLAM3 fork

We added a `loopCorrection` yaml flag to the fork
(`github.com/gs25058/CR_RNE1`, branch `covor-loop-correction`), mirroring the
existing `loopClosing`/`activeLC` pattern:

- `include/LoopClosing.h` — new member `bool mbActiveLoopCorrection`.
- `src/System.cc` — reads `loopCorrection` from the settings file, passes it to
  the `LoopClosing` constructor.
- `src/LoopClosing.cc` — the loop-detection branch calls `CorrectLoop()` **only if
  `mbActiveLoopCorrection`**.

Effect with `loopClosing: 1 + loopCorrection: 0`: loops are still **detected** and
maps are still **merged** (trajectory stays continuous across tracking loss), but
the loop-closure **correction** (pose-graph optimisation + global BA) is skipped —
i.e. a **continuous, drifting VO**, exactly the paper's mono-VO baseline. Local BA
inside `LocalMapping` is kept, as it is part of the VO front-end.

**Coverage after the fix** (fraction of flight in the single continuous trajectory):
ifo001 **94 %** / ifo002 **93 %** / ifo003 **98 %** — the fragmentation from plain
`loopClosing: 0` is resolved.

---

## 2. Faithful ATE results

Per-robot ATE-RMSE (m) vs mocap ground truth. VO baseline aligned with scale
(Sim3, `with_scale=True`); fused trajectories are metric (`with_scale=False`).
Reproduce with `python scripts/faithful_covor.py` (rows → `results/diagnosis_results.csv`).

| config | ifo001 | ifo002 | ifo003 | mean |
|---|---|---|---|---|
| VO baseline (drifting, loop-corr OFF) | 0.167 | 0.782 | 0.606 | 0.518 |
| VO + inter only (robust OFF)          | 0.983 | 1.233 | 1.286 | 1.17  |
| VO + inter + anchor (robust OFF, **faithful**) | 0.600 | 0.453 | 0.467 | 0.507 |
| VO + inter + anchor (robust ON)       | 0.527 | 0.450 | 0.463 | **0.480** |

**The paper's core demonstration reproduces:** range fusion corrects VO drift on
the two drifting robots (ifo002 0.78→0.45, ifo003 0.61→0.46). Notes:

- **inter-only collapses** — without anchors there is no absolute frame; this is
  aggravated by drone tracking-loss map breaks (7–13 per robot) that the paper's
  ground rover did not suffer.
- **robust ON is marginally better** (0.480 vs 0.507). MILUV UWB has a heavy tail
  (p99 ≈ 0.9 m) that the paper's outdoor data lacked; this is a **data** difference,
  not a method one (see §3, robust kernel row).
- **ifo001 is hurt by fusion** (0.167 → 0.53). Its VO error is already below the
  ~0.22 m UWB noise floor, so adding ranges can only degrade it — consistent with
  the diagnosis rule "fusion helps a robot iff its VO error > ~0.22 m"
  (`DIAGNOSIS.md`). This is a property of this easy LOS sequence, not of the method.

---

## 3. Faithfulness checklist

Each element is marked **[match]** (as in the paper), **[justified detour]**
(mathematically/operationally equivalent workaround), or **[deviation]**.

| element | paper | ours | verdict |
|---|---|---|---|
| **Front-end** | drifting continuous VO, no inter-agent loop closure | ORB-SLAM3 mono, loop-correction OFF, merging kept; robots run independently so no inter-agent visual association | **match** |
| **State variables** | Sim(3), 7-DoF | Pose3 (metric) + scalar scale — GTSAM Python does not expose `Similarity3` retract/local; re-parameterisation is mathematically equivalent | **justified detour** |
| **Odometry factor** | Sim(3) relative pose between consecutive frames | Sim(3) relative pose (Pose3 + scale) between consecutive VO frames | **match** |
| **Inter-agent range factor** | yes | yes; analytic Jacobian ∂‖p−a‖/∂ξ = [0,0,0, uᵀR], unit-tested to 4e-16 vs GTSAM `RangeFactorPose3` | **match** |
| **Anchor range factor** | yes | yes (same factor form) | **match** |
| **Robust kernel** | none (relies on measurement redundancy) | **faithful config: OFF.** A Huber-ON variant is reported separately for contrast, motivated by MILUV's heavy UWB tail (p99 ≈ 0.9 m) | **match** (faithful config); deviation is opt-in and labelled |
| **Initial frame alignment** | ArUco markers | mocap Umeyama alignment | **justified detour** (both are just an initial alignment device) |
| **Antenna offset** | assumed removed by offline calibration | explicit moment-arm factor from MILUV `tags.yaml` lever arms | **match** (honours the paper's "pre-compensated" assumption) |
| **Prior** | φ¹_pri,k at each agent's first pose only | prior at each agent's first pose only (`prior_every=0`) | **match** |
| **Optimiser** | — (nonlinear LS) | GTSAM Levenberg–Marquardt | **match** |

### MILUV-specific extensions kept OFF for faithfulness

The following help on MILUV but have no counterpart in the paper, so the faithful
config disables them: **height (downward-laser) factor**, **online/const range
bias estimation**. They are available in `covor/fusion.py` (`use_height`,
`bias_mode`) for the separate MILUV-tuned study but are **not** part of this
reproduction.

---

## 4. Remaining faithfulness deltas (honest limitations)

- **VO covariance** — we use a fixed per-measurement σ (with a small floor) rather
  than a covariance propagated from the VO front-end as the paper does.
- **Robust kernel** — the paper uses none; our faithful config matches, but the
  MILUV data's heavy tail means the Huber-ON variant is the practically better one
  here. Reported side by side rather than hidden.
- **Init alignment** — mocap Umeyama vs the paper's ArUco markers.
- **Sim(3) parameterisation** — Pose3 + scale rather than a native `Similarity3`
  manifold (equivalent, but not bit-identical to a true Sim(3) retraction).

These are documented so the reproduction can be read exactly for what it is: a
faithful re-implementation of CoVOR-SLAM's method, evaluated on a MILUV LOS drone
sequence whose difficulty (easy, low-drift ifo001) differs from the paper's data.
