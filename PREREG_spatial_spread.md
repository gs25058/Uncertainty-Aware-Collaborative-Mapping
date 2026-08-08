# Pre-registration — spatial spreading of pose position uncertainty

**Committed BEFORE any implementation.** The scaling result is already known, so
forking-paths risk is real; everything that could be chosen after seeing an outcome
is fixed here first. One test, no variants.

Date: 2026-08-08. Sequence `default_3_zigzag_0`. Fixed coverage (ifo001 camera only).

---

## 1. Why this alternative, derived from mechanism not from data

Two established results:

- **tr(Σ) predicts where the map goes wrong** — cells whose evidence came from the
  least certain poses are 1.6–2.4× more likely to be falsely called free
  (OCCUPANCY_PIPELINE.md, per-cell deciles).
- **Weighting log-odds by tr(Σ) makes false-free worse**, +0.05…+0.11 pp on the
  anchor-free conditions, with the D/E negative controls at 0.000/+0.001.

Not a contradiction: the signal exists and the current encoding cannot use it.

**Diagnosis — a category error.** Pose covariance Σ is a *position* uncertainty
("the obstacle is there, I am unsure **where**"). Multiplying log-odds by `w` encodes
an *existence* uncertainty ("I am unsure **whether** it is there"). Those are
different statements, and the pipeline currently writes the first as the second.
Concretely, `w_pose` scales `l_occ = +0.85` and `l_free = −0.40` by the same factor,
so a cell whose occupied evidence comes from a down-weighted frame and whose free
evidence comes from up-weighted frames tips to **free** — the §4.6 safety asymmetry
is defeated by the weight ratio.

This also explains why the depth term *does* work: σ_Z is nominally a position
uncertainty too, but it is *correlated with existence* uncertainty — large σ_Z means
long range, where stereo genuinely fabricates points that are not there. Scaling is
the right tool there by coincidence. Pose error fabricates nothing; every observation
is real and only misplaced, so scaling is the wrong tool.

**The alternative.** If the pose is uncertain by √tr(Σ), the whole ray is uncertain
by that much. The correct handling is not to *weaken* the evidence but to **spread it
over that range**. An uncertain observation then becomes *smeared* evidence rather
than *weak* evidence: it neither carves free space sharply nor pins an obstacle to
one cell, so the safety asymmetry is not defeated.

---

## 2. Design — every free parameter fixed here

| choice | value | why it is not a tuning knob |
|---|---|---|
| spread σ | **σ = √(tr Σ_pos)** of the contributing frame | comes straight from the measurement; no coefficient multiplies it |
| spread axis | along the ray direction | that is the direction the endpoint slides in |
| profile | Gaussian, truncated at **±2σ** | 2σ ≈ 95 %; one standard constant, used for BOTH the spread support and the free-carving stop |
| mass | **conserved**: the distributed evidence sums to exactly `l_occ` | this is the decisive difference from scaling — evidence is *relocated*, never reduced |
| free evidence | carve free only up to **d − 2σ**; the interval [d − 2σ, d + 2σ] receives spread occupied mass and **no** free evidence | inside the uncertain band we must not assert free |
| `w_pose` | **≡ 1** (off) | spreading handles pose uncertainty; scaling it too would double-count |
| `w_depth` | kept, unchanged | its effect is established; leaving it on isolates the pose treatment |
| α | **unused** in arm S | the width comes from tr(Σ) directly — one less free parameter, and that is a point in the method's favour |
| β, l_occ, l_free, tau_occ, tau_free, resolution | unchanged | not touched |

σ_Z is **not** folded into the spread width: depth uncertainty stays with `w_depth`
so each source is handled exactly once, and S vs arm2 differs *only* in the pose
treatment.

---

## 3. Arms (each of conditions A–E)

| arm | treatment |
|---|---|
| **S** | spatial spread (new) + `w_depth`, `w_pose ≡ 1` |
| 2 | `w_depth` only, no spread (baseline for the primary comparison) |
| 1 | full scaling `w_pose · w_depth` (the known-failing encoding, kept as contrast) |

Primary comparison **S vs arm 2**. Arm 1 is reported alongside as
"scaling vs spreading" on the same axes.

## 4. Metrics

- **Primary: false-free rate** (the metric §4.9 names).
- Secondary, reported separately, never merged into IoU alone:
  precision, recall, occupied total vs GT, object voxels.
- Condition A's low IoU is refusal-to-map, not wrongness; precision/recall keep that
  visible.

## 5. Predictions

- **P1** Δ false-free (S − arm2) **< 0** on A, B, C.
- **P2** Δ false-free ≈ 0 on **D and E** (negative controls: tr(Σ) spans only
  0.00098–0.00235, so √tr(Σ) ≈ 0.03–0.05 m ≈ half a voxel — the spread is
  sub-resolution and can do nothing).
- **P3** Effect size orders like √tr(Σ): A > B > C ≫ D ≈ E.
- **P4** Recall does **not** degrade as much as scaling did (A: −0.145), because
  evidence is relocated rather than removed.

## 6. Falsifier

If **P2 fails** — a material effect appears on D or E — the implementation or the
measurement is wrong. In that case report the defect and make **no** claim about
spreading.

## 7. Decision rule

- **P1 and P2 both hold** → the alternative stands. The central claim is recovered:
  pose uncertainty helps the map when encoded as position uncertainty. Report as such.
- **P1 fails** → **record as NOT ESTABLISHED and stop.** Report scaling-failed *and*
  spreading-failed together. **Do not try a third variant.**
- Large recall loss → state its size and report as a trade-off.

## 8. Single-shot rule

The formula and parameters above are frozen. No re-running with a different width,
profile, truncation, or free-carving rule. The previous (scaling) result is preserved
as a control and will not be softened or deleted.

## 9. Verification before trusting any number

Correctness of the spread is checked against a **reference implementation (plain
loop) by comparing per-cell log-odds values**, not voxel counts — a count-only check
previously let a real ray-traversal bug through (HANDOFF pitfall 4).
