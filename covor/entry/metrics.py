"""Scoring an entry map against the GT entry map (DESIGN_entry_map.md §6).

``score`` returns every metric the design lists as ONE record, for the same
reason covor/synth/metrics.py does: false-passable rate cannot be quoted alone.
Declare the whole floor blocked and it is 0.000 -- the 2D form of the trap
RESULTS_SUMMARY §9-1 describes, which §8.3 walked into once already. So the rate
never travels without the passable count, the recall, and the raw numerator and
denominator it came from.

DOMAIN. Every cell of the grid is scored. There is deliberately no observability
mask here, unlike the 3D metrics: a column no beam reached comes out UNKNOWN,
unknown is an obstacle, and an obstacle is not passable -- so an unobserved
column can only ever cost recall, never create a false-passable cell. Masking
them out would remove the honest penalty for not having looked.

GT SIDE. The GT entry map is built by running THIS pipeline on gt_voxel.npz with
the same cfg and k_sigma = 0, so pipeline and ground truth differ only in the 3D
grid handed in.
"""
import numpy as np

from .config import UNKNOWN, FREE, OCCUPIED, BLOCKED, NARROW, WALK

GRADES = (BLOCKED, NARROW, WALK)
GRADE_NAMES = {BLOCKED: "blocked", NARROW: "narrow", WALK: "walk"}


def _f(num, den):
    """num/den, or NaN when the denominator is empty.

    NaN rather than 0: an empty denominator means the question was not asked on
    this map, which is a different statement from "the answer is zero", and only
    one of the two should ever reach a table.
    """
    den = int(den)
    return float(num) / den if den else float("nan")


def score(pred, gt, band):
    """All of DESIGN §6's map metrics for one band, as one dict.

    pred, gt: dicts (or loaded .npz) from build_entry_map.build_entry_grid.
    """
    P = np.asarray(pred["%s_passable" % band], bool)
    G = np.asarray(gt["%s_passable" % band], bool)
    Rp = np.asarray(pred["%s_reachable" % band], bool)
    Rg = np.asarray(gt["%s_reachable" % band], bool)
    Lp = np.asarray(pred["%s_label" % band])
    Lg = np.asarray(gt["%s_label" % band])
    Wp = np.asarray(pred["%s_width_class" % band])
    Wg = np.asarray(gt["%s_width_class" % band])
    if P.shape != G.shape:
        raise ValueError("pred %s and gt %s are different grids" % (P.shape, G.shape))

    fp = P & ~G                      # the map says walk here, the truth says no
    tp = P & G
    out = dict(
        band=band,
        n_cells=int(P.size),
        # --- safety ------------------------------------------------------
        false_passable_rate=_f(fp.sum(), (~G).sum()),
        n_false_passable=int(fp.sum()),
        n_gt_not_passable=int((~G).sum()),
        # where the false-passable cells came from: a wall the map missed, or a
        # place the truth never observed. Different failures, different fixes.
        n_false_passable_on_gt_occupied=int((fp & (Lg == OCCUPIED)).sum()),
        n_false_passable_on_gt_unknown=int((fp & (Lg == UNKNOWN)).sum()),
        n_false_passable_on_gt_free=int((fp & (Lg == FREE)).sum()),
        # --- usefulness ---------------------------------------------------
        passable_recall=_f(tp.sum(), G.sum()),
        n_true_passable=int(tp.sum()),
        n_gt_passable=int(G.sum()),
        n_pred_passable=int(P.sum()),
        # --- completeness -------------------------------------------------
        reachable_iou=_f((Rp & Rg).sum(), (Rp | Rg).sum()),
        n_reachable_inter=int((Rp & Rg).sum()),
        n_reachable_union=int((Rp | Rg).sum()),
        n_pred_reachable=int(Rp.sum()),
        n_gt_reachable=int(Rg.sum()),
        # --- the label composition, which every rate above has to be read
        #     against (RESULTS_SUMMARY §9-1)
        n_pred_unknown=int((Lp == UNKNOWN).sum()),
        n_pred_occupied=int((Lp == OCCUPIED).sum()),
        n_pred_free=int((Lp == FREE).sum()),
        n_gt_unknown=int((Lg == UNKNOWN).sum()),
        n_gt_occupied=int((Lg == OCCUPIED).sum()),
        n_gt_free=int((Lg == FREE).sum()),
        # --- absolute accuracy: needs a tape measure, so synth cannot have it
        door_width_error=float("nan"),
    )

    # --- width grade confusion ------------------------------------------
    # Reported over ALL cells (complete, and free of any choice of subset) and
    # again over GT-free cells only, where the grade question actually arises --
    # the all-cells matrix is dominated by blocked/blocked agreement on the two
    # thirds of the grid that is wall and outside.
    agree_all = Wp == Wg
    gtfree = Lg == FREE
    out["grade_agreement"] = _f(agree_all.sum(), Wp.size)
    out["grade_agreement_on_gt_free"] = _f((agree_all & gtfree).sum(), gtfree.sum())
    for g in GRADES:
        for p in GRADES:
            out["cm_gt_%s_pred_%s" % (GRADE_NAMES[g], GRADE_NAMES[p])] = int(
                ((Wg == g) & (Wp == p)).sum())
            out["cmfree_gt_%s_pred_%s" % (GRADE_NAMES[g], GRADE_NAMES[p])] = int(
                ((Wg == g) & (Wp == p) & gtfree).sum())
    return out


def confusion_table(rec, on_gt_free=True):
    """The 3x3 grade confusion as text, rows = GT, columns = prediction."""
    pre = "cmfree" if on_gt_free else "cm"
    lines = ["%-10s %8s %8s %8s" % ("GT \\ pred", "blocked", "narrow", "walk")]
    for g in GRADES:
        lines.append("%-10s %8d %8d %8d" % (
            GRADE_NAMES[g],
            *[rec["%s_gt_%s_pred_%s" % (pre, GRADE_NAMES[g], GRADE_NAMES[p])]
              for p in GRADES]))
    return "\n".join(lines)


FIELDS_HEAD = ("band", "w", "k_sigma", "false_passable_rate", "n_false_passable",
               "n_gt_not_passable", "passable_recall", "n_true_passable",
               "n_gt_passable", "n_pred_passable", "reachable_iou",
               "n_pred_unknown", "n_gt_unknown")


def row_text(rec):
    """One line carrying the rate together with everything §9-1 requires."""
    return ("%-6s w=%.2f k=%.0f | ff-pass %.4f (%d/%d) | recall %.4f (%d/%d) | "
            "pred passable %d | reach IoU %.4f | unknown pred %d gt %d"
            % (rec["band"], rec.get("w", float("nan")), rec.get("k_sigma", 0),
               rec["false_passable_rate"], rec["n_false_passable"],
               rec["n_gt_not_passable"], rec["passable_recall"],
               rec["n_true_passable"], rec["n_gt_passable"],
               rec["n_pred_passable"], rec["reachable_iou"],
               rec["n_pred_unknown"], rec["n_gt_unknown"]))
