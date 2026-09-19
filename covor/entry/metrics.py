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


# ---------------------------------------------------------------------------
# route metrics (DESIGN §6, last two rows)
# ---------------------------------------------------------------------------
def score_route(rec, gt, band, cfg):
    """Safety and efficiency of one planned route against the GT entry map.

    route_validity  fraction of the route's cells that the GT also calls
                    passable. DESIGN §6 words it as "every cell GT passable", so
                    the pass/fail form is kept too -- but the fraction is what
                    says how badly a failing route fails.
    length_ratio    the route's length over the GT's shortest route to the same
                    goal. Computed with pure distance on the GT side (no
                    clearance weighting): the denominator should be the best a
                    perfect map could do, not the best it would CHOOSE to do.
                    NaN when the GT cannot reach that goal at all, which is a
                    different statement from a bad ratio.
    """
    from .route import geodesic
    if not rec.get("reachable"):
        return dict(route_validity=float("nan"), route_all_valid=False,
                    route_length_ratio=float("nan"), n_route_cells=0,
                    n_route_cells_not_gt_passable=0, gt_can_reach_goal=False)
    cells = np.asarray(rec["cells"])
    G = np.asarray(gt["%s_passable" % band], bool)
    ok = G[cells[:, 0], cells[:, 1]]
    start = tuple(int(v) for v in np.asarray(gt["entry_ij"]))
    goal = tuple(int(v) for v in rec["goal_ij"])
    d = geodesic(G, start)
    gt_len = float(d[goal]) * cfg.res
    return dict(
        route_validity=_f(ok.sum(), len(ok)),
        route_all_valid=bool(ok.all()),
        n_route_cells=int(len(ok)),
        n_route_cells_not_gt_passable=int((~ok).sum()),
        gt_can_reach_goal=bool(np.isfinite(d[goal])),
        gt_shortest_m=gt_len if np.isfinite(d[goal]) else float("nan"),
        route_length_ratio=(rec["length_m"] / gt_len
                            if np.isfinite(d[goal]) and gt_len > 0
                            else float("nan")),
        route_min_width_m=rec["min_width_m"],
        route_length_m=rec["length_m"],
        route_n_unknown_adjacent=rec["n_unknown_adjacent"],
    )


def workspace_scores(pred, gt, band, cfg):
    """Reachable-floor agreement in WORKSPACE as well as configuration space.

    Ported from uacm/entry_map's evaluate_against_gt. Both numbers are returned
    together and named apart, because they answer different questions and are
    easy to quote as if they were the same one:

      reachable_iou            centres: the set a planner may put the body in
      reachable_iou_workspace  floor: the set the body actually sweeps

    The workspace figure is always the kinder of the two -- dilating both sides
    by the same disk merges the erosion bands that hug every wall. It is not a
    better measurement, it is a different one, and RESULTS_entry.md quotes both.
    """
    from .inflate import swept_workspace
    Rp = np.asarray(pred["%s_reachable" % band], bool)
    Rg = np.asarray(gt["%s_reachable" % band], bool)
    # the sweep is clipped to each map's OWN free cells. A shoulder passing over
    # a wall cell is not floor, and a shoulder passing over an unobserved cell is
    # not floor either -- it is the same unknown-is-not-free rule as everywhere
    # else. uacm's version does not clip; clipping keeps the number and the
    # picture (render.draw_band) describing the same set.
    Wp = swept_workspace(Rp, cfg) & (np.asarray(pred["%s_label" % band]) == FREE)
    Wg = swept_workspace(Rg, cfg) & (np.asarray(gt["%s_label" % band]) == FREE)
    a = cfg.res ** 2
    return dict(
        reachable_iou_workspace=_f((Wp & Wg).sum(), (Wp | Wg).sum()),
        n_pred_reachable_workspace=int(Wp.sum()),
        n_gt_reachable_workspace=int(Wg.sum()),
        area_pred_reachable_m2=float(Rp.sum() * a),
        area_pred_reachable_workspace_m2=float(Wp.sum() * a),
        area_gt_reachable_m2=float(Rg.sum() * a),
        area_gt_reachable_workspace_m2=float(Wg.sum() * a),
    )
