"""Scoring a built map against the mesh-derived GT (Parts C and D).

Every metric the task brief lists is computed here, and they are computed
TOGETHER: ``score`` returns them as one record so none can be quoted alone. In
particular false-free rate is meaningless by itself -- declaring everything
occupied drives it to zero (RESULTS_SUMMARY §9-1) -- so it always travels with
the free-cell count and with precision/recall.

DOMAIN. A cell is scored only if
    * the GT knows what it is  (label OCC or FREE, never UNKNOWN), and
    * some beam touched it     (the observability mask, built once per coverage
                                mode from GT poses -- so every condition inside
                                one coverage mode is scored on the same cells).
Anything else is excluded rather than counted as free.
"""
import numpy as np
from scipy import ndimage

from . import mesh_gt as MG


def _dense(idx, shape, ijk_min):
    m = np.zeros(shape, bool)
    if len(idx) == 0:
        return m
    i = np.asarray(idx) - ijk_min
    ok = np.all((i >= 0) & (i < np.array(shape)), axis=1)
    i = i[ok]
    m[i[:, 0], i[:, 1], i[:, 2]] = True
    return m


def dilate1(m, n=1):
    """n-voxel (26-neighbourhood) dilation -- the '@n vox' tolerance."""
    if n <= 0:
        return m
    return ndimage.binary_dilation(m, structure=np.ones((3, 3, 3), bool),
                                   iterations=int(n))


def map_masks(builder, res, lab, ijk_min):
    """(occ, free) dense boolean grids from a finished builder.

    Must be called BEFORE write_bt: classify_points enforces that (the export
    collapses log-odds to max-likelihood and prunes).
    """
    occ, free = builder.classify_points()
    return (_dense(MG.index_of(occ, res), lab.shape, ijk_min),
            _dense(MG.index_of(free, res), lab.shape, ijk_min))


def logodds_grid(builder, res, shape, ijk_min):
    """Dense per-cell log-odds. Like classify_points, must run BEFORE write_bt."""
    L = np.full(shape, np.nan, np.float32)
    builder.tree.updateInnerOccupancy()
    for it in builder.tree.begin_leafs():
        i = MG.index_of(np.asarray(it.getCoordinate(), float), res) - ijk_min
        if np.all(i >= 0) and np.all(i < np.array(shape)):
            L[i[0], i[1], i[2]] = it.getValue()
    return L


def score(M_occ, M_free, lab, ijk_min, res, obs, gt_occ_total=None):
    """All the brief's map metrics for one map, as one dict.

    M_occ, M_free : dense bool, what the map declares.
    obs           : dense bool observability mask (see module docstring).
    """
    G_occ = lab == MG.OCC
    G_free = lab == MG.FREE
    known = G_occ | G_free
    dom = known & obs

    g = G_occ & dom
    m = M_occ & dom
    tp = int((m & g).sum())
    fp = int((m & G_free & dom).sum())
    fn = int((g & ~M_occ).sum())

    # tolerant (@1 voxel) counts: a predicted cell is correct if it is within one
    # voxel of GT occupied; a GT cell is covered if within one voxel of a
    # predicted one. At tolerance 0 this reduces to the plain IoU.
    dg, dm = dilate1(G_occ), dilate1(M_occ)
    tp_t = int((m & dg).sum())
    fp_t = int((m & ~dg).sum())
    fn_t = int((g & ~dm).sum())

    free_on_occ = int((M_free & g).sum())
    n_dom = int(dom.sum())
    n_known_map = int((m | (M_free & dom)).sum())
    out = dict(
        precision=tp / max(tp + fp, 1),
        recall=tp / max(int(g.sum()), 1),
        iou=tp / max(tp + fp + fn, 1),
        iou_1vox=tp_t / max(tp_t + fp_t + fn_t, 1),
        precision_1vox=tp_t / max(int(m.sum()), 1),
        recall_1vox=1.0 - fn_t / max(int(g.sum()), 1),
        false_free_rate=free_on_occ / max(int(g.sum()), 1),
        n_free=int((M_free & dom).sum()),
        n_occupied=int(m.sum()),
        occ_over_gt=int(m.sum()) / max(int(g.sum()), 1),
        n_unknown=n_dom - n_known_map,
        n_domain=n_dom,
        n_gt_occ_domain=int(g.sum()),
        n_gt_free_domain=int((G_free & dom).sum()),
        tp=tp, fp=fp, fn=fn, n_false_free=free_on_occ,
    )
    tot = int(G_occ.sum()) if gt_occ_total is None else int(gt_occ_total)
    out["coverage"] = int((G_occ & obs).sum()) / max(tot, 1)

    # Tolerance sweep. At res = 0.10 m a zero-thickness surface cannot be
    # represented exactly: the GT marks every cell the surface TOUCHES (often
    # two, where it crosses a voxel boundary) while a beam terminates in exactly
    # one of them, and a beam arriving at grazing incidence passes through many
    # cells of a surface before hitting it. Both are discretisation, not
    # estimation error, and both move with tolerance -- so the whole curve is
    # reported instead of one threshold (RESULTS_SUMMARY §9-4).
    for tol in (0, 1, 2):
        dgt = dilate1(G_occ, tol)
        dmt = dilate1(M_occ, tol)
        out["precision_tol%d" % tol] = int((m & dgt).sum()) / max(int(m.sum()), 1)
        out["recall_tol%d" % tol] = 1.0 - int((g & ~dmt).sum()) / max(int(g.sum()), 1)
        out["false_free_tol%d" % tol] = int((M_free & g & ~dmt).sum()) / \
            max(int(g.sum()), 1)
    return out


# ---------------------------------------------------------------------------
# trajectory metrics
# ---------------------------------------------------------------------------
def ate(p_est, p_gt, yaw_only=True):
    """RMSE after the 4-DoF (yaw + translation) alignment the project uses.

    Both frames are gravity-aligned, so fitting a full SO(3) would absorb real
    roll/pitch error into the alignment (covor.fusion.umeyama_yaw).
    """
    from covor.fusion import umeyama_yaw, umeyama_rigid
    R, t = (umeyama_yaw if yaw_only else umeyama_rigid)(p_est, p_gt)
    e = np.linalg.norm(p_est @ R.T + t - p_gt, axis=1)
    return dict(ate_rmse=float(np.sqrt((e ** 2).mean())),
                ate_median=float(np.median(e)), ate_max=float(e.max()))


def interp_gt(gt, robot, t):
    g = gt[robot]
    tq = np.clip(np.asarray(t, float), g["t"][0], g["t"][-1])
    return np.stack([np.interp(tq, g["t"], g["p"][:, i]) for i in range(3)], 1)
