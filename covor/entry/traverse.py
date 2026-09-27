"""Human-like traversability on the 3D map (DESIGN_traversability.md).

The band model (project.py -> inflate.py -> reach.py) flattens the building onto
ONE floor row and asks whether a fixed slab above it is empty. That fails the
way a person does not: a floor that spreads over two voxel rows puts the floor
itself inside the slab, a 5 cm bump is a wall, a stair is a wall, and a low
beam cannot be ducked under. This module asks the question a walker asks, cell
by cell:

  1. SUPPORT. Where can a foot go?  A support is an occupied voxel with a free
     voxel directly above it: floor, a stair tread, a box top. A column may hold
     several (the floor under a desk and the desk top). No global floor row.
  2. BODY. Standing on support k, does a body fit?  The torso occupies rows
     [k + 1 + S, k + 1 + n_h) over a disc of the body radius, where S is the
     step height in rows and n_h the posture height. Anything lower than the
     step, beside the feet, is stepped over or onto -- that is what a step
     height means. The centre column must be free from k + 1 up. Unknown
     counts as blocked everywhere (DESIGN §3-2 is kept).
  3. POSTURES. stand (1.90 m), stoop (1.40 m), crawl (0.90 m), each with the
     normal width (0.70 m) or sideways (0.45 m). A node takes the cheapest
     posture/width that fits.
  4. STEPS. A move to an 8-neighbour support k' is allowed when |k' - k| <= S.
     Stairs are a chain of such moves; a 0.3 m ledge is not.
  5. GAPS (amendment, DESIGN_traversability.md §7). A map rarely observes
     every floor voxel: rays graze the floor and carve it free, or never reach
     it. A walker does not stop at an unseen patch of floor; they step across
     it. A column whose body space is free at the current support level but
     which has no observed support there is a GAP cell; a run of at most
     `gap` metres (0.5 m, a cautious step) of gap cells may be crossed at the
     level of the last support. A wider unseen region -- a stair void -- still
     blocks.
  6. ROUTE. Dijkstra from the entry node. Cost = step length x posture/width
     factor + a per-row charge for every change of support height.

Everything is a function of the 3D label grid and TravCfg; nothing is tuned
against a result (the values and their sources are in TravCfg).
"""
import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import UNKNOWN, FREE, OCCUPIED

# posture codes, cheapest first
STAND, STOOP, CRAWL = 0, 1, 2
POSTURE_NAMES = {STAND: "stand", STOOP: "stoop", CRAWL: "crawl"}
NORMAL, SIDEWAYS = 0, 1


@dataclass(frozen=True)
class TravCfg:
    res: float = 0.10
    # body (DESIGN_entry_map §4 values kept; stoop is new, anthropometric)
    w: float = 0.70              # shoulder width with SCBA -> radius w/2
    w_side: float = 0.45         # sideways pass (= old narrow-grade boundary x2)
    h_stand: float = 1.90        # = H_walk
    h_stoop: float = 1.40        # bent-forward walking height
    h_crawl: float = 0.90        # = H_crawl
    step: float = 0.18           # building-code maximum stair riser
    gap: float = 0.50            # unseen floor crossed in one cautious step (§7)
    # route costs (only shape WHICH route is chosen, never whether one exists)
    f_stoop: float = 1.5
    f_crawl: float = 3.0
    f_side: float = 2.0
    c_rise: float = 1.0          # extra metres of cost per metre of height change

    @property
    def step_rows(self):
        return int(round(self.step / self.res))

    def rows(self, h):
        return int(round(h / self.res))

    @property
    def gap_cells(self):
        return int(round(self.gap / self.res))


def supports(lab):
    """Boolean (nx, ny, nz): occupied voxel with a FREE voxel directly above."""
    lab = np.asarray(lab)
    s = np.zeros(lab.shape, bool)
    s[:, :, :-1] = (lab[:, :, :-1] == OCCUPIED) & (lab[:, :, 1:] == FREE)
    return s


def _clear2d(blocked, res):
    """Surface clearance [m] from each cell centre (inflate.py's convention)."""
    d = ndimage.distance_transform_edt(~blocked)
    return np.maximum(d - 0.5, 0.0) * res


def node_modes(lab, cfg, air=False):
    """For every support voxel, the cheapest (posture, width) that fits.

    Returns (S (nx,ny,nz) bool supports, posture (nx,ny,nz) int8 with -1 = no
    fit, width (nx,ny,nz) int8). With air=True the posture/width are computed
    for EVERY column at every support level, support or not -- "would a body
    standing at level k here fit" -- which is what a gap cell needs.
    """
    lab = np.asarray(lab)
    nx, ny, nz = lab.shape
    S = supports(lab)
    notfree = lab != FREE
    posture = np.full(lab.shape, -1, np.int8)
    width = np.full(lab.shape, -1, np.int8)
    st = cfg.step_rows
    heights = ((STAND, cfg.rows(cfg.h_stand)), (STOOP, cfg.rows(cfg.h_stoop)),
               (CRAWL, cfg.rows(cfg.h_crawl)))
    ks = np.unique(np.nonzero(S)[2])
    for k in ks:
        col = np.ones((nx, ny), bool) if air else S[:, :, k]
        for p, nh in heights:
            top = k + 1 + nh
            if top > nz:
                continue
            # centre column: free all the way from k+1 to the head
            centre_ok = ~notfree[:, :, k + 1:top].any(axis=2)
            # torso slab over the body disc: rows above the step height
            lo = k + 1 + st
            slab = notfree[:, :, lo:top].any(axis=2) if lo < top else np.zeros((nx, ny), bool)
            clear = _clear2d(slab, cfg.res)
            for wmode, rad in ((NORMAL, cfg.w / 2), (SIDEWAYS, cfg.w_side / 2)):
                ok = col & centre_ok & (clear >= rad - 1e-9) & (posture[:, :, k] < 0)
                posture[:, :, k][ok] = p
                width[:, :, k][ok] = wmode
    return S, posture, width


def _factor(p, wmode, cfg):
    f = {STAND: 1.0, STOOP: cfg.f_stoop, CRAWL: cfg.f_crawl}[int(p)]
    return f * (cfg.f_side if wmode == SIDEWAYS else 1.0)


def entry_node(posture, xy, ijk_min, res, snap_m=None):
    """(i, j, k) of the feasible node nearest the xy entry point (lowest k in a
    column); within snap_m of it, or None."""
    feas = posture >= 0
    cols = feas.any(axis=2)
    if not cols.any():
        return None
    i0 = int(np.floor(xy[0] / res)) - int(ijk_min[0])
    j0 = int(np.floor(xy[1] / res)) - int(ijk_min[1])
    ii, jj = np.nonzero(cols)
    d = np.hypot(ii - i0, jj - j0) * res
    n = int(np.argmin(d))
    if snap_m is not None and d[n] > snap_m + 1e-9:
        return None
    i, j = int(ii[n]), int(jj[n])
    k = int(np.nonzero(feas[i, j])[0][0])
    return (i, j, k), float(d[n])


def traverse(lab, cfg, entry_xy, ijk_min, snap_m=1.5):
    """Dijkstra over support nodes from the entry.

    Returns a dict of 2D arrays (per column, the best reached node):
      reach (bool), cost (m, inf if unreached), posture (-1 unreached),
      width, support_row (-1), feasible (any node fits, reached or not),
      feasible_stand (a node fits standing, normal width)
    plus the entry node and the snap distance.
    """
    S, P, W = node_modes(lab, cfg)
    _, PA, WA = node_modes(lab, cfg, air=True)
    lab = np.asarray(lab)
    # a gap cell at level k: body fits at level k, no observed support there,
    # and the voxel itself is not solid (an occupied voxel with unknown above is
    # not a floor we have seen, but it is not a gap in the floor either)
    gap3 = (PA >= 0) & ~S & (lab != OCCUPIED)
    G = cfg.gap_cells
    nx, ny, nz = S.shape
    feas3 = P >= 0
    out = dict(feasible=feas3.any(axis=2),
               feasible_stand=((P == STAND) & (W == NORMAL)).any(axis=2))
    e = entry_node(P, entry_xy, ijk_min, cfg.res, snap_m)
    cost = np.full((nx, ny), np.inf)
    post = np.full((nx, ny), -1, np.int8)
    wid = np.full((nx, ny), -1, np.int8)
    sup = np.full((nx, ny), -1, np.int16)
    lvl = np.full((nx, ny), -1, np.int16)       # support row, or the row a gap is crossed at
    out.update(entry=None, entry_snapped_m=None)
    if e is None:
        out.update(reach=np.zeros((nx, ny), bool), cost=cost, posture=post,
                   width=wid, support_row=sup, level_row=sup.copy(),
                   gap_only=np.zeros((nx, ny), bool))
        return out
    start, snapped = e
    st = cfg.step_rows
    dist = {start + (0,): 0.0}
    pq = [(0.0, start + (0,))]
    gapreach = np.zeros((nx, ny), bool)
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]
    while pq:
        d0, (i, j, k, g) = heapq.heappop(pq)
        if d0 > dist.get((i, j, k, g), np.inf):
            continue
        if g == 0:
            pu, wu = P[i, j, k], W[i, j, k]
        else:
            pu, wu = PA[i, j, k], WA[i, j, k]
            gapreach[i, j] = True
        if d0 < cost[i, j]:
            cost[i, j] = d0
            post[i, j] = pu
            wid[i, j] = wu
            sup[i, j] = k if g == 0 else -1
            lvl[i, j] = k
        fu = _factor(pu, wu, cfg)
        for di, dj, ln in steps:
            a, b = i + di, j + dj
            if not (0 <= a < nx and 0 <= b < ny):
                continue
            # step across unseen floor at the current level
            if g < G and gap3[a, b, k] and not (di and dj and not (
                    (feas3[i, b, k] or gap3[i, b, k]) and (feas3[a, j, k] or gap3[a, j, k]))):
                fv = _factor(PA[a, b, k], WA[a, b, k], cfg)
                nd = d0 + cfg.res * ln * max(fu, fv)
                if nd < dist.get((a, b, k, g + 1), np.inf):
                    dist[(a, b, k, g + 1)] = nd
                    heapq.heappush(pq, (nd, (a, b, k, g + 1)))
            for kk in range(max(k - st, 0), min(k + st, nz - 1) + 1):
                if P[a, b, kk] < 0:
                    continue
                if di and dj:
                    # no corner cutting through a cell with no node at this level
                    lo_, hi_ = max(kk - st, 0), kk + st + 1
                    if not ((feas3[i, b, lo_:hi_] | gap3[i, b, lo_:hi_]).any()
                            and (feas3[a, j, lo_:hi_] | gap3[a, j, lo_:hi_]).any()):
                        continue
                fv = _factor(P[a, b, kk], W[a, b, kk], cfg)
                nd = (d0 + cfg.res * ln * max(fu, fv)
                      + cfg.c_rise * abs(kk - k) * cfg.res)
                if nd < dist.get((a, b, kk, 0), np.inf):
                    dist[(a, b, kk, 0)] = nd
                    heapq.heappush(pq, (nd, (a, b, kk, 0)))
    reach = np.isfinite(cost)
    out.update(reach=reach, cost=cost, posture=post, width=wid,
               support_row=sup, level_row=lvl, entry=start, entry_snapped_m=snapped,
               gap_only=reach & (sup < 0))
    return out


def score(tm, tg):
    """Map traversal `tm` vs GT traversal `tg` (both from traverse())."""
    Rm, Rg = tm["reach"], tg["reach"]
    Fg = tg["feasible"]
    inter = int((Rm & Rg).sum())
    union = int((Rm | Rg).sum())
    false = Rm & ~Fg
    return dict(
        reach_recall=inter / max(int(Rg.sum()), 1),
        n_reach_true=inter, n_gt_reach=int(Rg.sum()), n_map_reach=int(Rm.sum()),
        reach_iou=inter / max(union, 1),
        false_reach_rate=int(false.sum()) / max(int((~Fg).sum()), 1),
        n_false_reach=int(false.sum()),
        stand_agreement=float(((tm["posture"] == STAND) == (tg["posture"] == STAND))[Rm & Rg].mean())
        if inter else float("nan"),
        n_map_stoop=int((Rm & (tm["posture"] == STOOP)).sum()),
        n_map_crawl=int((Rm & (tm["posture"] == CRAWL)).sum()),
        n_map_sideways=int((Rm & (tm["width"] == SIDEWAYS)).sum()),
    )
