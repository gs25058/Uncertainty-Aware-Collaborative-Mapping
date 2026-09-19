"""Entry point -> goal routing on the entry grid (DESIGN_entry_map.md §2-4).

    cost(step into b) = len(step) * (1 + lambda / clearance(b))
                        + len(step) * unknown_penalty_steps   if b touches unknown

A* over PASSABLE cells only. Unknown is never routed through -- it is an obstacle
(DESIGN §3-2) and so is not in the search space at all; the penalty is for
walking ALONGSIDE it, which is a different and weaker statement: the map is
saying "I cannot see what is beside you here", and a route that has an
alternative should take it.

The clearance term is bounded by construction: a passable cell has
clearance >= r >= w/2, so the multiplier never exceeds 1 + 2*lambda/w. At the
defaults that is 1 + 1.43, i.e. a hugging-the-wall step costs at most 2.4 times
an open-floor one. Big enough to push a route into the middle of a corridor,
small enough that it will still use a narrow gap rather than not arrive.

8-connected movement, with diagonal steps costing sqrt(2) -- unlike reachability,
which is 4-connected. A route is a path a body walks and may cut a corner; a
connected component is a claim about whether two cells are the same space, and
there the stricter rule is the safe one.
"""
import heapq

import numpy as np

from .config import UNKNOWN

_STEPS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
          (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]


def unknown_adjacent(band):
    """Cells with an UNKNOWN cell in their 8-neighbourhood."""
    u = np.asarray(band) == UNKNOWN
    out = np.zeros(u.shape, bool)
    for di, dj, _ in _STEPS:
        out |= np.roll(np.roll(u, di, 0), dj, 1)
    # np.roll wraps; the border can only be made MORE penalised by that, never
    # less, so it is left rather than special-cased into a second code path.
    return out


def astar(passable, clearance, band, start, goal, cfg):
    """Least-cost path from ``start`` to ``goal``, as a list of (i, j), or None."""
    passable = np.asarray(passable, bool)
    clear = np.asarray(clearance, float)
    pen = unknown_adjacent(band)
    res, lam = cfg.res, cfg.lambda_route
    up = cfg.unknown_penalty_steps
    ni, nj = passable.shape
    if not passable[start] or not passable[goal]:
        return None

    def h(c):
        # euclidean distance in metres: every step costs at least its length,
        # so this never over-estimates and A* stays exact.
        return res * np.hypot(c[0] - goal[0], c[1] - goal[1])

    g = {start: 0.0}
    prev = {}
    pq = [(h(start), 0.0, start)]
    seen = set()
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            path = [cur]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1]
        ci, cj = cur
        for di, dj, w in _STEPS:
            b = (ci + di, cj + dj)
            if not (0 <= b[0] < ni and 0 <= b[1] < nj) or not passable[b] or b in seen:
                continue
            step = res * w
            c = gc + step * (1.0 + lam / max(clear[b], 1e-9)) + (step * up if pen[b] else 0.0)
            if c < g.get(b, np.inf):
                g[b] = c
                prev[b] = cur
                heapq.heappush(pq, (c + h(b), c, b))
    return None


def geodesic(passable, start):
    """Step-count distance from ``start`` over passable cells (8-connected).

    Used to pick goals, not to cost routes -- it answers "how far into the
    building is this", which is what makes a goal worth walking to.
    """
    passable = np.asarray(passable, bool)
    d = np.full(passable.shape, np.inf)
    d[start] = 0.0
    frontier = [start]
    ni, nj = passable.shape
    while frontier:
        nxt = []
        for ci, cj in frontier:
            for di, dj, w in _STEPS:
                b = (ci + di, cj + dj)
                if (0 <= b[0] < ni and 0 <= b[1] < nj and passable[b]
                        and d[b] > d[ci, cj] + w):
                    d[b] = d[ci, cj] + w
                    nxt.append(b)
        frontier = nxt
    return d


def default_goals(passable, reachable, clearance, start, n=2):
    """Stand-in for DESIGN §2-4's "each room centre", which needs a room
    segmentation this pipeline does not have.

    Two goals, both defined on the reachable set alone so they are a function of
    the map: the cell FARTHEST from the entry by geodesic distance (the far end
    of the space, i.e. the longest journey the map claims is possible), and the
    cell of GREATEST clearance (the most open spot, where a team would stage).
    If the two coincide only one is returned.
    """
    R = np.asarray(reachable, bool)
    if not R.any():
        return []
    d = geodesic(passable, start)
    d = np.where(R, d, -np.inf)
    far = np.unravel_index(int(np.argmax(np.where(np.isfinite(d), d, -np.inf))), d.shape)
    c = np.where(R, np.asarray(clearance, float), -np.inf)
    open_ = np.unravel_index(int(np.argmax(c)), c.shape)
    goals = [far] + ([open_] if open_ != far else [])
    return goals[:n]


def describe(path, clearance, band, cfg, ijk_min):
    """The per-route record DESIGN §1 asks routes.json to carry.

    ``min_width_m`` is twice the smallest clearance on the path: clearance is a
    half-width (distance to the nearest obstacle), and what a reader needs is the
    gap they have to fit through.
    """
    p = np.asarray(path)
    clear = np.asarray(clearance, float)[p[:, 0], p[:, 1]]
    step = cfg.res * np.where(
        (np.abs(np.diff(p[:, 0])) + np.abs(np.diff(p[:, 1]))) == 2, np.sqrt(2), 1.0)
    pen = unknown_adjacent(band)[p[:, 0], p[:, 1]]
    k = int(np.argmin(clear))
    # contiguous runs of unknown-adjacent cells, which is what a reader acts on:
    # "20 m in, 8 cells of the route have unseen space beside them".
    runs, run = [], 0
    for v in pen.tolist() + [False]:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    world = (p + np.asarray(ijk_min[:2]) + 0.5) * cfg.res
    return dict(
        n_cells=int(len(p)),
        length_m=float(step.sum()),
        min_clearance_m=float(clear[k]),
        min_width_m=float(2.0 * clear[k]),
        min_width_at=[float(world[k, 0]), float(world[k, 1])],
        median_width_m=float(2.0 * np.median(clear)),
        n_unknown_adjacent=int(pen.sum()),
        unknown_adjacent_runs=runs,
        start_xy=[float(world[0, 0]), float(world[0, 1])],
        goal_xy=[float(world[-1, 0]), float(world[-1, 1])],
        cells=p.tolist(),
    )


def plan(grid, band, cfg, goals=None):
    """Every route for one band: {"routes": [...], "entry_ij": [...]}.

    ``grid`` is a build_entry_map.build_entry_grid result (or the loaded npz).
    """
    P = np.asarray(grid["%s_passable" % band], bool)
    R = np.asarray(grid["%s_reachable" % band], bool)
    C = np.asarray(grid["%s_clearance" % band], float)
    L = np.asarray(grid["%s_label" % band])
    ijk = np.asarray(grid["ijk_min"])
    start = tuple(int(v) for v in np.asarray(grid["entry_ij"]))
    if goals is None:
        goals = default_goals(P, R, C, start)
    out = []
    for gl in goals:
        gl = tuple(int(v) for v in gl)
        path = astar(P, C, L, start, gl, cfg)
        if path is None:
            out.append(dict(goal_ij=list(gl), reachable=False))
            continue
        rec = describe(path, C, L, cfg, ijk)
        rec.update(goal_ij=list(gl), reachable=True)
        out.append(rec)
    return dict(band=band, entry_ij=[int(v) for v in start], routes=out)
