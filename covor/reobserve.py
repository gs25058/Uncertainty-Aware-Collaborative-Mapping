"""Rule-based single-view selection. No truth or future image enters this API.

Candidates contain proposed estimated camera poses, robot IDs and timestamps.
The visibility proxy raycasts only the CURRENT label grid. It predicts evidence
coverage, not the actual information content of an unseen observation.
"""
import heapq

import numpy as np
from scipy import ndimage


def route_importance(labels, ijk_min, res, entry, goals):
    """Optimistic paths to fixed mission goals, costly through unknown/solid.

    This 2D proxy is ONLY a selector; the existing 3D posture model makes all
    final verdicts. Solid columns remain costly rather than being assumed
    traversable. Candidate measurements and GT are not inputs.
    """
    nx, ny, nz = labels.shape
    z = (np.arange(nz) + ijk_min[2] + .5) * res
    band = (z >= .3) & (z <= 1.8)
    unknown = (labels[:, :, band] == 0).mean(axis=2)
    occupied = (labels[:, :, band] == 2).mean(axis=2)
    cost = 1 + 3 * unknown + 20 * occupied
    start = np.clip(np.floor(np.array(entry)/res).astype(int)-ijk_min[:2],
                    0, np.array([nx,ny])-1)
    start = tuple(start)
    distance = np.full((nx,ny), np.inf)
    parent = np.full((nx,ny,2), -1, np.int32)
    distance[start] = 0
    queue = [(0., *start)]
    while queue:
        d, i, j = heapq.heappop(queue)
        if d > distance[i,j]:
            continue
        for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
            a,b = i+di,j+dj
            if 0 <= a < nx and 0 <= b < ny:
                nd = d + .5*(cost[i,j]+cost[a,b])
                if nd < distance[a,b]:
                    distance[a,b] = nd
                    parent[a,b] = i,j
                    heapq.heappush(queue, (nd,a,b))
    use = np.zeros((nx,ny), float)
    for goal in goals:
        end = np.floor(np.array([goal["x"],goal["y"]])/res).astype(int)-ijk_min[:2]
        if not ((end>=0)&(end<np.array([nx,ny]))).all():
            continue
        node = tuple(end)
        while node != start:
            use[node] += 1
            node = tuple(parent[node])
            if node[0] < 0:
                break
    # A half-body-width neighbourhood of every path. Routes shared by several
    # goals receive more weight; current obstacle boundaries mark bottlenecks.
    near = ndimage.maximum_filter(use, size=7)
    blocked = occupied > .1
    boundary = ndimage.binary_dilation(blocked) & ~blocked
    return near * (1 + boundary.astype(float)) * (1 + unknown)


def predicted_unknown(labels, ijk_min, res, T_wc, sd, step=.2):
    """Coarse camera-frustum rays, stopped at CURRENT occupied cells only."""
    h, w = sd.calib.size[1], sd.calib.size[0]
    v,u = np.meshgrid(np.linspace(0,h-1,16), np.linspace(0,w-1,24), indexing="ij")
    dirs = np.stack(((u-sd.cx)/sd.fx,(v-sd.cy)/sd.fy,np.ones_like(u)), -1).reshape(-1,3)
    dirs = dirs @ sd.R1 @ T_wc[:3,:3].T  # rectified -> raw -> world
    dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    active = np.ones(len(dirs), bool)
    seen = []
    for d in np.arange(.3,5.01,step):
        ind = np.floor((T_wc[:3,3] + dirs*d)/res).astype(int)-ijk_min
        inside = ((ind>=0)&(ind<np.array(labels.shape))).all(axis=1)
        active &= inside
        ids = np.nonzero(active)[0]
        if len(ids)==0:
            break
        p = ind[ids]
        lab = labels[tuple(p.T)]
        if (lab==0).any():
            seen.append(p[lab==0])
        active[ids[lab==2]] = False
    return np.unique(np.concatenate(seen),axis=0) if seen else np.empty((0,3),int)


def rank_candidates(labels, ijk_min, res, candidates, calibrations,
                    reference_positions, entry, goals):
    """All three policies see exactly these inputs, never unseen depth/GT."""
    importance = route_importance(labels,ijk_min,res,entry,goals)
    scores=[]
    for cand in candidates:
        T = np.asarray(cand["T"])
        cells = predicted_unknown(labels,ijk_min,res,T,calibrations[cand["robot"]])
        cols = np.unique(cells[:,:2],axis=0)
        route_score = float(importance[tuple(cols.T)].sum()) if len(cols) else 0.
        scores.append(dict(id=cand["id"], robot=cand["robot"], timestamp=cand["timestamp"],
                           distance=float(np.linalg.norm(T[:3,3]-reference_positions[cand["robot"]])),
                           expected_unknown_voxels=len(cells),
                           route_critical_score=route_score))
    if not scores:
        raise ValueError("candidate set is empty")
    # Stable ID breaks ties, never the subsequently observed utility.
    picks = dict(nearest=min(scores,key=lambda s:(s["distance"],s["id"]))["id"],
                 unknown=max(scores,key=lambda s:(s["expected_unknown_voxels"],-s["id"]))["id"],
                 route=max(scores,key=lambda s:(s["route_critical_score"],-s["id"]))["id"])
    return picks,scores,importance


def goal_verdicts(traversal, strict_traversal, labels, ijk_min, res, goals):
    """No goal snap. A gap-assisted route is provisional, not confirmed.

    Strict traversal uses identical body/posture/step rules but gap=0: a
    confirmation therefore requires observed supports along an entire route.
    Failure to find a route is unverified, not proof of a blocked route.
    """
    out=[]
    for goal in goals:
        xy=np.floor(np.array([goal["x"],goal["y"]])/res).astype(int)-ijk_min[:2]
        inside=((xy>=0)&(xy<np.array(labels.shape[:2]))).all()
        at=tuple(xy)
        reach=bool(inside and traversal["reach"][at])
        confirmed=bool(inside and strict_traversal["reach"][at])
        out.append(dict(goal=goal["id"],x=goal["x"],y=goal["y"],
                        verdict="confirmed" if confirmed else "unverified",
                        gap_permitted_reach=reach,confirmed=confirmed))
    return out
