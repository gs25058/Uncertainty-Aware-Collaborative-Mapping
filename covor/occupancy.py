"""slam_to_occupancy: uncertainty-aware occupancy mapping from fused poses.

Implements the research-proposal pipeline (§3③, §4) on top of the CoVOR-SLAM
fused poses produced by ``covor.fusion``:

    fused pose T^k_n (SE3) + marginal covariance Sigma^k_n   (covor.fusion)
        -> stereo depth Z, sigma_Z            (StereoDepth,  proposal Step A)
        -> back-projected camera points P_cam (backproject,  proposal Step B)
        -> P_world = R P_cam + t              (              proposal §3③)
        -> voxel ray traversal + WEIGHTED log-odds update into an OctoMap OcTree
           with w = exp(-tr Sigma_pos/alpha) * exp(-sigma_Z^2/beta)   (Step C/D/E/F)
        -> free / occupied / unknown classification + .bt export      (Step G)

The whole thing is pure Python (OpenCV SGBM + octomap-python ``updateNode`` with a
float log-odds argument); no C++ of OctoMap/OpenCV is modified, matching the
proposal's §7.3 tooling table. The uncertainty weighting can be toggled off
(``OccCfg.weighted=False``) to recover a standard OctoMap update, so the
uniform-vs-weighted ablation (proposal Phase 3-7) is structural.

This module is reused unchanged for the 1 / 2 / 3-drone comparison (proposal
Phase 3-3): more drones only changes which robots' observations are accumulated
into the shared tree and how small their Sigma is.
"""
from dataclasses import dataclass, field
import os
import numpy as np
import yaml
import cv2

MILUV = "/src/gs25058/cr_RNE/miluv"


# ---------------------------------------------------------------------------
# Stereo calibration (MILUV Kalibr intrinsics.yaml)
# ---------------------------------------------------------------------------
@dataclass
class StereoCalib:
    K_l: np.ndarray      # 3x3 left  (infra1) intrinsics
    D_l: np.ndarray      # 4  left  radtan distortion
    K_r: np.ndarray      # 3x3 right (infra2) intrinsics
    D_r: np.ndarray      # 4  right radtan distortion
    R: np.ndarray        # 3x3 rotation, left->right (p_right = R p_left + T)
    T: np.ndarray        # 3   translation, left->right
    size: tuple          # (width, height)

    @property
    def baseline(self):
        return float(abs(self.T[0]))


def _K(intr):
    fx, fy, cx, cy = intr
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)


def load_stereo_calib(robot="ifo001"):
    """Load the left/right IR stereo calibration for a MILUV robot.

    cam0 = infra1 (left), cam1 = infra2 (right). Kalibr stores cam1.T_cn_cnm1 =
    the transform from cam0 to cam1 (p_cam1 = T @ p_cam0), i.e. exactly the
    (R, T) cv2.stereoRectify expects (left as camera 1, right as camera 2).
    """
    path = os.path.join(MILUV, "config", "realsense", robot, "intrinsics.yaml")
    with open(path) as f:
        c = yaml.safe_load(f)
    c0, c1 = c["cam0"], c["cam1"]
    Tcn = np.array(c1["T_cn_cnm1"], float)      # cam0 -> cam1
    w, h = c0["resolution"]
    return StereoCalib(
        K_l=_K(c0["intrinsics"]), D_l=np.array(c0["distortion_coeffs"], float),
        K_r=_K(c1["intrinsics"]), D_r=np.array(c1["distortion_coeffs"], float),
        R=Tcn[:3, :3], T=Tcn[:3, 3], size=(int(w), int(h)))


# ---------------------------------------------------------------------------
# Step A: stereo depth (proposal §4.2)
# ---------------------------------------------------------------------------
@dataclass
class DepthCfg:
    z_max: float = 5.0          # m; discard beyond this (stereo unreliable) §4.2
    z_min: float = 0.3          # m; below this the disparity is saturated/bogus
    disp_sigma_px: float = 1.0  # Delta d: sub-pixel disparity uncertainty for sigma_Z
    # SGBM parameters (kept standard; not tuned for score, per task constraint)
    min_disp: int = 0
    num_disp: int = 96          # must be divisible by 16
    block: int = 7


class StereoDepth:
    """Rectified SGBM stereo -> metric depth Z and depth uncertainty sigma_Z.

    Returns depth in the RAW infra1 optical frame (points are rotated back by
    R1^T from the rectified frame), so the depth shares the exact camera frame of
    the ORB-SLAM3 / fused pose, keeping P_world = T_wc @ P_cam correct. Invalid
    pixels (no disparity, or Z outside [z_min, z_max]) are left as NaN so no ray
    is cast for them -> the cell stays 'unknown' (proposal: never force free).
    """

    def __init__(self, calib: StereoCalib, cfg: DepthCfg = None):
        self.calib = calib
        self.cfg = cfg or DepthCfg()
        w, h = calib.size
        # Rectification. Distortion ~1e-4 and inter-cam rotation ~0.2 deg, but we
        # rectify properly for correctness; R1 (tiny) is stored to map points back
        # to the raw infra1 frame.
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            calib.K_l, calib.D_l.reshape(1, -1),
            calib.K_r, calib.D_r.reshape(1, -1), (w, h),
            np.ascontiguousarray(calib.R), calib.T.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.R1, self.Q = R1, Q
        self.P1 = P1
        # rectified-left intrinsics (used for sigma_Z and as the effective K)
        self.fx = float(P1[0, 0]); self.fy = float(P1[1, 1])
        self.cx = float(P1[0, 2]); self.cy = float(P1[1, 2])
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            calib.K_l, calib.D_l, R1, P1, (w, h), cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            calib.K_r, calib.D_r, R2, P2, (w, h), cv2.CV_32FC1)
        cc = self.cfg
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=cc.min_disp, numDisparities=cc.num_disp,
            blockSize=cc.block,
            P1=8 * cc.block ** 2, P2=32 * cc.block ** 2,
            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
            disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def rectify_pair(self, img_l, img_r):
        rl = cv2.remap(img_l, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rr = cv2.remap(img_r, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rl, rr

    def disparity(self, img_l, img_r):
        rl, rr = self.rectify_pair(img_l, img_r)
        disp = self.sgbm.compute(rl, rr).astype(np.float32) / 16.0  # SGBM Q4.4
        return disp, rl

    def depth(self, img_l, img_r):
        """-> (Z, sigma_Z, valid) maps in the rectified-left frame.

        Z, sigma_Z: float32 (H,W), NaN where invalid. valid: bool (H,W).
        """
        cc = self.cfg
        disp, _ = self.disparity(img_l, img_r)
        fB = self.fx * self.calib.baseline
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = fB / disp
        valid = (disp > 0) & np.isfinite(Z) & (Z >= cc.z_min) & (Z <= cc.z_max)
        Z = np.where(valid, Z, np.nan).astype(np.float32)
        # sigma_Z = Z^2/(f B) * Delta d   (proposal §4.2) -> grows with Z^2
        sigma_Z = (Z ** 2 / fB * cc.disp_sigma_px).astype(np.float32)
        return Z, sigma_Z, valid

    # -- Step B: back-projection (proposal §4.3) ---------------------------
    def backproject(self, Z, sigma_Z, valid, downsample=4):
        """Rectified depth -> camera-frame points in the RAW infra1 frame.

        Returns (P_cam (N,3), sZ (N,)). 4x4 downsampling (proposal §4.2): the
        pixel density far exceeds the voxel resolution, so this is lossless for
        mapping. Points are rotated by R1^T from the rectified frame back to the
        raw infra1 optical frame, which is the frame of the fused/VO pose.
        """
        ds = downsample
        Zs = Z[::ds, ::ds]; sZs = sigma_Z[::ds, ::ds]; vs = valid[::ds, ::ds]
        h, w = Zs.shape
        vv, uu = np.mgrid[0:h, 0:w]
        u = uu[vs] * ds; v = vv[vs] * ds; z = Zs[vs]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        P_rect = np.stack([x, y, z], axis=1)            # rectified-left frame
        P_cam = P_rect @ self.R1                        # = R1^T @ P (rows) -> raw infra1
        return P_cam.astype(np.float64), sZs[vs].astype(np.float64)


# ---------------------------------------------------------------------------
# Step C/D/E/F/G: uncertainty-aware occupancy (proposal §4.4 - §4.8)
# ---------------------------------------------------------------------------
@dataclass
class OccCfg:
    resolution: float = 0.10     # voxel size (m)
    # Safety-asymmetric observation model (proposal §4.6): |l_free| < |l_occ|
    l_occ: float = 0.85          # ~ logodds(0.7), evidence for OCCUPIED (terminal)
    l_free: float = -0.40        # ~ logodds(0.4), evidence for FREE (pass-through)
    clamp_min: float = -2.0      # octomap default log clamp (prob 0.12)
    clamp_max: float = 3.5       # octomap default log clamp (prob 0.97)
    # Classification thresholds (proposal §4.8)
    # tau_occ = l_occ is a DEFINITION, not a tuned value: a single observation
    # contributes w*l_occ <= l_occ, so l > tau_occ is unreachable from one
    # observation whatever w is. "Occupied requires more evidence than one ideal
    # observation can provide." Measured on default_3_zigzag_0/ifo001, 75% of the
    # cells that tau_occ=0 called occupied were single-observation cells -- one
    # stereo mismatch became a permanent floating cube. Demoting them is safe in
    # the proposal's terms: they become UNKNOWN, not free, and §4.8 forbids
    # treating unknown as traversable, so nothing gains traversability.
    tau_occ: float = 0.85        # l > +tau_occ -> occupied  (== l_occ)
    tau_free: float = 0.0        # l < -tau_free -> free (else unknown)
    # Uncertainty weighting (proposal §4.5): w = exp(-trSig/alpha)*exp(-sZ^2/beta)
    weighted: bool = True        # ablation toggle: False -> standard OctoMap (w=1)
    # Mediation arm for the §4.9 analysis. w is a PRODUCT of two terms, so turning
    # `weighted` off removes both and only re-measures the already-established
    # depth term. This flag substitutes w_pose = 1 at evaluation time, leaving
    # w_depth intact, so (full - pose_off) isolates the pose term -- the one §4.9
    # claims. It changes nothing about the §4.5 formula itself.
    use_w_pose: bool = True      # False -> w = w_depth only (w_pose forced to 1)
    # Record, per cell, the evidence-weighted mean tr(Sigma) of the frames that
    # wrote it (OccupancyBuilder.sigma_attribution). Turns the tr(Sigma)-vs-outcome
    # correlation from one point per condition into one point per cell.
    track_sigma_attribution: bool = False
    # Spatial spreading of POSE position uncertainty (PREREG_spatial_spread.md).
    # Pose covariance is a POSITION uncertainty; scaling log-odds by w encodes an
    # EXISTENCE uncertainty, which defeats the §4.6 safety asymmetry (measured:
    # false-free +0.05..+0.11 pp). Instead, spread the endpoint's occupied evidence
    # along the ray over sigma = sqrt(tr Sigma_pos) -- MASS CONSERVED, so evidence
    # is relocated, never weakened -- and stop free carving 2 sigma short of the
    # endpoint so nothing asserts free inside the uncertain band. Width comes
    # straight from the measurement: no coefficient, and alpha is unused here.
    # Forces w_pose = 1 (spreading already handles pose uncertainty).
    # --- free-side pose-uncertainty encoding (PREREG_free_side.md) ---
    # POST-HOC DERIVED AFTER §8. Both §8 encodings changed the OCCUPIED evidence;
    # the damage mechanism the proposal describes is a FREE-side one (a ray punching
    # through a wall and painting the space behind it free). These modes put the
    # pose uncertainty on the free evidence ONLY -- l_occ is never touched.
    #   "off"         current behaviour
    #   "free_scale"  l_free *= exp(-sigma_u^2 / alpha_u)          (arm_F1)
    #   "free_trunc"  drop the last floor(k*sigma_u/res) free cells of each ray,
    #                 leaving that band UNKNOWN                     (arm_F2)
    # sigma_u = sqrt(u^T Sigma_pos u) is computed PER RAY -- punch-through is
    # direction-dependent and an isotropic tr(Sigma) averages that away.
    # Both modes carry ONE sub-voxel guard: sigma_u < res/2 -> exact no-op. (§8.3
    # fired its falsifier because that guard was applied to one half of a rule and
    # not the other; here it is a single decision per ray.)
    pose_mode: str = "off"
    alpha_u: float = None        # None -> res**2 (sigma_u = 1 voxel gives w ~ e^-1)
    trunc_k: float = 1.0
    spread_pose_sigma: bool = False
    spread_trunc: float = 2.0    # +-2 sigma ~ 95%; used for BOTH the spread
                                 # support and the free-carving stop
    # alpha, beta are the proposal's "experimentally-determined" scale constants
    # (§4.5). They are set to the data's uncertainty scale so a MEDIAN-quality
    # observation keeps w~0.85 while outliers collapse toward 0:
    #  - median well-registered pose has tr(Sigma_pos) ~ 0.05 m^2 (measured);
    #    alpha=0.3 -> w_pose=exp(-0.05/0.3)=0.85. Map-break poses (tr~100) -> ~0.
    #  - mid-range depth sigma_Z ~ 0.3 m (Z~2.3 m); beta=0.5 -> w_depth=exp(-0.09/0.5)
    #    =0.84. Far points (Z~5 m, sigma_Z~1 m) -> w_depth~0.13.
    alpha: float = 0.3           # pose-covariance scale (m^2); tr(Sigma_pos) units
    beta: float = 0.5            # depth-uncertainty scale (m^2); sigma_Z^2 units
    max_ray: float = 5.0         # m; cap ray length (matches depth z_max)
    # Teammate (dynamic-object) exclusion. The drones fly together and see each
    # other constantly; a beam terminating on a teammate is a correct observation
    # of a MOVING object, which has no place in a static occupancy map. Measured
    # on default_3_zigzag_0/ifo001 with GT poses, 79% of the occupied voxels left
    # floating inside the room sit within 0.4 m of a teammate's position at the
    # time of the observation. Only the endpoint (occupied) evidence is dropped;
    # the free evidence along the beam is kept, since that space really was
    # traversed. This is available only because the CoVOR fusion puts every robot
    # in one frame -- a side benefit of §4.7's collaborative accumulation.
    dyn_radius: float = 0.35     # m; drone half-size + pose error. 0 disables.


def _sigmoid(x):
    """log-odds -> probability, without overflowing for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (1.0 + e)


def _voxel_traverse(o, e, res):
    """Amanatides & Woo 3D DDA for ONE ray -- the reference implementation.

    Kept as the ground truth that ``_dda_batch`` (the vectorized version used in
    the hot loop) is asserted against, cell-by-cell AND value-by-value, in
    tests/test_ray_traversal.py. Yield integer voxel indices from o to e,
    EXCLUDING the terminal voxel (returned separately as the occupied cell).

    o, e: world points (m). res: voxel size. Returns (free_idx (M,3) int,
    end_idx (3,) int). free_idx are the pass-through voxels; end_idx the voxel
    containing e (the beam endpoint / occupied evidence).
    """
    oi = np.floor(o / res).astype(np.int64)
    ei = np.floor(e / res).astype(np.int64)
    d = e - o
    step = np.sign(d).astype(np.int64)
    free = []
    cur = oi.copy()
    # tMax: param at which the ray crosses the next voxel boundary per axis
    tmax = np.full(3, np.inf); tdelta = np.full(3, np.inf)
    for a in range(3):
        if step[a] != 0:
            nxt = (cur[a] + (step[a] > 0)) * res
            tmax[a] = (nxt - o[a]) / d[a]
            tdelta[a] = res / abs(d[a])
    guard = 0
    max_guard = int(np.abs(ei - oi).sum()) + 3
    while not np.array_equal(cur, ei) and guard < max_guard:
        a = int(np.argmin(tmax))
        cur = cur.copy(); cur[a] += step[a]
        tmax[a] += tdelta[a]
        if np.array_equal(cur, ei):
            break
        free.append(cur.copy())
        guard += 1
    free_idx = np.array(free, dtype=np.int64) if free else np.empty((0, 3), np.int64)
    return free_idx, ei


def _dda_batch(o, P, res):
    """Vectorized Amanatides & Woo DDA from a shared origin ``o`` to every point
    in ``P`` (N,3). Returns (ray, vox): ``vox`` (M,3) are the PASS-THROUGH voxel
    indices, ``ray`` (M,) the index of the ray each one belongs to. Origin and
    endpoint cells are excluded -- identical semantics to ``_voxel_traverse``.

    Every ray visits every cell it crosses EXACTLY ONCE, which is what makes the
    free evidence contributed by one beam to one cell exactly ``w * l_free``
    (proposal §4.5). The previous implementation sampled each beam at res/2 in
    arc length instead, which both (a) dropped 2-4x the intended free evidence
    into cells containing several samples -- inverting the safety asymmetry
    |l_free| < |l_occ| of §4.6 -- and (b) skipped cells the beam only clipped,
    biasing the carving by ray direction. All rays share one origin (the camera
    centre), so tMax/tDelta are set up once and the loop marches the whole active
    set in lockstep, shrinking as rays reach their endpoint.
    """
    N = len(P)
    d = P - o
    oi = np.floor(o / res).astype(np.int64)
    ei = np.floor(P / res).astype(np.int64)
    step = np.sign(d).astype(np.int64)
    cur = np.tile(oi, (N, 1))
    nz = step != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        nxt = (cur + (step > 0)) * res              # next boundary per axis
        tmax = np.where(nz, (nxt - o) / d, np.inf)
        tdelta = np.where(nz, res / np.abs(d), np.inf)
    act = np.nonzero(~np.all(cur == ei, axis=1))[0]
    max_steps = int(np.abs(ei - oi).sum(axis=1).max()) + 2 if N else 0
    rays, voxs = [], []
    for _ in range(max_steps):
        if act.size == 0:
            break
        a = np.argmin(tmax[act], axis=1)            # axis of the nearest crossing
        cur[act, a] += step[act, a]
        tmax[act, a] += tdelta[act, a]
        act = act[~np.all(cur[act] == ei[act], axis=1)]   # drop rays that arrived
        rays.append(act.copy())
        voxs.append(cur[act].copy())
    if not rays:
        return np.empty(0, np.int64), np.empty((0, 3), np.int64)
    return np.concatenate(rays), np.concatenate(voxs)


class OccupancyBuilder:
    """Reusable slam_to_occupancy module (proposal Phase 1-6).

    Accumulates weighted log-odds evidence from any number of (pose, Sigma,
    depth) observations, from one or several drones, into a single OctoMap
    OcTree. The same instance is used for the 1/2/3-drone comparison: just call
    integrate_frame(...) for each robot's keyframes into the same tree.
    """

    def __init__(self, cfg: OccCfg = None):
        import octomap
        self.cfg = cfg or OccCfg()
        self.tree = octomap.OcTree(self.cfg.resolution)
        self.tree.setClampingThresMin(_sigmoid(self.cfg.clamp_min))
        self.tree.setClampingThresMax(_sigmoid(self.cfg.clamp_max))
        # keep OctoMap's own notion of "occupied" (isNodeOccupied, .bt export)
        # identical to the §4.8 classification below, so every consumer agrees
        self.tree.setOccupancyThres(_sigmoid(self.cfg.tau_occ))
        self.n_frames = 0
        self.n_points = 0
        self._written = False        # write_bt is destructive; see its docstring
        # (num, den) dicts keyed by the same packed voxel key integrate_frame uses;
        # enabled by track_sigma_attribution.
        self._attr = ({}, {}) if getattr(cfg, "track_sigma_attribution", False) else None
        self._su = []       # per-frame arrays of per-ray sigma_u (free-side modes)
        self._trunc = []    # per-frame arrays of per-ray truncated cell counts

    def weight(self, tr_sigma_pos, sigma_Z):
        """w = exp(-tr(Sigma_pos)/alpha) * exp(-sigma_Z^2/beta)  (proposal §4.5).

        tr_sigma_pos: scalar (m^2), the pose position-covariance trace (bigger =
        registration less certain). sigma_Z: (N,) per-point depth uncertainty.
        Returns w in (0,1] per point. If weighting is off, returns ones.
        """
        c = self.cfg
        if not c.weighted:
            return np.ones_like(sigma_Z)
        # use_w_pose=False is the mediation arm: keep w_depth, drop w_pose.
        # spread_pose_sigma also forces it off -- the spread handles pose
        # uncertainty, and scaling on top would double-count it.
        w_pose = (np.exp(-tr_sigma_pos / c.alpha)
                  if (c.use_w_pose and not c.spread_pose_sigma) else 1.0)
        w_depth = np.exp(-(sigma_Z ** 2) / c.beta)            # per-point
        return np.clip(w_pose * w_depth, 0.0, 1.0)

    def integrate_frame(self, T_wc, tr_sigma_pos, P_cam, sigma_Z, teammates=None,
                        sigma_pos=None):
        """Integrate one keyframe's observation (proposal §4.4-4.7), vectorized.

        T_wc: 4x4 world<-camera pose (fused). tr_sigma_pos: pose position-cov
        trace. P_cam: (N,3) camera-frame points (raw infra1). sigma_Z: (N,).
        teammates: (K,3) world positions of the OTHER robots at this instant, or
        None. Beams ending within cfg.dyn_radius of one contribute free evidence
        but no occupied evidence (see OccCfg.dyn_radius).

        Casts one ray per point (free evidence w*l_free along the beam, occupied
        evidence w*l_occ at the endpoint) with an exact vectorized DDA, and
        scatter-adds the weighted log-odds per cell. Per-cell evidence from all
        rays is summed (log-odds additivity, §4.7): a cell crossed by two beams
        gets 2*w*l_free, and a cell that is an endpoint for one ray and
        pass-through for another sums both. But a SINGLE beam contributes to a
        cell exactly once -- see ``_dda_batch``.
        """
        c = self.cfg
        res = c.resolution
        R = T_wc[:3, :3]; t = T_wc[:3, 3]
        P = P_cam @ R.T + t                                    # §3③: R P_cam + t
        w = self.weight(tr_sigma_pos, sigma_Z)
        L = np.linalg.norm(P - t, axis=1)
        keep = (L > res) & (L <= c.max_ray)
        P, w = P[keep], w[keep]
        if len(P) == 0:
            self.n_frames += 1
            return 0

        # occupied endpoints, minus the ones that landed on a flying teammate
        static = np.ones(len(P), bool)
        if teammates is not None and len(teammates) and c.dyn_radius > 0:
            near = np.linalg.norm(P[:, None, :] - np.atleast_2d(teammates)[None, :, :],
                                  axis=2).min(axis=1) < c.dyn_radius
            static = ~near
        if not c.spread_pose_sigma:
            evox = np.floor(P[static] / res).astype(np.int64)
            ew = w[static] * c.l_occ          # OCCUPIED evidence: never modified here
            # free evidence: exact DDA, one contribution per (ray, pass-through cell)
            ray, fvox = _dda_batch(t, P, res)
            fw = w[ray] * c.l_free
            if c.pose_mode != "off":
                # sigma_u per ray, then a single sub-voxel guard per ray.
                if sigma_pos is None:
                    raise ValueError("pose_mode requires sigma_pos (3x3)")
                S = np.asarray(sigma_pos, float).reshape(3, 3)
                U = (P - t) / np.linalg.norm(P - t, axis=1)[:, None]
                su = np.sqrt(np.maximum(np.einsum('ni,ij,nj->n', U, S, U), 0.0))
                su = np.where(su >= res / 2, su, 0.0)      # guard -> exact no-op
                self._su.append(su)
                if c.pose_mode == "free_scale":
                    au = c.alpha_u if c.alpha_u is not None else res ** 2
                    fw = fw * np.exp(-(su[ray] ** 2) / au)
                elif c.pose_mode == "free_trunc":
                    # drop the LAST floor(k*sigma_u/res) cells of each ray. _dda_batch
                    # emits a ray's cells in origin->endpoint order (verified), so the
                    # tail of each ray's block is the band next to the endpoint.
                    ndrop = np.floor(c.trunc_k * su / res).astype(np.int64)
                    if ndrop.any():
                        order = np.argsort(ray, kind='stable')
                        r_s = ray[order]
                        cnt = np.bincount(r_s, minlength=len(P))
                        start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
                        pos = np.arange(len(r_s)) - start[r_s]      # index within ray
                        keep_s = pos < (cnt[r_s] - np.minimum(ndrop[r_s], cnt[r_s]))
                        keep = np.zeros(len(ray), bool); keep[order] = keep_s
                        self._trunc.append(np.minimum(ndrop, cnt).astype(float))
                        ray, fvox, fw = ray[keep], fvox[keep], fw[keep]
                    else:
                        self._trunc.append(np.zeros(len(P)))
                else:
                    raise ValueError("unknown pose_mode %r" % c.pose_mode)
        else:
            # --- spatial spreading (PREREG_spatial_spread.md) ---
            sig = float(np.sqrt(max(tr_sigma_pos, 0.0)))
            # ONE sub-voxel decision, applied to BOTH halves of the rule.
            # PREREG_spatial_spread.md guarded only the occupied spread and left the
            # free truncation unconditional, so at sigma < res/2 the two halves
            # disagreed: the spread was a no-op while the truncation still stripped
            # ~0.78 of a voxel from every ray (2.3 % of the free evidence on D).
            # That is what fired P2 there. With trunc forced to 0 when inactive,
            # sc == 1 and the free carving is bit-identical to the no-spread path,
            # so D/E are a true no-op. (PREREG_RESOLUTION.md §4)
            spread_active = sig >= res / 2
            trunc = c.spread_trunc * sig if spread_active else 0.0
            Ps = P[static]
            if not spread_active or len(Ps) == 0:
                evox = np.floor(Ps / res).astype(np.int64)
                ew = w[static] * c.l_occ
            else:
                Ls = np.linalg.norm(Ps - t, axis=1)
                u = (Ps - t) / Ls[:, None]                 # unit ray directions
                ns = int(np.ceil(2 * trunc / res)) + 1     # one sample per voxel step
                off = np.linspace(-trunc, trunc, ns)
                g = np.exp(-off ** 2 / (2 * sig ** 2))
                g /= g.sum()                               # MASS CONSERVED: sums to 1
                pts = (Ps[:, None, :] + u[:, None, :] * off[None, :, None])
                evox = np.floor(pts.reshape(-1, 3) / res).astype(np.int64)
                ew = np.repeat(w[static] * c.l_occ, ns) * np.tile(g, len(Ps))
            # free carving stops trunc short of the endpoint: inside the uncertain
            # band nothing may assert free.
            L_all = np.linalg.norm(P - t, axis=1)
            sc = np.clip((L_all - trunc) / L_all, 0.0, 1.0)
            P_free = t + (P - t) * sc[:, None]
            live = sc > 0
            if live.any():
                ray, fvox = _dda_batch(t, P_free[live], res)
                fw = w[live][ray] * c.l_free
            else:
                fvox = np.zeros((0, 3), np.int64); fw = np.zeros(0)

        vox = np.vstack([fvox, evox])
        dlo = np.concatenate([fw, ew])
        # scatter-add weighted log-odds per unique voxel via a 1-D key
        B = 1 << 18
        D = 1 << 19
        key = ((vox[:, 0] + B) * D + (vox[:, 1] + B)) * D + (vox[:, 2] + B)
        uk, inv = np.unique(key, return_inverse=True)
        acc = np.bincount(inv, weights=dlo)
        # Optional per-cell attribution: evidence-weighted mean tr(Sigma) of the
        # frames that wrote this cell. Lets tr(Sigma) be correlated against
        # per-cell outcomes (thousands of points) instead of one point per
        # condition. Off by default -- it costs a dict update per frame.
        if self._attr is not None:
            aw = np.bincount(inv, weights=np.abs(dlo))
            num, den = self._attr
            for k_, a_ in zip(uk.tolist(), aw.tolist()):
                num[k_] = num.get(k_, 0.0) + a_ * float(tr_sigma_pos)
                den[k_] = den.get(k_, 0.0) + a_
        iz = (uk % D) - B; uk //= D
        iy = (uk % D) - B; uk //= D
        ix = uk - B
        centers = (np.stack([ix, iy, iz], axis=1) + 0.5) * res
        for ctr, val in zip(centers, acc):
            self.tree.updateNode(ctr, float(val), True)        # direct log-odds
        self.n_frames += 1
        self.n_points += len(P)
        return len(P)

    def finalize(self):
        self.tree.updateInnerOccupancy()
        return self.tree

    def sigma_u_stats(self):
        """(median, p90) of sigma_u over every ray of every frame, or (nan, nan)."""
        if not self._su:
            return float('nan'), float('nan')
        a = np.concatenate(self._su)
        return float(np.median(a)), float(np.percentile(a, 90))

    def trunc_stats(self):
        """Mean truncated cells per ray, averaged over frames (free_trunc only)."""
        if not self._trunc:
            return float('nan')
        return float(np.mean([t.mean() for t in self._trunc if len(t)]))

    def sigma_attribution(self, pts):
        """(N,) evidence-weighted mean tr(Sigma_pos) for each of `pts`, or NaN
        where the cell was never written. Requires track_sigma_attribution."""
        if self._attr is None:
            raise RuntimeError("set OccCfg.track_sigma_attribution=True first")
        num, den = self._attr
        res = self.cfg.resolution
        B = 1 << 18
        D = 1 << 19
        v = np.floor(np.asarray(pts, float) / res).astype(np.int64)
        keys = ((v[:, 0] + B) * D + (v[:, 1] + B)) * D + (v[:, 2] + B)
        out = np.full(len(keys), np.nan)
        for i, k_ in enumerate(keys.tolist()):
            d_ = den.get(k_)
            if d_:
                out[i] = num[k_] / d_
        return out

    # -- Step G: classification / export ----------------------------------
    def classify_points(self):
        """Return (occ (No,3), free (Nf,3)) voxel-center arrays by log-odds.

        occupied: l > +tau_occ; free: l < -tau_free; unknown otherwise (not
        returned). Unknown is never conflated with free (proposal §4.8).

        Must be called BEFORE write_bt (see its docstring).
        """
        if self._written:
            raise RuntimeError(
                "classify_points() after write_bt(): the tree has been collapsed "
                "to max-likelihood and pruned, so log-odds and cell counts are "
                "no longer meaningful. Classify first, export last.")
        c = self.cfg
        occ, free = [], []
        self.tree.updateInnerOccupancy()
        for it in self.tree.begin_leafs():
            lo = it.getValue()               # log-odds of the leaf
            coord = it.getCoordinate()
            if lo > c.tau_occ:
                occ.append(coord)
            elif lo < -c.tau_free:
                free.append(coord)
        occ = np.array(occ) if occ else np.empty((0, 3))
        free = np.array(free) if free else np.empty((0, 3))
        return occ, free

    def write_bt(self, path):
        """Export the binary .bt map. DESTRUCTIVE and therefore TERMINAL.

        OctoMap's writeBinary converts the tree to its maximum-likelihood estimate
        (every log-odds collapses to the min/max clamp) and prunes it (uniform
        octants merge into single coarse leaves, so leaves no longer correspond
        one-to-one with cells at ``resolution``). Anything that reads log-odds or
        counts cells must run BEFORE this call -- classify_points() enforces that.
        """
        self.tree.updateInnerOccupancy()
        self.tree.writeBinary(path.encode() if isinstance(path, str) else path)
        self._written = True
