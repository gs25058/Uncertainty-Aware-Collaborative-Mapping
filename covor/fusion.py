"""CoVOR-SLAM multi-agent visual-range fusion (paper Sec. II-C).

Builds a pure SE(3) factor graph over per-node body(IMU) poses (Pose3), initialised
by rigidly aligning each robot's VIO trajectory to the world frame, then constrained
by VIO odometry (SE(3) between-factors) and UWB inter-agent + anchor range factors,
and solved with Levenberg-Marquardt.

The VINS-Fusion front-end is metric and gravity-aligned, so there is no scale to
estimate: the per-node scale variable, its prior and its random-walk factor are
gone (Sim(3) version at commit a0a5e07). Because roll/pitch are observed by gravity
throughout, the only free gauge left is yaw + position.
"""
from dataclasses import dataclass, field
import numpy as np
import gtsam

from . import data as D
from . import factors as F


@dataclass
class Cfg:
    # VIO relative-pose noise, MEASURED against mocap at the 7.49 Hz node density
    # (not carried over from the mono pipeline, where 0.05/0.05 covered a much
    # sparser, up-to-scale front-end). Per-axis RMS of the relative-pose residual,
    # pooled over the 3 zigzag robots.
    #
    # Measured on CLEANED mocap (data.load_mocap now applies MILUV's gap/outlier
    # rules), so no hand-written outlier filter is involved. The earlier values --
    # 0.0045 / 0.0131, from raw mocap plus a manual ">20 deg mocap step" exclusion --
    # agree to within 8 %, which is the check that the loader does what the hand
    # filter did. On raw mocap with no filter at all the per-robot rotation sigma
    # scattered as 0.094 / 0.124 / 0.013 rad; cleaned, it is 0.0134 / 0.0121 / 0.0115.
    sigma_odo_rot: float = 0.0124    # rad (0.71 deg); mono default was 0.05
    sigma_odo_trans: float = 0.0041  # m; mono default was 0.05
    sigma_prior_rot: float = 0.1
    sigma_prior_trans: float = 0.3   # frame-alignment prior strength (legacy mode)

    # --- gauge fixing (proposal §2.3-다) ---
    # "component": ONE TIGHT prior per connected component of the inter-range graph,
    #   and only for components no anchor grounds. This is what §2.3-다 describes
    #   ("드론1의 첫 자세를 기준에 고정") generalised to the ablation ladder, where
    #   0 or 1 pairs leave more than one component -- a prior on drone 1 alone would
    #   leave the others singular.
    # "per_robot": the legacy behaviour -- a WEAK prior on every robot's first pose.
    #   Do not use it anchor-free: at sigma_trans = 0.3 m that prior carries
    #   tr = 0.27 m^2, comparable to the tr(Sigma) the §4.9 experiment is trying to
    #   measure, so it floors every condition and flattens the very effect under
    #   test. It also hands each robot its own mocap-derived absolute reference,
    #   which is the job the inter-agent ranges are supposed to do.
    # The prior only removes the unobservable 4 DoF (yaw + position); evaluation
    # aligns anyway, so its value defines a frame rather than adding information --
    # which is why it can and should be tight.
    gauge_mode: str = "component"
    sigma_gauge_rot: float = 1e-3
    sigma_gauge_trans: float = 1e-3

    # Where the gauge prior's MEAN comes from, and how many of them there are.
    # PREREG_synth_gauge.md; measured in results/synth_room909/gauge_probe_seed0.json.
    #
    # "umeyama" (default, UNCHANGED behaviour): the mean is the robot's first VIO
    #   pose mapped through a WHOLE-TRAJECTORY umeyama fit to mocap, and one prior
    #   is placed per connected component of the inter-range graph. Both of those
    #   make the drone-count ladder confounded: the number of priors then varies
    #   with the number of UWB pairs (3 / 2 / 1 for the 0 / 1 / 3-pair conditions),
    #   so condition A hands every robot its own absolute reference while condition
    #   C has one. Measured effect on the pooled unaligned position error:
    #   0.166 -> 0.489 m, which swamps the collaboration effect being measured.
    #
    # "first_pose": the mean is the robot's FIRST pose only (yaw + position, the
    #   same 4 DoF init_yaw_only selects), and EVERY robot gets one prior in every
    #   condition. Physically: the drones take off together from known positions,
    #   so the absolute reference is identical across conditions and the ranges can
    #   only correct drift accumulated after take-off. Changing the mean alone is
    #   worth 0.018 m; the count is what matters. Both are changed here because
    #   the whole-trajectory fit also leaks ground truth into the gauge.
    gauge_init: str = "umeyama"

    # Gravity (roll/pitch) prior on every node -- see factors.gravity_prior for why
    # it is required rather than optional. sigma is MEASURED on cleaned mocap: the
    # VIO tilt residual is Rayleigh in magnitude, so sigma = median/1.1774 gives
    # 0.741 / 0.703 / 0.516 deg on the three zigzag robots -> 0.65 deg pooled
    # (raw mocap + hand filter gave 0.68 deg, i.e. 4 % apart).
    # The tail is heavier than Gaussian (p90/median 2.17-2.82 vs 1.82), and the
    # native factor will not accept a robust kernel, so treat this sigma as
    # describing the bulk and not the tail.
    use_gravity_prior: bool = True
    sigma_tilt: float = 0.0114       # rad (0.65 deg), per axis
    frontend: str = "vins"           # "vins" (metric SE(3)) | "orb" (legacy reader)
    vins_stride: int = D.VINS_STRIDE  # node density; see data.VINS_STRIDE
    init_yaw_only: bool = True       # constrain the L_k->G init to yaw+position
                                     # (4-DoF): both frames are gravity-aligned, so
                                     # a full SO(3) fit adds roll/pitch freedom that
                                     # is not physically there.
    max_odo_gap: float = 1.0         # s; skip odometry across map breaks
    range_tol: float = 0.07          # s; node<->range time association. Half the
                                     # 7.49 Hz node spacing is 0.067 s, so this
                                     # keeps ~all ranges; the timing error it admits
                                     # is charged to sigma via range_motion_sigma.
    range_motion_sigma: bool = True  # inflate range sigma by the distance moved
                                     # within the association window:
                                     # sigma_eff^2 = sigma^2 + sum_k (v_k * dt_k)^2
                                     # using VINS's own velocity (no ground truth).
    range_subsample: int = 1         # keep every Nth associated range
    robust: bool = True
    prior_every: int = 0             # extra weak pose priors every N KFs (0=only first)
    use_ranges: bool = True          # ablation: disable all UWB range factors
    use_anchor: bool = True          # ablation: disable anchor range factors
    use_inter: bool = True           # ablation: disable inter-agent range factors

    # --- proposal §4.9 collaboration ladder: these change the GRAPH, not the output ---
    # inter_pairs: which robot-index pairs contribute inter-agent ranges. None = all
    #   C(N,2) pairs. The proposal's ladder is exactly ((),) -> ((0,1),) ->
    #   ((0,1),(0,2),(1,2)), i.e. C(N,2) = 0 -> 1 -> 3 for 1 -> 2 -> 3 drones.
    # anchor_robots: which robots get anchor ranges. None = all, () = ANCHOR-FREE.
    #   The proposal is anchor-free (§1 constraint, §2.4 objective has no anchor
    #   term); the anchors this pipeline currently uses are a CoVOR-reproduction
    #   inheritance, kept only as a side condition for continuity with earlier numbers.
    inter_pairs: tuple = None
    anchor_robots: tuple = None

    # --- synthetic UWB observation model (scripts/sweep_uwb_noise.py) ---
    # Replaces the measured `range` VALUE with gt_range + controlled error, so the
    # "fusion helps only when VO error > UWB effective accuracy" proposition can be
    # turned from a claim into a curve. TIMESTAMPS AND ASSOCIATION ARE UNTOUCHED --
    # only the value changes. (The ORB-era failure was losing 89.9 % of the UWB
    # association; perturbing the association structure here would confound that.)
    # gt_range is antenna-to-antenna (verified sub-mm), so it is consistent with
    # use_moment_arm=True.
    #   None            -> use the real measurement, as before
    #   dict(sigma=..., bias=bool, outlier=bool, seed=int)
    #     sigma    Gaussian noise sigma [m]
    #     bias     add the measured per-kind bias (anchor +0.0871, inter +0.0035)
    #     outlier  with the measured rate (anchor 8.61 %, inter 2.17 %) add
    #              +U(0.5, 1.8) m -- one-sided because 99.8 % of measured
    #              |e| > 0.5 m outliers are positive
    range_inject: dict = None
    # sigma handed to the range factor. None -> max(csv_std, range_sigma_floor) as
    # before. With synthetic ranges the csv `std` column describes a measurement
    # that no longer exists, so the sweep sets this explicitly.
    range_sigma_override: float = None
    range_sigma_floor: float = 0.3   # realistic UWB noise floor (empirical resid std)
    range_bias: float = 0.14         # systematic offset (antenna moment arm); const-mode value
    bias_mode: str = "const"         # "const" | "online" | "off": antenna-bias handling
    sigma_bias_prior: float = 1.0    # weak prior sigma on the online bias variable
    huber_k: float = 1.0             # robust kernel threshold (sigma units)
    use_gt_range: bool = False       # diagnostic: use mocap gt_range as the range
                                     # measurement (bias forced off) to isolate the
                                     # observation model from real UWB quality
    gt_range_sigma: float = 0.05     # sigma used when use_gt_range is on
    use_moment_arm: bool = False     # model the UWB antenna moment arm (tags.yaml):
                                     # range residual uses q = p + R*l, not p. Fixes
                                     # the ~0.15 m orientation-dependent model error.
    use_height: bool = False         # add downward-laser height factors (vertical obs.)
    height_sigma_floor: float = 0.12  # laser height noise floor (empirical resid std)
    height_huber_k: float = 1.5      # robust kernel for height factors (sigma units)
    height_subsample: int = 1        # keep every Nth keyframe's height factor


def umeyama_yaw(src, dst):
    """Rigid alignment restricted to yaw + translation: dst ~= Rz(psi) src + t.

    Both the VINS init frame and the mocap world frame are gravity-aligned, so the
    L_k -> G transform has only 4 physical DoF. Fitting a full SO(3) would absorb
    real roll/pitch error into the alignment instead of leaving it in the residual.
    Closed form: psi = atan2(sum cross_z, sum dot_xy) on the centred xy components.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    psi = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(),
                     (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R, mu_d - R @ mu_s


def umeyama_rigid(src, dst):
    """Full SO(3) + translation alignment (scale fixed at 1)."""
    _, R, _ = umeyama_sim3(src, dst)
    return R, dst.mean(0) - R @ src.mean(0)


def umeyama_sim3(src, dst):
    """Similarity alignment: find s,R,t with dst ~= s R src + t. src,dst: (N,3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    H = Sc.T @ Dc / len(src)
    U, Dd, Vt = np.linalg.svd(H)
    Sgn = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sgn[2, 2] = -1
    R = (Vt.T @ Sgn @ U.T)
    var_s = (Sc ** 2).sum() / len(src)
    s = np.trace(np.diag(Dd) @ Sgn) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


class Robot:
    def __init__(self, seq, robot, k, cfg=None):
        self.name = robot
        self.k = k
        self.cfg = cfg or Cfg()
        if self.cfg.frontend == "vins":
            self.vo = D.load_vins(seq, robot, self.cfg.vins_stride)
        else:
            self.vo = D.load_vo(seq, robot)
        self.mocap = D.load_mocap(seq, robot)
        self.height = D.load_height(seq, robot)
        self.poses_vo = []   # Pose3 in Lk (body/IMU pose for the VINS front-end)
        self.t = []          # node timestamps (MILUV relative seconds)
        self.v = []          # body speed |v| at each node (VINS estimate), m/s
        if self.vo is not None:
            for _, r in self.vo.iterrows():
                self.poses_vo.append(F.pose_from_quat(
                    [r.x, r.y, r.z], [r.qx, r.qy, r.qz, r.qw]))
                self.t.append(float(r.t))
                self.v.append(float(r.v) if "v" in self.vo.columns else 0.0)
        self.t = np.array(self.t)
        self.v = np.array(self.v)
        self.init_world = []   # Pose3 in G (initial guess)

    def n(self):
        return len(self.t)

    def align_to_world(self):
        """Initialise world poses by rigidly aligning the VIO trajectory to mocap.

        The front-end is metric, so this is a rigid (scale-1) fit, restricted to
        yaw+translation when cfg.init_yaw_only. Note this only sets the INITIAL
        GUESS -- it does not enter the objective, so it is not the silent-GT-leak
        of feeding a mocap-aligned trajectory in as the measurement. Replacing it
        with an anchor-range-derived initialisation is a separate, later step.
        """
        if self.cfg.gauge_init == "first_pose":
            R, tvec = self._first_pose_align()
        else:
            src = np.array([p.translation() for p in self.poses_vo])
            dst = np.array([D.mocap_position_at(self.mocap, t) for t in self.t])
            R, tvec = (umeyama_yaw(src, dst) if self.cfg.init_yaw_only
                       else umeyama_rigid(src, dst))
        self.align_R, self.align_t = R, tvec
        Ralign = gtsam.Rot3(R)
        for p in self.poses_vo:
            Rw = Ralign.compose(p.rotation())
            tw = R @ p.translation() + tvec
            self.init_world.append(gtsam.Pose3(Rw, tw))

    def _first_pose_align(self):
        """L_k -> G from the robot's FIRST pose alone (yaw + position).

        Same 4 DoF ``init_yaw_only`` selects, and the same reference source the
        umeyama path uses (the cleaned mocap) -- only the first sample is read
        instead of the whole track, so the alignment cannot absorb drift. The
        first mocap sample survives the spline cleaner to ~1e-5 m at every rate
        tested (PREREG_synth_gauge.md F4), so this is not sensitive to the
        smoothing that the whole-trajectory fit is.
        """
        i = int(np.abs(self.mocap.timestamp.values - self.t[0]).argmin())
        row = self.mocap.iloc[i]
        R_gt = gtsam.Rot3.Quaternion(float(row.qw), float(row.qx),
                                     float(row.qy), float(row.qz)).matrix()
        R_v0 = self.poses_vo[0].rotation().matrix()
        psi = (np.arctan2(R_gt[1, 0], R_gt[0, 0])
               - np.arctan2(R_v0[1, 0], R_v0[0, 0]))
        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        p_gt = np.array([float(row.x), float(row.y), float(row.z)])
        return R, p_gt - R @ self.poses_vo[0].translation()


class CoVOR:
    def __init__(self, seq, cfg: Cfg = None):
        self.seq = seq
        self.cfg = cfg or Cfg()
        self.robots = []
        for k, name in enumerate(D.ROBOTS):
            rb = Robot(seq, name, k, self.cfg)
            self.robots.append(rb)
        self.anchors = D.load_anchors(seq)
        self.ranges = D.load_ranges(seq)
        self.arms = D.load_tag_arms()   # per-tag body-frame moment arms (tags.yaml)
        self.stats = {}

    # -- synthetic observation model ----------------------------------------
    # Measured on default_3_zigzag_0 from e = range - gt_range (n = 24,787):
    #   anchor  bias +0.0871  std 0.2457  P(|e|>0.5) = 8.61 %
    #   inter   bias +0.0035  std 0.1660  P(|e|>0.5) = 2.17 %
    # 99.8 % of the |e| > 0.5 m tail is positive, magnitudes 0.50-1.78 m.
    _BIAS = {"anchor": 0.0871, "inter": 0.0035}
    _P_OUT = {"anchor": 0.0861, "inter": 0.0217}

    def _inject_range(self, r):
        """gt_range + bias + N(0, sigma) + outlier, per cfg.range_inject.

        Deterministic in (seed, timestamp, from_id, to_id) so a row gets the same
        perturbation regardless of iteration order or how many rows are skipped --
        the sweep's seeds are then reproducible and comparable across conditions.
        """
        c = self.cfg.range_inject
        kind = str(r["kind"])
        z = float(r["gt_range"])
        if c.get("bias"):
            z += self._BIAS[kind]
        key = (int(c.get("seed", 0)), float(r["timestamp"]),
               int(r["from_id"]), int(r["to_id"]))
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        sd = float(c.get("sigma", 0.0))
        if sd > 0:
            z += rng.normal(0.0, sd)
        if c.get("outlier") and rng.random() < self._P_OUT[kind]:
            z += rng.uniform(0.5, 1.8)
        return z

    # -- keyframe association -------------------------------------------------
    def _assoc(self, k, t):
        """Nearest node index to time t, plus the motion-induced position slop
        |v| * |dt| that associating across that gap injects. Returns (None, 0.0)
        outside the tolerance."""
        rb = self.robots[k]
        if rb.n() == 0:
            return None, 0.0
        i = int(np.abs(rb.t - t).argmin())
        dt = abs(rb.t[i] - t)
        if dt > self.cfg.range_tol:
            return None, 0.0
        slop = float(rb.v[i]) * dt if self.cfg.range_motion_sigma else 0.0
        return i, slop

    # -- graph construction ---------------------------------------------------
    def build(self):
        c = self.cfg
        for rb in self.robots:
            if rb.n() > 0:
                rb.align_to_world()

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        constrained = set()   # pose keys touched by odometry or range factors
        n_height = 0
        n_grav = 0
        n_legacy_prior = 0

        # variables + priors + odometry
        for rb in self.robots:
            k = rb.k
            for i in range(rb.n()):
                values.insert(F.X(k, i), rb.init_world[i])
            if rb.n() == 0:
                continue
            if c.gauge_mode == "per_robot":
                graph.add(F.pose_prior(k, 0, rb.init_world[0],
                                       c.sigma_prior_rot, c.sigma_prior_trans))
                constrained.add(F.X(k, 0))
                n_legacy_prior += 1

            # Absolute roll/pitch on EVERY node, from the front-end's gravity
            # observation. Without it the graph retains no absolute tilt at all
            # (only node 0's prior) and the tilt wanders within the odometry
            # chain's random-walk slack. Note this does NOT make a node
            # "constrained" for the weak-prior fallback below: it fixes 2 of the
            # 3 rotational DoF and none of the translational ones.
            if c.use_gravity_prior:
                for i in range(rb.n()):
                    graph.add(F.gravity_prior(
                        k, i, rb.poses_vo[i].rotation().matrix(), c.sigma_tilt))
                    n_grav += 1
            for i in range(rb.n() - 1):
                if c.prior_every and (i % c.prior_every == 0) and i > 0:
                    graph.add(F.pose_prior(k, i, rb.init_world[i],
                                           c.sigma_prior_rot, c.sigma_prior_trans * 3))
                    constrained.add(F.X(k, i))
                gap = rb.t[i + 1] - rb.t[i]
                if gap <= c.max_odo_gap:
                    dpose = rb.poses_vo[i].between(rb.poses_vo[i + 1])
                    graph.add(F.odometry_factor(k, i, dpose,
                                                c.sigma_odo_rot, c.sigma_odo_trans))
                    constrained.add(F.X(k, i)); constrained.add(F.X(k, i + 1))

            # height factors (downward laser altimeter -> vertical observability)
            if c.use_height and rb.height is not None:
                for i in range(rb.n()):
                    if c.height_subsample > 1 and (i % c.height_subsample):
                        continue
                    h = D.height_at(rb.height, rb.t[i], tol=self.cfg.range_tol * 2)
                    if h is None:
                        continue
                    graph.add(F.height_factor(k, i, h, c.height_sigma_floor,
                                              True, c.height_huber_k))
                    constrained.add(F.X(k, i))
                    n_height += 1

        # Bias handling. "const": subtract range_bias from the measurement (native
        # C++ range factors, fast). "online": per-robot scalar bias variable, which
        # needs the custom analytic CustomFactor. "off": no correction (native).
        online = (c.bias_mode == "online")
        bias_used = set()
        anchor_used = {}

        def bias_key_for(k):
            if not online:
                return None
            if k not in bias_used:
                values.insert(F.Bias(k), c.range_bias)
                graph.add(F.bias_prior(k, 0.0, c.sigma_bias_prior))
                bias_used.add(k)
            return F.Bias(k)

        def anchor_key_for(aid):
            if aid not in anchor_used:
                pin, pose = F.anchor_pin(aid, self.anchors[aid])
                values.insert(F.Anchor(aid), pose)
                graph.add(pin)
                anchor_used[aid] = True
            return F.Anchor(aid)

        # range factors
        n_inter = n_anchor = 0
        # normalise the pair list once so (k,k') and (k',k) both match
        _pairset = (None if c.inter_pairs is None
                    else {(min(a, b), max(a, b)) for a, b in c.inter_pairs})
        for cnt, (_, r) in enumerate(self.ranges.iterrows() if c.use_ranges else []):
            if c.range_subsample > 1 and (cnt % c.range_subsample):
                continue
            t = float(r.timestamp)
            fk = D.robot_of_tag(int(r.from_id))
            if fk is None:
                continue
            ka = int(fk[-1]) - 1
            ia, slop_a = self._assoc(ka, t)
            if ia is None:
                continue
            if c.range_inject is not None:
                z = self._inject_range(r)
                sig = (c.range_sigma_override if c.range_sigma_override is not None
                       else max(float(r["std"]), c.range_sigma_floor))
            elif c.use_gt_range:
                z = float(r["gt_range"]); sig = c.gt_range_sigma
            else:
                z = float(r["range"]) - (c.range_bias if c.bias_mode == "const" else 0.0)
                sig = (c.range_sigma_override if c.range_sigma_override is not None
                       else max(float(r["std"]), c.range_sigma_floor))
            ma = c.use_moment_arm
            la = self.arms.get(int(r.from_id)) if ma else None
            if r["kind"] == "anchor":
                if not c.use_anchor:
                    continue
                if c.anchor_robots is not None and ka not in c.anchor_robots:
                    continue
                aid = int(r.to_id)
                if aid not in self.anchors:
                    continue
                sig = float(np.hypot(sig, slop_a))
                if ma:
                    graph.add(F.ma_anchor_range_factor(ka, ia, la, self.anchors[aid],
                                                       z, sig, c.robust, c.huber_k,
                                                       bias_key_for(ka)))
                elif online:
                    graph.add(F.anchor_range_factor(ka, ia, self.anchors[aid], z, sig,
                                                    c.robust, c.huber_k, bias_key_for(ka)))
                else:
                    graph.add(F.native_range_factor(F.X(ka, ia), anchor_key_for(aid),
                                                    z, sig, c.robust, c.huber_k))
                constrained.add(F.X(ka, ia))
                n_anchor += 1
            else:
                if not c.use_inter:
                    continue
                tk = D.robot_of_tag(int(r.to_id))
                if tk is None:
                    continue
                kb = int(tk[-1]) - 1
                if c.inter_pairs is not None and \
                        (min(ka, kb), max(ka, kb)) not in _pairset:
                    continue
                ib, slop_b = self._assoc(kb, t)
                if ib is None:
                    continue
                # both endpoints move during their own association gap
                sig = float(np.sqrt(sig ** 2 + slop_a ** 2 + slop_b ** 2))
                if ma:
                    lb = self.arms.get(int(r.to_id))
                    graph.add(F.ma_inter_range_factor(ka, ia, la, kb, ib, lb, z, sig,
                                                      c.robust, c.huber_k, bias_key_for(ka)))
                elif online:
                    graph.add(F.inter_range_factor(ka, ia, kb, ib, z, sig,
                                                   c.robust, c.huber_k, bias_key_for(ka)))
                else:
                    graph.add(F.native_range_factor(F.X(ka, ia), F.X(kb, ib),
                                                    z, sig, c.robust, c.huber_k))
                constrained.add(F.X(ka, ia)); constrained.add(F.X(kb, ib))
                n_inter += 1

        # --- gauge fixing: one tight prior per ungrounded connected component ---
        # Components come from the inter-range graph. A component that any anchor
        # touches is already grounded in the world frame, so it gets no prior --
        # adding one would fight the anchors with a mocap-derived pose.
        n_gauge, gauge_on = 0, []
        if c.gauge_mode == "component":
            live = [rb.k for rb in self.robots if rb.n() > 0]
            parent = {k: k for k in live}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in (_pairset or set()) if c.inter_pairs is not None else \
                    {(i, j) for i in live for j in live if i < j}:
                if c.use_inter and c.use_ranges and a in parent and b in parent:
                    parent[find(a)] = find(b)
            grounded = set()
            if c.use_ranges and c.use_anchor:
                grounded = set(live if c.anchor_robots is None else c.anchor_robots)
            comps = {}
            for k in live:
                comps.setdefault(find(k), []).append(k)
            if c.gauge_init == "first_pose":
                # One prior per ROBOT, in every condition: the number of absolute
                # references must not vary with the number of UWB pairs, or the
                # ladder measures gauge count instead of collaboration.
                targets = [[k] for k in live]
            else:
                targets = list(comps.values())
            for members in targets:
                if grounded & set(members):
                    continue                       # anchors already fix this frame
                k0 = min(members)                  # proposal: drone 1 of the group
                graph.add(F.pose_prior(k0, 0, self.robots[k0].init_world[0],
                                       c.sigma_gauge_rot, c.sigma_gauge_trans))
                constrained.add(F.X(k0, 0))
                n_gauge += 1
                gauge_on.append(k0)

        # weak priors on any pose left unconstrained (isolated across map breaks),
        # to guarantee a well-posed elimination without biasing constrained poses
        n_weak = 0
        for rb in self.robots:
            for i in range(rb.n()):
                if F.X(rb.k, i) not in constrained:
                    graph.add(F.pose_prior(rb.k, i, rb.init_world[i], 5.0, 10.0))
                    n_weak += 1

        self.bias_used = bias_used
        self.stats = dict(
            keyframes={rb.name: rb.n() for rb in self.robots},
            frontend=c.frontend, node_stride=c.vins_stride,
            inter_pairs=("all" if c.inter_pairs is None else tuple(c.inter_pairs)),
            anchor_robots=("all" if c.anchor_robots is None else tuple(c.anchor_robots)),
            n_inter_range=n_inter, n_anchor_range=n_anchor, n_height=n_height,
            n_gravity_prior=n_grav,
            gauge_mode=c.gauge_mode, gauge_on=tuple(gauge_on),
            n_gauge_prior=(n_legacy_prior if c.gauge_mode == "per_robot" else n_gauge),
            n_weak_priors=n_weak, n_bias=len(bias_used), bias_mode=c.bias_mode,
            n_factors=graph.size(), n_vars=values.size())
        self.graph, self.values = graph, values
        return self

    def biases(self, values):
        """Converged online range-bias per robot (empty unless bias_mode='online')."""
        return {D.ROBOTS[k]: round(values.atDouble(F.Bias(k)), 4)
                for k in sorted(getattr(self, "bias_used", set()))}

    def optimize(self, max_iter=100, verbose=True):
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(max_iter)
        if verbose:
            params.setVerbosityLM("SUMMARY")
        opt = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)
        self.result = opt.optimize()
        self.stats["initial_error"] = self.graph.error(self.values)
        self.stats["final_error"] = self.graph.error(self.result)
        return self.result

    # -- trajectory extraction ------------------------------------------------
    def trajectory(self, values, k):
        rb = self.robots[k]
        out = []
        for i in range(rb.n()):
            T = values.atPose3(F.X(k, i))
            out.append([rb.t[i], *T.translation()])
        return np.array(out)
