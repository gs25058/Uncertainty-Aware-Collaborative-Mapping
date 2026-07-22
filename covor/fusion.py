"""CoVOR-SLAM multi-agent visual-range fusion (paper Sec. II-C).

Builds a factor graph over per-keyframe metric camera poses (Pose3) and scales,
initialised by aligning each robot's up-to-scale VO trajectory to the world frame
(the paper's SL1L2 / initial-scale step, here seeded from mocap), then constrained
by VO odometry, scale random-walk, and UWB inter-agent + anchor range factors, and
solved with Levenberg-Marquardt.
"""
from dataclasses import dataclass, field
import numpy as np
import gtsam

from . import data as D
from . import factors as F


@dataclass
class Cfg:
    sigma_odo_rot: float = 0.05      # rad, VO relative-rotation noise
    sigma_odo_trans: float = 0.05    # m,  VO relative-translation noise (metric)
    sigma_scale_walk: float = 0.02   # per-keyframe scale drift
    sigma_prior_rot: float = 0.1
    sigma_prior_trans: float = 0.3   # frame-alignment prior strength
    sigma_scale_prior: float = 0.3
    max_odo_gap: float = 1.0         # s; skip odometry across map breaks
    range_tol: float = 0.05          # s; keyframe<->range time association
    range_subsample: int = 1         # keep every Nth associated range
    robust: bool = True
    prior_every: int = 0             # extra weak pose priors every N KFs (0=only first)
    use_ranges: bool = True          # ablation: disable all UWB range factors
    use_anchor: bool = True          # ablation: disable anchor range factors
    use_inter: bool = True           # ablation: disable inter-agent range factors
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
    def __init__(self, seq, robot, k):
        self.name = robot
        self.k = k
        self.vo = D.load_vo(seq, robot)
        self.mocap = D.load_mocap(seq, robot)
        self.height = D.load_height(seq, robot)
        self.poses_vo = []   # Pose3 in Lk
        self.t = []          # keyframe timestamps
        if self.vo is not None:
            for _, r in self.vo.iterrows():
                self.poses_vo.append(F.pose_from_quat(
                    [r.x, r.y, r.z], [r.qx, r.qy, r.qz, r.qw]))
                self.t.append(float(r.t))
        self.t = np.array(self.t)
        self.init_world = []   # Pose3 in G (initial guess)
        self.s0 = 1.0

    def n(self):
        return len(self.t)

    def align_to_world(self):
        """Initialise world poses + scale by Umeyama-aligning VO to mocap."""
        src = np.array([p.translation() for p in self.poses_vo])
        dst = np.array([D.mocap_position_at(self.mocap, t) for t in self.t])
        s, R, tvec = umeyama_sim3(src, dst)
        self.s0 = float(abs(s)) if abs(s) > 1e-6 else 1.0
        Ralign = gtsam.Rot3(R)
        for p in self.poses_vo:
            Rw = Ralign.compose(p.rotation())
            tw = s * (R @ p.translation()) + tvec
            self.init_world.append(gtsam.Pose3(Rw, tw))


class CoVOR:
    def __init__(self, seq, cfg: Cfg = None):
        self.seq = seq
        self.cfg = cfg or Cfg()
        self.robots = []
        for k, name in enumerate(D.ROBOTS):
            rb = Robot(seq, name, k)
            self.robots.append(rb)
        self.anchors = D.load_anchors(seq)
        self.ranges = D.load_ranges(seq)
        self.arms = D.load_tag_arms()   # per-tag body-frame moment arms (tags.yaml)
        self.stats = {}

    # -- keyframe association -------------------------------------------------
    def _assoc(self, k, t):
        rb = self.robots[k]
        if rb.n() == 0:
            return None
        i = int(np.abs(rb.t - t).argmin())
        if abs(rb.t[i] - t) > self.cfg.range_tol:
            return None
        return i

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

        # variables + priors + odometry + scale walk
        for rb in self.robots:
            k = rb.k
            for i in range(rb.n()):
                values.insert(F.X(k, i), rb.init_world[i])
                values.insert(F.Sc(k, i), rb.s0)
            if rb.n() == 0:
                continue
            graph.add(F.pose_prior(k, 0, rb.init_world[0],
                                   c.sigma_prior_rot, c.sigma_prior_trans))
            graph.add(F.scale_prior(k, 0, rb.s0, c.sigma_scale_prior))
            constrained.add(F.X(k, 0))
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
                graph.add(F.scale_walk_factor(k, i, c.sigma_scale_walk))

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
        for cnt, (_, r) in enumerate(self.ranges.iterrows() if c.use_ranges else []):
            if c.range_subsample > 1 and (cnt % c.range_subsample):
                continue
            t = float(r.timestamp)
            fk = D.robot_of_tag(int(r.from_id))
            if fk is None:
                continue
            ka = int(fk[-1]) - 1
            ia = self._assoc(ka, t)
            if ia is None:
                continue
            if c.use_gt_range:
                z = float(r["gt_range"]); sig = c.gt_range_sigma
            else:
                z = float(r["range"]) - (c.range_bias if c.bias_mode == "const" else 0.0)
                sig = max(float(r["std"]), c.range_sigma_floor)
            ma = c.use_moment_arm
            la = self.arms.get(int(r.from_id)) if ma else None
            if r["kind"] == "anchor":
                if not c.use_anchor:
                    continue
                aid = int(r.to_id)
                if aid not in self.anchors:
                    continue
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
                ib = self._assoc(kb, t)
                if ib is None:
                    continue
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
            scales_init={rb.name: round(rb.s0, 4) for rb in self.robots},
            n_inter_range=n_inter, n_anchor_range=n_anchor, n_height=n_height,
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
