"""GTSAM CustomFactor builders for CoVOR-SLAM.

Each keyframe state is a metric camera pose ``Pose3`` (translation = metric
position in world G) plus a scalar scale variable, which together parameterise the
paper's 7-DoF Sim(3) state. Monocular VO supplies up-to-scale relative motion; the
per-keyframe scale converts it to metric, and a random-walk factor on scale models
scale drift. UWB range factors (inter-agent and anchor) inject metric information.

Antenna moment arms are treated as pre-compensated, so range residuals use the
camera translation directly (as stated in the paper, Sec. II-B).

All Jacobians are computed by finite differences on the Lie-group tangent
(Pose3.retract) and on the scalar, which is robust and adequate for LM here.
"""
import numpy as np
import gtsam
from gtsam import Pose3, Rot3

EPS = 1e-6


def X(k, i):
    """Key for robot k, keyframe i pose."""
    return gtsam.symbol(chr(ord('a') + k), i)


def Sc(k, i):
    """Key for robot k, keyframe i scale."""
    return gtsam.symbol(chr(ord('p') + k), i)


def Bias(k):
    """Key for robot k's online range-bias scalar (antenna moment-arm offset)."""
    return gtsam.symbol('t', k)


def Anchor(aid):
    """Key for a fixed UWB anchor, represented as a strongly-priored Pose3 so that
    the native C++ RangeFactorPose3 (analytic, no Python callback) can be used."""
    return gtsam.symbol('z', aid)


def pose_from_quat(pos, quat_xyzw) -> Pose3:
    qx, qy, qz, qw = quat_xyzw
    return Pose3(Rot3.Quaternion(qw, qx, qy, qz), np.asarray(pos, dtype=float))


def _num_jac_pose(fn, pose, m):
    """d(fn)/d(pose tangent), shape (m, 6), via retract finite differences."""
    J = np.zeros((m, 6))
    r0 = fn(pose)
    for k in range(6):
        dv = np.zeros(6); dv[k] = EPS
        J[:, k] = (fn(pose.retract(dv)) - r0) / EPS
    return J


# ----------------------------------------------------------------------------
# VO odometry factor: connects poses i, i+1 of robot k and scale_i.
#   predicted T_{i+1} = T_i * Pose3(dR_vo, s_i * dt_vo)
#   residual = localCoordinates(predicted^{-1} * T_{i+1})   (6-vector)
# ----------------------------------------------------------------------------
def odometry_factor(k, i, dpose_vo: Pose3, sigma_rot, sigma_trans):
    dR = dpose_vo.rotation()
    dt = dpose_vo.translation()
    keys = [X(k, i), X(k, i + 1), Sc(k, i)]
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_rot] * 3 + [sigma_trans] * 3))

    def err(this, values, H):
        Ti = values.atPose3(keys[0])
        Tj = values.atPose3(keys[1])
        s = values.atDouble(keys[2])

        # residual = Log(pred^{-1} * Tj), zero when the predicted pose matches Tj
        def res(Ti_, Tj_, s_):
            dT = Pose3(dR, s_ * dt)
            pred = Ti_.compose(dT)
            return Pose3.localCoordinates(pred, Tj_)

        r = res(Ti, Tj, s)
        if H is not None:
            H[0] = _num_jac_pose(lambda p: res(p, Tj, s), Ti, 6)
            H[1] = _num_jac_pose(lambda p: res(Ti, p, s), Tj, 6)
            ds = (res(Ti, Tj, s + EPS) - r) / EPS
            H[2] = ds.reshape(6, 1)
        return r

    return gtsam.CustomFactor(noise, keys, err)


# ----------------------------------------------------------------------------
# Scale random-walk factor: s_{i+1} - s_i ~ N(0, sigma^2). Models scale drift.
# ----------------------------------------------------------------------------
def scale_walk_factor(k, i, sigma):
    keys = [Sc(k, i), Sc(k, i + 1)]
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

    def err(this, values, H):
        a = values.atDouble(keys[0]); b = values.atDouble(keys[1])
        if H is not None:
            H[0] = np.array([[-1.0]]); H[1] = np.array([[1.0]])
        return np.array([b - a])

    return gtsam.CustomFactor(noise, keys, err)


# ----------------------------------------------------------------------------
# Analytic range Jacobian.
#   r = ||p_a - target|| - z            (target = other pose's translation or anchor)
#   p = pose.translation();  under GTSAM's retract convention  dp/dxi = [0_3 | R],
#   so  dr/dxi = u^T . [0_3 | R] = [0, 0, 0,  (u^T R)]   with  u = (p_a - target)/||.||.
# Only the translation half of the tangent enters; rotation columns are exactly 0.
# ----------------------------------------------------------------------------
def _range_jac(pose, unit_dir):
    """(1,6) Jacobian of a range residual w.r.t. this pose's retract tangent."""
    row = np.zeros((1, 6))
    row[0, 3:] = unit_dir @ pose.rotation().matrix()   # u^T R
    return row


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def _antenna_range_jac(pose, l, unit_dir):
    """(1,6) Jacobian of a range residual w.r.t. this pose's retract tangent, where
    the ranging antenna sits at q = p + R*l (l = body-frame moment arm).

    Under GTSAM's Pose3 retract (T -> T*Exp(xi), xi = [omega(3); rho(3)]):
        dq/dxi = R * [ -[l]_x | I3 ]
    so with u = unit direction (q - target)/||.||:
        dr/dxi = u^T R [ -[l]_x | I3 ]  = [ -u^T R [l]_x ,  u^T R ]
    Reduces to _range_jac (rotation cols = 0) when l = 0.
    """
    R = pose.rotation().matrix()
    uR = unit_dir @ R                       # (3,)
    row = np.zeros((1, 6))
    row[0, :3] = -uR @ _skew(l)             # rotation part
    row[0, 3:] = uR                         # translation part
    return row


# ----------------------------------------------------------------------------
# Moment-arm-aware range factors. The UWB antenna is offset from the camera/body
# origin by a known body-frame moment arm l (config/uwb/tags.yaml). MILUV's
# gt_range is the true ANTENNA-to-antenna distance (verified sub-mm), so the range
# residual must use q = p + R*l, not the body/camera position p. Omitting l injects
# an orientation-dependent error of ~0.15 m (std) into every range factor.
# ----------------------------------------------------------------------------
def ma_inter_range_factor(ka, ia, la, kb, ib, lb, z, sigma, robust=True,
                          huber_k=1.345, bias_key=None):
    la = np.asarray(la, float); lb = np.asarray(lb, float)
    keys = [X(ka, ia), X(kb, ib)] + ([bias_key] if bias_key is not None else [])
    noise = _robust_noise(sigma, robust, huber_k)

    def err(this, values, H):
        Ta = values.atPose3(keys[0]); Tb = values.atPose3(keys[1])
        b = values.atDouble(keys[2]) if bias_key is not None else 0.0
        qa = Ta.translation() + Ta.rotation().matrix() @ la
        qb = Tb.translation() + Tb.rotation().matrix() @ lb
        dvec = qa - qb
        dist = float(np.linalg.norm(dvec))
        if H is not None:
            u = dvec / max(dist, EPS)
            H[0] = _antenna_range_jac(Ta, la, u)
            H[1] = _antenna_range_jac(Tb, lb, -u)
            if bias_key is not None:
                H[2] = np.array([[1.0]])
        return np.array([dist - z + b])

    return gtsam.CustomFactor(noise, keys, err)


def ma_anchor_range_factor(ka, ia, la, p_anchor, z, sigma, robust=True,
                           huber_k=1.345, bias_key=None):
    la = np.asarray(la, float); p = np.asarray(p_anchor, float)
    keys = [X(ka, ia)] + ([bias_key] if bias_key is not None else [])
    noise = _robust_noise(sigma, robust, huber_k)

    def err(this, values, H):
        Ta = values.atPose3(keys[0])
        b = values.atDouble(keys[1]) if bias_key is not None else 0.0
        qa = Ta.translation() + Ta.rotation().matrix() @ la
        dvec = qa - p
        dist = float(np.linalg.norm(dvec))
        if H is not None:
            u = dvec / max(dist, EPS)
            H[0] = _antenna_range_jac(Ta, la, u)
            if bias_key is not None:
                H[1] = np.array([[1.0]])
        return np.array([dist - z + b])

    return gtsam.CustomFactor(noise, keys, err)


def _robust_noise(sigma, robust, huber_k):
    base = gtsam.noiseModel.Isotropic.Sigma(1, max(sigma, 1e-2))
    if not robust:
        return base
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k), base)


def native_range_factor(key_a, key_b, z, sigma, robust=True, huber_k=1.345):
    """GTSAM's native C++ RangeFactorPose3: residual ||t_a - t_b|| - z with an
    analytic Jacobian evaluated entirely in C++ (no Python callback -> fast).
    Used for both inter-agent ranges and anchor ranges (anchor = fixed Pose3)."""
    return gtsam.RangeFactorPose3(key_a, key_b, z, _robust_noise(sigma, robust, huber_k))


def anchor_pin(aid, p_anchor):
    """A strong prior pinning an anchor Pose3 variable at its known position."""
    pose = gtsam.Pose3(gtsam.Rot3(), np.asarray(p_anchor, dtype=float))
    noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)
    return gtsam.PriorFactorPose3(Anchor(aid), pose, noise), pose


# ----------------------------------------------------------------------------
# Inter-agent range factor: || t_a - t_b || - z   (analytic Jacobian).
# Optional bias_key adds an online range-bias scalar b: residual += b.
# ----------------------------------------------------------------------------
def inter_range_factor(ka, ia, kb, ib, z, sigma, robust=True, huber_k=1.345,
                       bias_key=None):
    keys = [X(ka, ia), X(kb, ib)] + ([bias_key] if bias_key is not None else [])
    noise = _robust_noise(sigma, robust, huber_k)

    def err(this, values, H):
        Ta = values.atPose3(keys[0]); Tb = values.atPose3(keys[1])
        b = values.atDouble(keys[2]) if bias_key is not None else 0.0
        dvec = Ta.translation() - Tb.translation()
        dist = float(np.linalg.norm(dvec))
        if H is not None:
            u = dvec / max(dist, EPS)
            H[0] = _range_jac(Ta, u)
            H[1] = _range_jac(Tb, -u)
            if bias_key is not None:
                H[2] = np.array([[1.0]])
        return np.array([dist - z + b])

    return gtsam.CustomFactor(noise, keys, err)


# ----------------------------------------------------------------------------
# Anchor range factor: || t_a - p_anchor || - z   (p_anchor fixed, analytic Jac).
# Optional bias_key adds an online range-bias scalar b: residual += b.
# ----------------------------------------------------------------------------
def anchor_range_factor(ka, ia, p_anchor, z, sigma, robust=True, huber_k=1.345,
                        bias_key=None):
    keys = [X(ka, ia)] + ([bias_key] if bias_key is not None else [])
    p = np.asarray(p_anchor, dtype=float)
    noise = _robust_noise(sigma, robust, huber_k)

    def err(this, values, H):
        Ta = values.atPose3(keys[0])
        b = values.atDouble(keys[1]) if bias_key is not None else 0.0
        dvec = Ta.translation() - p
        dist = float(np.linalg.norm(dvec))
        if H is not None:
            u = dvec / max(dist, EPS)
            H[0] = _range_jac(Ta, u)
            if bias_key is not None:
                H[1] = np.array([[1.0]])
        return np.array([dist - z + b])

    return gtsam.CustomFactor(noise, keys, err)


# ----------------------------------------------------------------------------
# Height factor (downward laser altimeter): residual = p_z - h_meas.
#   p_z = e_z^T p,  dp/dxi = [0 | R]  ->  dr/dxi = [0, 0, 0,  e_z^T R]
#   (e_z^T R is the third row of the pose's rotation matrix).
# ----------------------------------------------------------------------------
def height_factor(k, i, h_meas, sigma, robust=False, huber_k=1.345):
    key = X(k, i)
    noise = _robust_noise(sigma, robust, huber_k)

    def err(this, values, H):
        Ta = values.atPose3(key)
        if H is not None:
            row = np.zeros((1, 6))
            row[0, 3:] = Ta.rotation().matrix()[2, :]     # e_z^T R
            H[0] = row
        return np.array([Ta.translation()[2] - h_meas])

    return gtsam.CustomFactor(noise, [key], err)


def bias_prior(k, b0, sigma):
    """Weak prior anchoring an online range-bias variable near b0."""
    key = Bias(k)
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

    def err(this, values, H):
        b = values.atDouble(key)
        if H is not None:
            H[0] = np.array([[1.0]])
        return np.array([b - b0])

    return gtsam.CustomFactor(noise, [key], err)


# ----------------------------------------------------------------------------
# Pose prior (gauge / frame-alignment prior, like the paper's phi_pri).
# ----------------------------------------------------------------------------
def pose_prior(k, i, pose0: Pose3, sigma_rot, sigma_trans):
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_rot] * 3 + [sigma_trans] * 3))
    return gtsam.PriorFactorPose3(X(k, i), pose0, noise)


def scale_prior(k, i, s0, sigma):
    key = Sc(k, i)
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

    def err(this, values, H):
        s = values.atDouble(key)
        if H is not None:
            H[0] = np.array([[1.0]])
        return np.array([s - s0])

    return gtsam.CustomFactor(noise, [key], err)
