"""GTSAM factor builders for CoVOR-SLAM.

Each node is a 6-DoF body(IMU) pose ``Pose3`` in the world frame G. The
VINS-Fusion stereo+IMU front-end is metric and gravity-aligned, so relative
motion needs no scale correction: odometry is a plain SE(3) between-factor and
the graph is pure SE(3). (The mono front-end's 7-DoF Sim(3) state -- a per-node
scale variable with a random-walk factor -- was removed in that transition; it is
recoverable at commit a0a5e07 together with the ORB-SLAM3 comparison group.)

UWB range factors (inter-agent and anchor) inject the absolute information. Their
Jacobians are analytic; the moment-arm variants account for the tag sitting off
the body origin, which is what makes yaw weakly observable from ranges.
"""
import numpy as np
import gtsam
from gtsam import Pose3, Rot3

EPS = 1e-6


def X(k, i):
    """Key for robot k, keyframe i pose."""
    return gtsam.symbol(chr(ord('a') + k), i)


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


# ----------------------------------------------------------------------------
# VIO odometry factor: connects poses i, i+1 of robot k.
#   residual = Log(dpose_vio^{-1} * (T_i^{-1} T_{i+1}))    (6-vector)
# The front-end is metric, so this is GTSAM's native C++ BetweenFactorPose3:
# analytic Jacobians, no Python callback (the mono version needed a CustomFactor
# only to carry the scale variable).
# ----------------------------------------------------------------------------
def odometry_factor(k, i, dpose_vio: Pose3, sigma_rot, sigma_trans):
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_rot] * 3 + [sigma_trans] * 3))
    return gtsam.BetweenFactorPose3(X(k, i), X(k, i + 1), dpose_vio, noise)


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
# Gravity (attitude) prior: constrains roll/pitch ONLY, leaving yaw free.
#
# Why this has to exist. The VIO front-end observes the gravity direction with the
# accelerometer, which is why its tilt error is ~0.7 deg and flat. But converting
# that trajectory into relative-pose between-factors DISCARDS the absolute part:
# without this factor the only absolute orientation constraint in the whole graph
# is the single pose_prior on each robot's first node, and the relative chain's
# random walk (sigma_odo_rot per step) opens up ~7.5 deg of tilt slack in 13 s and
# ~35 deg over a full run. Measured consequence: fused tilt degraded from 0.6-0.9
# deg to 3.0-4.1 deg. UWB cannot take up the slack -- with a 0.231 m tag moment arm
# a 5 deg tilt moves the antenna 0.020 m, well under the 0.05 m range noise floor,
# so ranges are effectively blind to tilt.
#
# Because nothing else in the graph observes tilt, this prior is the ONLY tilt
# information present; its sigma therefore just sets how far the range factors can
# drag tilt away from the front-end's estimate, and any such drag is spurious.
# Erring tight is the safe direction.
#
# GTSAM's native Pose3AttitudeFactor gives a 2-dim residual with analytic
# Jacobians: it enforces R * bRef ~= nZ. With bRef = the gravity direction in the
# body frame as VIO measured it, and nZ = world up, the residual is exactly the
# tilt disagreement, and rotation about gravity (yaw) leaves it unchanged.
#
# Consistency: align_to_world uses a yaw-only rotation Rz(psi), and
# Rz(psi) * R_vio * (R_vio^T e_z) = Rz(psi) e_z = e_z, so the initial guess starts
# at exactly zero residual. The factor adds no new frame assumption beyond the one
# the yaw-only alignment already makes (both frames gravity-aligned).
# ----------------------------------------------------------------------------
_E_Z = np.array([0.0, 0.0, 1.0])


def gravity_prior(k, i, R_vio, sigma_tilt):
    """Absolute roll/pitch constraint from the VIO's gravity observation.

    R_vio: (3,3) front-end rotation at this node, in its own gravity-aligned frame.
    sigma_tilt: per-axis tilt sigma in rad (measured, see Cfg.sigma_tilt).
    """
    b_ref = gtsam.Unit3(np.asarray(R_vio, dtype=float).T @ _E_Z)
    # Pose3AttitudeFactor requires a Diagonal model -- it rejects a Robust wrapper
    # (TypeError), so the heavy tail measured on this residual (p90/median 2.2-2.8
    # vs 1.82 for a Gaussian) cannot be down-weighted by an m-estimator here.
    noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_tilt)
    return gtsam.Pose3AttitudeFactor(X(k, i), gtsam.Unit3(_E_Z), noise, b_ref)


# ----------------------------------------------------------------------------
# Pose prior (gauge / frame-alignment prior, like the paper's phi_pri).
# ----------------------------------------------------------------------------
def pose_prior(k, i, pose0: Pose3, sigma_rot, sigma_trans):
    noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_rot] * 3 + [sigma_trans] * 3))
    return gtsam.PriorFactorPose3(X(k, i), pose0, noise)
