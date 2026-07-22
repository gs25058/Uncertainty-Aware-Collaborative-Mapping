#!/usr/bin/env python3
"""Unit tests for the analytic range-factor Jacobians in covor/factors.py.

For each factor we compare the factor's own (analytic) linearization against
(a) finite differences of the residual taken on the GTSAM retract tangent, and
(b) for the inter-agent factor, GTSAM's built-in RangeFactorPose3 — an
independent C++ analytic implementation of the same residual.

Run:  python scripts/test_jacobians.py
"""
import sys
import numpy as np
import gtsam
from gtsam import Pose3, Rot3

sys.path.insert(0, "/src/gs25058/cr_RNE/covor_slam")
from covor import factors as F

EPS = 1e-6
rng = np.random.default_rng(0)


def rand_pose():
    return Pose3(Rot3.RzRyRx(*rng.uniform(-np.pi, np.pi, 3)),
                 rng.uniform(-3, 3, 3))


def analytic_AH(factor, values, keys, sigma):
    """Return the factor's analytic UNWHITENED Jacobian per key (H = A * sigma)
    and the raw residual, by linearizing the actual factor."""
    jf = factor.linearize(values)
    A, b = jf.jacobian()                 # whitened: A = H / sigma, b = -r/sigma
    Hs, off = {}, 0
    for key in keys:
        w = values.at(key) if False else None
        dim = 6 if gtsam.Symbol(key).chr() in b"abc" else 1
        Hs[key] = A[:, off:off + dim] * sigma
        off += dim
    return Hs, -b * sigma                 # unwhitened residual


def num_jac_pose(res_fn, values, key):
    """Finite-diff (1x6) of scalar residual w.r.t. a Pose3's retract tangent."""
    p = values.atPose3(key)
    r0 = res_fn(values)
    J = np.zeros((1, 6))
    for j in range(6):
        dv = np.zeros(6); dv[j] = EPS
        v2 = gtsam.Values(values); v2.update(key, p.retract(dv))
        J[0, j] = (res_fn(v2) - r0) / EPS
    return J


def num_jac_scalar(res_fn, values, key):
    b0 = values.atDouble(key)
    r0 = res_fn(values)
    v2 = gtsam.Values(values); v2.update(key, b0 + EPS)
    return np.array([[(res_fn(v2) - r0) / EPS]])


def check(name, Ha, Hn, tol=1e-4):
    err = np.max(np.abs(Ha - Hn))
    ok = err < tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:34s} max|analytic-numeric| = {err:.2e}")
    return ok


def test_anchor(bias=False):
    print(f"\n== anchor_range_factor (bias={'on' if bias else 'off'}) ==")
    sigma = 0.3; z = rng.uniform(1, 5)
    anchor = rng.uniform(-3, 3, 3)
    ka, ia = 1, 7
    values = gtsam.Values()
    values.insert(F.X(ka, ia), rand_pose())
    bk = F.Bias(ka) if bias else None
    keys = [F.X(ka, ia)] + ([bk] if bias else [])
    if bias:
        values.insert(bk, 0.11)
    fac = F.anchor_range_factor(ka, ia, anchor, z, sigma, robust=False, bias_key=bk)

    def res(v):
        p = v.atPose3(F.X(ka, ia)).translation()
        b = v.atDouble(bk) if bias else 0.0
        return float(np.linalg.norm(p - anchor) - z + b)

    Ha, r_fac = analytic_AH(fac, values, keys, sigma)
    ok = check("dr/dpose", Ha[F.X(ka, ia)], num_jac_pose(res, values, F.X(ka, ia)))
    ok &= abs(r_fac[0] - res(values)) < 1e-9
    if bias:
        ok &= check("dr/dbias", Ha[bk], num_jac_scalar(res, values, bk))
    return ok


def test_inter(bias=False):
    print(f"\n== inter_range_factor (bias={'on' if bias else 'off'}) ==")
    sigma = 0.25; z = rng.uniform(1, 5)
    ka, ia, kb, ib = 0, 3, 2, 9
    values = gtsam.Values()
    values.insert(F.X(ka, ia), rand_pose())
    values.insert(F.X(kb, ib), rand_pose())
    bk = F.Bias(ka) if bias else None
    keys = [F.X(ka, ia), F.X(kb, ib)] + ([bk] if bias else [])
    if bias:
        values.insert(bk, -0.05)
    fac = F.inter_range_factor(ka, ia, kb, ib, z, sigma, robust=False, bias_key=bk)

    def res(v):
        pa = v.atPose3(F.X(ka, ia)).translation()
        pb = v.atPose3(F.X(kb, ib)).translation()
        b = v.atDouble(bk) if bias else 0.0
        return float(np.linalg.norm(pa - pb) - z + b)

    Ha, r_fac = analytic_AH(fac, values, keys, sigma)
    ok = check("dr/dpose_a", Ha[F.X(ka, ia)], num_jac_pose(res, values, F.X(ka, ia)))
    ok &= check("dr/dpose_b", Ha[F.X(kb, ib)], num_jac_pose(res, values, F.X(kb, ib)))
    ok &= abs(r_fac[0] - res(values)) < 1e-9
    if bias:
        ok &= check("dr/dbias", Ha[bk], num_jac_scalar(res, values, bk))

    # cross-check against GTSAM's built-in RangeFactorPose3 (independent analytic)
    if not bias:
        nm = gtsam.noiseModel.Isotropic.Sigma(1, sigma)
        builtin = gtsam.RangeFactorPose3(F.X(ka, ia), F.X(kb, ib), z, nm)
        Ab, _ = builtin.linearize(values).jacobian()
        Aa, _ = fac.linearize(values).jacobian()
        ok &= check("vs builtin RangeFactorPose3", Aa, Ab)
    return ok


def test_ma_anchor():
    print("\n== ma_anchor_range_factor (moment arm) ==")
    sigma = 0.3; z = rng.uniform(1, 5)
    anchor = rng.uniform(-3, 3, 3); l = rng.uniform(-0.2, 0.2, 3)
    ka, ia = 1, 7
    values = gtsam.Values()
    values.insert(F.X(ka, ia), rand_pose())
    fac = F.ma_anchor_range_factor(ka, ia, l, anchor, z, sigma, robust=False)

    def res(v):
        T = v.atPose3(F.X(ka, ia))
        q = T.translation() + T.rotation().matrix() @ l
        return float(np.linalg.norm(q - anchor) - z)

    Ha, r_fac = analytic_AH(fac, values, [F.X(ka, ia)], sigma)
    ok = check("dr/dpose", Ha[F.X(ka, ia)], num_jac_pose(res, values, F.X(ka, ia)))
    ok &= abs(r_fac[0] - res(values)) < 1e-9
    return ok


def test_ma_inter():
    print("\n== ma_inter_range_factor (moment arms) ==")
    sigma = 0.25; z = rng.uniform(1, 5)
    la = rng.uniform(-0.2, 0.2, 3); lb = rng.uniform(-0.2, 0.2, 3)
    ka, ia, kb, ib = 0, 3, 2, 9
    values = gtsam.Values()
    values.insert(F.X(ka, ia), rand_pose())
    values.insert(F.X(kb, ib), rand_pose())
    fac = F.ma_inter_range_factor(ka, ia, la, kb, ib, lb, z, sigma, robust=False)

    def res(v):
        Ta = v.atPose3(F.X(ka, ia)); Tb = v.atPose3(F.X(kb, ib))
        qa = Ta.translation() + Ta.rotation().matrix() @ la
        qb = Tb.translation() + Tb.rotation().matrix() @ lb
        return float(np.linalg.norm(qa - qb) - z)

    Ha, r_fac = analytic_AH(fac, values, [F.X(ka, ia), F.X(kb, ib)], sigma)
    ok = check("dr/dpose_a", Ha[F.X(ka, ia)], num_jac_pose(res, values, F.X(ka, ia)))
    ok &= check("dr/dpose_b", Ha[F.X(kb, ib)], num_jac_pose(res, values, F.X(kb, ib)))
    ok &= abs(r_fac[0] - res(values)) < 1e-9
    return ok


def test_height():
    print("\n== height_factor ==")
    sigma = 0.1; h = rng.uniform(0.2, 1.8)
    ka, ia = 2, 4
    values = gtsam.Values()
    values.insert(F.X(ka, ia), rand_pose())
    fac = F.height_factor(ka, ia, h, sigma, robust=False)

    def res(v):
        return float(v.atPose3(F.X(ka, ia)).translation()[2] - h)

    Ha, r_fac = analytic_AH(fac, values, [F.X(ka, ia)], sigma)
    ok = check("dr/dpose", Ha[F.X(ka, ia)], num_jac_pose(res, values, F.X(ka, ia)))
    ok &= abs(r_fac[0] - res(values)) < 1e-9
    return ok


if __name__ == "__main__":
    ok = True
    for b in (False, True):
        ok &= test_anchor(b)
        ok &= test_inter(b)
    ok &= test_ma_anchor()
    ok &= test_ma_inter()
    ok &= test_height()
    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    sys.exit(0 if ok else 1)
