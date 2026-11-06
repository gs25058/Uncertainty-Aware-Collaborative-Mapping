"""Controlled attenuation of rectified mono8 stereo, not a fire-smoke simulator.

Beer--Lambert transmission and the linear scattering blend follow the model
used by Foggy Cityscapes. No photometric calibration or IR aerosol model is
assumed. Clean depth is ONLY a synthesis input; callers must re-run SGBM.
"""
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class SmokeCfg:
    beta: float = 0.0            # experimental attenuation setting, m^-1
    field: str = "uniform"      # uniform | heterogeneous
    seed: int = 0
    airlight: float = 0.8        # intensity-domain approximation, normalized
    fallback_z: float = 5.0     # synthesis only; never a measured depth

    def __post_init__(self):
        if not np.isfinite(self.beta) or self.beta < 0 or self.field not in ("uniform", "heterogeneous"):
            raise ValueError("beta must be nonnegative and field recognized")
        if not 0 <= self.airlight <= 1 or self.fallback_z <= 0:
            raise ValueError("invalid smoke synthesis parameters")

    def metadata(self):
        return asdict(self)


def complete_depth(z, fallback=5.0):
    """Nearest valid Z for synthesis; all-invalid image -> stated fallback.

    Completion is per camera and never changes the downstream validity mask.
    Returns a separate completed array and the ORIGINAL invalid-pixel fraction.
    """
    z = np.asarray(z, np.float32)
    valid = np.isfinite(z) & (z > 0)
    if valid.all():
        return z.copy(), 0.0
    if not valid.any():
        return np.full(z.shape, fallback, np.float32), 1.0
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False,
                                            return_indices=True)
    return z[tuple(nearest)], float((~valid).mean())


def ray_directions(sd, shape):
    v, u = np.indices(shape, dtype=np.float32)
    return np.stack(((u - sd.cx) / sd.fx, (v - sd.cy) / sd.fy,
                     np.ones(shape, np.float32)), axis=-1)


def rectified_camera_poses(sd, T_wc):
    """Both optical centres in world, with common rectified orientation."""
    left = np.array(T_wc, float, copy=True)
    left[:3, :3] = T_wc[:3, :3] @ sd.R1.T
    right = left.copy()
    right[:3, 3] += left[:3, :3] @ np.array([sd.calib.baseline, 0, 0])
    return left, right


def _field_modes(seed):
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(3, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    freq = axes * (2 * np.pi / rng.uniform(1.2, 4.0, (3, 1)))
    return freq, rng.uniform(-np.pi, np.pi, 3)


def density(points, t, seed):
    """One positive, smooth advected 3D field shared by ALL cameras/frames.

    Three seeded sinusoidal modes, wavelengths 1.2--4.0 m, advection
    (0.06, 0.025, 0.015) m/s. Density lies in [0.05, 1.95], mean 1.
    No independent pixel/frame/camera random draws.
    """
    freq, phase = _field_modes(seed)
    p = np.asarray(points) - np.array([0.06, 0.025, 0.015]) * t
    return 1 + np.sin(p @ freq.T + phase) @ np.array([0.55, 0.30, 0.10])


def line_mean_density(origin, endpoint, t, seed):
    """Exact analytic sight-ray integral of the advected sinusoidal field.

    Integral_0^1 sin(a+b*s) ds = sin(a+b/2)*sinc(b/(2*pi)).
    Avoids a separate large 3D point array for every quadrature sample.
    """
    freq,phase = _field_modes(seed)
    origin=np.asarray(origin)
    a=(origin-np.array([.06,.025,.015])*t)@freq.T+phase
    b=(np.asarray(endpoint)-origin)@freq.T
    return 1+(np.sin(a+b/2)*np.sinc(b/(2*np.pi)))@np.array([.55,.30,.10])


def attenuate(image, z, dirs, T_rect, timestamp, cfg):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("expected rectified mono8 image")
    if cfg.beta == 0:
        return image.copy(), np.ones(image.shape, np.float32), 0.0
    filled, missing = complete_depth(z, cfg.fallback_z)
    length = filled * np.linalg.norm(dirs, axis=-1)
    mean_rho = 1.0
    if cfg.field == "heterogeneous":
        ray = dirs @ T_rect[:3, :3].T
        endpoint = T_rect[:3, 3] + ray * filled[..., None]
        mean_rho = line_mean_density(T_rect[:3,3],endpoint,timestamp,cfg.seed)
    transmission = np.exp(-cfg.beta * length * mean_rho).astype(np.float32)
    foggy = transmission * image + (1 - transmission) * (255 * cfg.airlight)
    return np.rint(np.clip(foggy, 0, 255)).astype(np.uint8), transmission, missing


def clean_right_depth(sd, left, right):
    """Right-view Z using mirrored right->left matching with identical SGBM.

    Both inputs are already rectified. Mirrors make disparities positive.
    This clean estimate is used only for right-image smoke synthesis.
    """
    from covor.synth.render import _depth_rectified
    z, _, _ = _depth_rectified(sd, np.ascontiguousarray(right[:, ::-1]),
                               np.ascontiguousarray(left[:, ::-1]))
    return np.ascontiguousarray(z[:, ::-1])


def stereo_smoke(sd, left, right, clean_z_left, clean_z_right, T_wc, t, cfg):
    dirs = ray_directions(sd, left.shape)
    Tl, Tr = rectified_camera_poses(sd, T_wc)
    il, tl, ml = attenuate(left, clean_z_left, dirs, Tl, t, cfg)
    ir, tr, mr = attenuate(right, clean_z_right, dirs, Tr, t, cfg)
    return il, ir, dict(fill_left=ml, fill_right=mr,
                        transmission_mean=float((tl.mean() + tr.mean()) / 2))
