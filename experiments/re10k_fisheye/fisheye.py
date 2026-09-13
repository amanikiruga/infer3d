#!/usr/bin/env python3
"""
OOD-fisheye synthesis + analytic undistortion for the RE10K Table-3 pipeline.

Pure numpy/cv2 — NO model/torch dependency, so both the eval harness and the blind
selector import geometry from here without dragging in a generative prior.

Fisheye model (equidistant projection + theta polynomial radial term):
    r_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
theta = incident ray angle from the optical axis; r_d = distorted radius in image-circle
units. Forward (perspective_to_fisheye) warps a pinhole image into a fisheye; inverse
(undistort_np) is the analytic fisheye->perspective remap used by every baseline and by
"ours" (with different (fov, fxf, k)).
"""
from __future__ import annotations
import math
from typing import Tuple
import cv2
import numpy as np


# ---- focal <-> field-of-view -----------------------------------------------
def fov2f(fov_deg: float, px: int) -> float:
    return (px / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def f2fov(f: float, px: int) -> float:
    return 2.0 * math.degrees(math.atan((px / 2.0) / f))


# ---- theta polynomial ------------------------------------------------------
def _theta_poly(theta, k):
    th2 = theta * theta
    return theta * (1.0 + k[0] * th2 + k[1] * th2**2 + k[2] * th2**3 + k[3] * th2**4)


def _theta_poly_deriv(theta, k):
    th2 = theta * theta
    return 1.0 + 3 * k[0] * th2 + 5 * k[1] * th2**2 + 7 * k[2] * th2**3 + 9 * k[3] * th2**4


def _invert_theta_poly(r_d, k, iters: int = 8):
    """Solve r_d = theta_poly(theta) by Newton iteration. k=0 => theta=r_d exactly."""
    if all(abs(x) < 1e-12 for x in k):
        return r_d
    theta = r_d.copy()
    for _ in range(iters):
        f = _theta_poly(theta, k) - r_d
        theta = theta - f / np.maximum(_theta_poly_deriv(theta, k), 1e-12)
    return theta


# ---- forward: perspective -> fisheye ---------------------------------------
def perspective_to_fisheye(
    img_bgr: np.ndarray,
    out_hw: Tuple[int, int],
    out_fov_deg: float,
    in_fov_deg: float,
    k: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    p: Tuple[float, float] = (0.0, 0.0),
    circle_scale: float = 1.0,
    border_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp a pinhole image into a synthetic fisheye.

    out_fov_deg   : fisheye diameter FOV (<180).
    in_fov_deg    : assumed source pinhole horizontal FOV.
    k, p          : radial (theta-poly) and tangential (OpenCV) distortion.
    circle_scale  : image-circle radius / (frame half-min-dim). 1.0 = inscribed circle
                    (black corners); ~1.15-1.25 = mild-barrel with a thin rim; sqrt(2) =
                    circumscribed full-frame (no black corners).
    Returns (fisheye_bgr uint8, valid_mask uint8 [0/255]).
    """
    if img_bgr.dtype != np.uint8 or img_bgr.ndim != 3:
        raise ValueError("img_bgr must be HxWx3 uint8 BGR")
    in_h, in_w = img_bgr.shape[:2]
    out_h, out_w = out_hw
    fx_in = fov2f(in_fov_deg, in_w); cx_in = (in_w - 1) / 2.0; cy_in = (in_h - 1) / 2.0

    theta_max = math.radians(out_fov_deg) / 2.0
    if not (0 < theta_max < math.radians(90.0)):
        raise ValueError(f"out_fov_deg must be in (0,180); got {out_fov_deg}")
    radius_px = circle_scale * min(out_w, out_h) / 2.0
    fx_out = radius_px / theta_max
    cx_out = (out_w - 1) / 2.0; cy_out = (out_h - 1) / 2.0

    uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x_d = (uu - cx_out) / fx_out; y_d = (vv - cy_out) / fx_out
    r_d = np.sqrt(x_d * x_d + y_d * y_d)
    theta = _invert_theta_poly(r_d, k)
    scale = np.tan(theta) / np.maximum(r_d, 1e-8)
    x_p = x_d * scale; y_p = y_d * scale

    p1, p2 = p
    if abs(p1) > 1e-12 or abs(p2) > 1e-12:
        r2 = x_p * x_p + y_p * y_p
        x_p = x_p + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p * x_p)
        y_p = y_p + p1 * (r2 + 2 * y_p * y_p) + 2 * p2 * x_p * y_p

    map_x = fx_in * x_p + cx_in; map_y = fx_in * y_p + cy_in
    valid = ((map_x >= 0) & (map_x <= in_w - 1) & (map_y >= 0) & (map_y <= in_h - 1)
             & (theta <= theta_max + 1e-6))
    out = cv2.remap(img_bgr, map_x.astype(np.float32), map_y.astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(border_value,) * 3)
    out[~valid] = border_value
    return out, (valid.astype(np.uint8) * 255)


# ---- inverse: fisheye -> perspective ---------------------------------------
def undistort_np(fish, fx_p, fx_f, k, H, W):
    """Analytic fisheye->perspective remap with given (perspective focal, fisheye focal,
    radial k). Returns (perspective_bgr uint8, valid_mask bool). Used identically by every
    undistortion-family method; only (fx_p, fx_f, k) differ."""
    cx = cy = (W - 1) / 2.0
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    xp = (uu - cx) / fx_p; yp = (vv - cy) / fx_p; rp = np.sqrt(xp * xp + yp * yp)
    th = np.arctan(rp); t2 = th * th
    rd = th * (1 + k[0] * t2 + k[1] * t2**2 + k[2] * t2**3 + k[3] * t2**4)
    sc = rd / np.maximum(rp, 1e-8); xd = xp * sc; yd = yp * sc
    uf = fx_f * xd + cx; vf = fx_f * yd + cy
    valid = (uf >= 0) & (uf <= W - 1) & (vf >= 0) & (vf <= H - 1)
    out = cv2.remap(fish, uf.astype(np.float32), vf.astype(np.float32),
                    cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    out[~valid] = 0
    return out, valid


def psnr(a, b, m):
    """Masked PSNR on uint8-range arrays; m is a boolean/0-1 validity mask."""
    m = m.astype(bool)
    if m.sum() == 0:
        return float('nan')
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2)[m].mean()
    return float('inf') if mse < 1e-9 else 10 * math.log10(255.0 ** 2 / mse)


def crop_valid_square(bgr, thresh=2, max_frac=0.002):
    """Center-crop away the black undistortion rim (largest centered square with <max_frac
    black), resize back to full size. Returns (image, zoom); the caller scales the source
    focal by `zoom`. Standard 'crop to valid region' step, applied uniformly to every
    undistortion-family method."""
    S = bgr.shape[0]
    nonblack = bgr.max(axis=2) > thresh
    for m in range(0, S // 4 + 1):
        if (~nonblack[m:S - m, m:S - m]).mean() < max_frac:
            if m == 0:
                return bgr, 1.0
            return cv2.resize(bgr[m:S - m, m:S - m], (S, S), interpolation=cv2.INTER_CUBIC), S / (S - 2 * m)
    m = S // 4
    return cv2.resize(bgr[m:S - m, m:S - m], (S, S), interpolation=cv2.INTER_CUBIC), 2.0
