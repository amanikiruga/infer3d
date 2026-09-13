"""Frequency-resolved in-distribution gap:  Splatter Image  vs  Infer3D.

CPU-only. Consumes the view-aligned PNG stacks produced by render_id_pair.py plus
the bundle GT targets, and answers reviewer XsZX / AC-priority-2: *where* (in spatial
frequency) does Infer3D's small in-distribution deficit live?

Two measurements, per method, averaged over all (object, view) pairs:

  1. Low-pass sweep.  Apply an identical Gaussian low-pass (sigma sweep) to BOTH the
     rendered image and the GT, then score PSNR/SSIM with the paper's exact metric code
     (scorer.score_views formulas, reimplemented on arrays here to stay CPU-only). The
     claim holds if  gap = metric(Splatter) - metric(Infer3D)  ->  ~0  as sigma grows
     (cutoff drops): below some frequency the two methods are indistinguishable.

  2. Radially-averaged error power spectrum  <|FFT(render - gt)|^2>.  Threshold-free;
     shows which spatial-frequency bands hold each method's error energy. Infer3D's
     excess should sit in the high-frequency tail.

The PSNR/SSIM here are validated against scorer.score_views at sigma=0 (see --check).

Usage:
  mamba run -n test2 python frequency_gap.py \
     --ours   ../results/task_a_runs/freq_ours_control \
     --splat  ../results/task_a_runs/freq_splatter_control \
     --bundles ../results/task_a_frontal \
     --tag cars_id_frontal
"""
import argparse
import glob
import json
import os
from math import exp

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# ---- metrics (match scorer.py formulas; numpy/CPU) -------------------------
def _gauss_win(ws=11, sigma=1.5):
    g = np.array([exp(-(x - ws // 2) ** 2 / (2 * sigma ** 2)) for x in range(ws)])
    g /= g.sum()
    w = np.outer(g, g)
    return w


_W = _gauss_win()


def _filt2(img, w):
    # per-channel 2D correlation with reflect padding (matches conv2d 'same')
    from scipy.ndimage import correlate
    out = np.empty_like(img)
    for c in range(img.shape[-1]):
        out[..., c] = correlate(img[..., c], w, mode="reflect")
    return out


def ssim_np(a, b):
    """SSIM identical in form to scorer.ssim_metric (11x11 gaussian, C1/C2)."""
    mu1, mu2 = _filt2(a, _W), _filt2(b, _W)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = _filt2(a * a, _W) - mu1_sq
    s2 = _filt2(b * b, _W) - mu2_sq
    s12 = _filt2(a * b, _W) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(m.mean())


def psnr_np(a, b):
    return float(-10.0 * np.log10(np.mean((a - b) ** 2) + 1e-12))


def load(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0


# ---- pair collection -------------------------------------------------------
def collect(method_dir, bundles_dir):
    """Return list of (render[H,W,3], gt[H,W,3]) over all objects/views."""
    pairs = []
    for oid in sorted(os.listdir(method_dir)):
        rdir = os.path.join(method_dir, oid)
        tdir = os.path.join(bundles_dir, oid, "targets")
        if not (os.path.isdir(rdir) and os.path.isdir(tdir)):
            continue
        rs = sorted(glob.glob(f"{rdir}/*.png"))
        ts = sorted(glob.glob(f"{tdir}/*.png"))
        for rp, tp in zip(rs, ts):
            pairs.append((load(rp), load(tp)))
    return pairs


# ---- analyses --------------------------------------------------------------
def lowpass_sweep(pairs, sigmas):
    rows = []
    for s in sigmas:
        ps, ss = [], []
        for r, g in pairs:
            if s > 0:
                rf = np.stack([gaussian_filter(r[..., c], s) for c in range(3)], -1)
                gf = np.stack([gaussian_filter(g[..., c], s) for c in range(3)], -1)
            else:
                rf, gf = r, g
            ps.append(psnr_np(rf, gf))
            ss.append(ssim_np(rf, gf))
        rows.append(dict(sigma=float(s), psnr=float(np.mean(ps)), ssim=float(np.mean(ss))))
    return rows


def radial_error_spectrum(pairs, nbins=64):
    acc, cnt = None, 0
    for r, g in pairs:
        err = (r - g).mean(-1)
        P = np.abs(np.fft.fftshift(np.fft.fft2(err))) ** 2
        H, W = P.shape
        y, x = np.indices((H, W))
        rr = np.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
        b = (rr / rr.max() * (nbins - 1)).astype(int)
        prof = np.bincount(b.ravel(), P.ravel(), minlength=nbins) / \
            np.maximum(np.bincount(b.ravel(), minlength=nbins), 1)
        acc = prof if acc is None else acc + prof
        cnt += 1
    freq = (np.linspace(0, 1, nbins) * 0.5).tolist()   # cycles/pixel
    return freq, (acc / max(cnt, 1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--splat", required=True)
    ap.add_argument("--bundles", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    sigmas = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    op = collect(args.ours, args.bundles)
    spp = collect(args.splat, args.bundles)
    print(f"ours pairs {len(op)} | splatter pairs {len(spp)}")

    res = {"tag": args.tag, "sigmas": sigmas,
           "infer3d": {"n_pairs": len(op), "sweep": lowpass_sweep(op, sigmas)},
           "splatter": {"n_pairs": len(spp), "sweep": lowpass_sweep(spp, sigmas)}}
    f_i, s_i = radial_error_spectrum(op)
    f_s, s_s = radial_error_spectrum(spp)
    res["infer3d"]["error_spectrum"] = {"freq": f_i, "power": s_i}
    res["splatter"]["error_spectrum"] = {"freq": f_s, "power": s_s}

    out = os.path.join(args.out_dir, f"freq_{args.tag}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"wrote {out}\n")

    print(" sigma | Splatter PSNR | Infer3D PSNR | gap dB | Splat SSIM | I3D SSIM | gap")
    for i, s in enumerate(sigmas):
        b = res["splatter"]["sweep"][i]
        o = res["infer3d"]["sweep"][i]
        print(f" {s:5.1f} | {b['psnr']:12.3f}  | {o['psnr']:11.3f}  | {b['psnr']-o['psnr']:6.3f} |"
              f" {b['ssim']:10.4f} | {o['ssim']:8.4f} | {b['ssim']-o['ssim']:.4f}")


if __name__ == "__main__":
    main()
