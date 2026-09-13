"""Plot the frequency-gap results (freq_<tag>.json from frequency_gap.py).

Two panels:
 (left)  PSNR gap  Splatter - Infer3D  vs Gaussian low-pass sigma  (+ absolute PSNR
         curves inset-style on twin axis) -> shows the gap collapsing as the cutoff drops.
 (right) radially-averaged error power spectrum (log-y) for both methods -> shows
         Infer3D's excess error concentrated in high spatial frequency.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = json.load(open(args.json))
    sig = np.array(d["sigmas"], float)
    sp = d["splatter"]["sweep"]; oo = d["infer3d"]["sweep"]
    b_psnr = np.array([r["psnr"] for r in sp]); o_psnr = np.array([r["psnr"] for r in oo])
    b_ssim = np.array([r["ssim"] for r in sp]); o_ssim = np.array([r["ssim"] for r in oo])
    gap = b_psnr - o_psnr

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

    # panel 1: gap vs cutoff
    ax[0].plot(sig, gap, "o-", color="crimson", lw=2, label="PSNR gap (Splatter - Infer3D)")
    ax[0].axhline(0, color="gray", ls=":", lw=1)
    ax[0].set_xlabel("Gaussian low-pass $\\sigma$ (px)  —  larger = lower cutoff")
    ax[0].set_ylabel("PSNR gap (dB)")
    ax[0].set_title(f"In-distribution gap vs frequency cutoff\n(sigma=0 gap = {gap[0]:.2f} dB)")
    axt = ax[0].twinx()
    axt.plot(sig, b_psnr, "s--", color="steelblue", alpha=.6, ms=4, label="Splatter PSNR")
    axt.plot(sig, o_psnr, "^--", color="seagreen", alpha=.6, ms=4, label="Infer3D PSNR")
    axt.set_ylabel("absolute PSNR (dB)")
    l1, la1 = ax[0].get_legend_handles_labels(); l2, la2 = axt.get_legend_handles_labels()
    ax[0].legend(l1 + l2, la1 + la2, loc="center right", fontsize=8)

    # panel 2: error power spectrum
    fs = np.array(d["splatter"]["error_spectrum"]["freq"])
    ps = np.array(d["splatter"]["error_spectrum"]["power"])
    po = np.array(d["infer3d"]["error_spectrum"]["power"])
    ax[1].semilogy(fs, ps, color="steelblue", lw=2, label="Splatter error power")
    ax[1].semilogy(fs, po, color="seagreen", lw=2, label="Infer3D error power")
    ax[1].fill_between(fs, ps, po, where=(po > ps), color="crimson", alpha=.15,
                       label="Infer3D excess")
    ax[1].set_xlabel("spatial frequency (cycles/pixel)")
    ax[1].set_ylabel("mean error power  $\\langle|FFT(err)|^2\\rangle$")
    ax[1].set_title("Where the error lives")
    ax[1].legend(fontsize=8)

    fig.suptitle(f"Frequency analysis of the in-distribution gap — {d['tag']} "
                 f"(n={d['infer3d']['n_pairs']} views)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out or args.json.replace(".json", ".png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
