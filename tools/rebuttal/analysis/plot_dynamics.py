"""Optimization-dynamics figure for the rebuttal (akZb: 'no convergence curve /
step count / dynamics'; uH4P: runtime & memory). Built from the already-logged
runs results_auroc_step_t.csv (n_id=n_ood=43) — no new compute.

Panel 1: mean best-loss vs optimization wall-time, ID vs OOD. Shows the Sec-3.5
mechanism: ID inputs start near-converged (encoder/prior gives a good basin), OOD
inputs start with ~4x higher reconstruction loss and need test-time compute to close it.
Panel 2: OOD-detection AUROC from the running best-loss vs compute (already separable at
step 0 => cheap routing).
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WT = f"{_EXT}/splatter-image-rebuttal"
rows = list(csv.DictReader(open(f"{WT}/results_auroc_step_t.csv")))
# steps 0..300 share one loss regime; step 783 is a different (final-stage) loss weighting
r = [x for x in rows if float(x["step_T"]) <= 300]
t = np.array([float(x["median_wall_time_ms"]) / 1000.0 for x in r])
ood = np.array([float(x["mean_ood_best_loss"]) for x in r])
idl = np.array([float(x["mean_id_best_loss"]) for x in r])
step = np.array([float(x["step_T"]) for x in r])
auroc = np.array([float(x["best_loss_AUROC"]) if x["best_loss_AUROC"] != "n/a" else np.nan for x in r])

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))

ax[0].plot(t, ood, "o-", color="crimson", lw=2, label="OOD input (mean best loss)")
ax[0].plot(t, idl, "s-", color="seagreen", lw=2, label="ID input (mean best loss)")
ax[0].axvline(t[0], color="gray", ls=":", lw=1)
ax[0].annotate(f"feed-forward init\nOOD {ood[0]:.3f} vs ID {idl[0]:.3f}\n({ood[0]/idl[0]:.1f}x separation)",
               (t[0], ood[0]), xytext=(t[1]*1.1, ood[0]*0.95), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="gray"))
ax[0].set_xscale("symlog", linthresh=1)
ax[0].set_xlim(-0.3, 2000)
ax[0].set_xlabel("optimization wall-time (s, symlog; 0 = feed-forward)")
ax[0].set_ylabel("mean reconstruction loss")
ax[0].set_title("Convergence: ID starts near-optimal, OOD needs compute\n"
                f"(n=43 ID / 43 OOD; full run = 783 iters / 6 stages)")
ax[0].legend(fontsize=9)
for x, s in zip(t, step):
    ax[0].annotate(f"{int(s)}", (x, min(ood.max(), ood[0])*1.02), fontsize=6, color="gray", ha="center")

ok = ~np.isnan(auroc)
ax[1].plot(t[ok], auroc[ok], "^-", color="steelblue", lw=2)
ax[1].axhline(0.9708, color="crimson", ls="--", lw=1.5, label="feed-forward init (step 0): 0.971")
ax[1].set_xscale("symlog", linthresh=1)
ax[1].set_xlabel("optimization wall-time (s, symlog)")
ax[1].set_ylabel("OOD-detection AUROC (running best-loss)")
ax[1].set_ylim(0.9, 1.0)
ax[1].set_title("OOD is detectable from the initial loss alone\n(=> cheap adaptive routing, no full optimization needed)")
ax[1].legend(fontsize=8)

fig.suptitle("Infer3D optimization dynamics (ShapeNet cars) — from logged runs, no new compute", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = f"{os.path.dirname(os.path.abspath(__file__))}/dynamics_id_vs_ood.png"
fig.savefig(out, dpi=140)
print("wrote", out)
