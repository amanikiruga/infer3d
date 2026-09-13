"""Collect ablation arm results into a leave-one-out (LOO) table.

For each arm we read every per-shard CSV (one object each), key metrics by example_id,
and compute PAIRED deltas vs the control arm (same object, same PYTHONHASHSEED=0 init).
Paired deltas are the right statistic: the component's contribution = control - arm on the
SAME object, so per-object difficulty cancels and the std-error is small.

Usage:
  python collect_ablation.py cars     # ShapeNet cars / StyleGAN
  python collect_ablation.py co3d      # CO3D hydrants / DiffAE
Writes ablation_table_<which>.{md,csv} next to this file and prints the table.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import csv
import glob
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = f"{_EXT}/splatter-image-rebuttal/rebuttal/results"

# which -> (run_dir, csv_basename, control_arm, ordered arm list with human labels)
SETTINGS = {
    "cars": dict(
        run_dir=f"{RESULTS}/ablation_runs",
        csv_name="task_a_ours_so3.csv",
        control="control",
        # (arm_dir, human label, which component it removes)
        arms=[
            ("control",         "Full Infer3D (control)",          "-"),
            ("no_pose_opt",     "- rendering-param (pose) opt",    "rendering-param optimization"),
            ("no_latent_opt",   "- latent-code opt",               "latent-code optimization"),
            ("search_only",     "- ALL gradient opt (search only)","latent + pose optimization"),
            ("mse_only",        "- LPIPS loss (MSE only)",         "L_LPIPS"),
            ("lpips_only",      "- MSE loss (LPIPS only)",         "L_MSE"),
            ("no_noise_reg",    "- noise-map regularizer",         "L_noise-reg"),
            ("few_rotations",   "reduce pose multi-start (30->10 rot)",   "pose multi-start breadth"),
            ("few_latents",     "reduce latent multi-start (20->2 lat)",  "latent multi-start breadth"),
        ],
    ),
    "co3d": dict(
        run_dir=f"{RESULTS}/ablation_runs_co3d",
        csv_name="co3d_se3_results.csv",
        control="b0_control",
        arms=[
            ("b0_control",      "Full Infer3D (control)",   "-"),
            ("no_depth",        "- depth loss",             "L_depth"),
            ("no_latent_prior", "- latent prior (w_reg)",   "L_prior"),
            ("no_lpips",        "- LPIPS loss",             "L_LPIPS"),
            ("no_mse",          "- MSE loss",               "L_MSE"),
            ("no_latent_opt",   "- latent-code opt",        "latent-code optimization"),
            ("no_pose_opt",     "- rendering-param (pose) opt", "rendering-param optimization"),
        ],
    ),
}

# metric column (best particle, novel views) differs by setting
PSNR_COL = {"cars": "PSNR_novel_best", "co3d": "PSNR_novel_best"}
SSIM_COL = {"cars": "SSIM_novel_best", "co3d": "SSIM_novel_best"}
LPIPS_COL = {"cars": "LPIPS_novel_best", "co3d": "LPIPS_novel_best"}
BASELINE_COL = {"cars": "PSNR_novel_baseline", "co3d": "PSNR_novel_baseline_ood"}


def load_arm(run_dir, arm, csv_name):
    """Return {example_id: {col: float}} merged across all shard CSVs for one arm."""
    out = {}
    for path in glob.glob(f"{run_dir}/{arm}/shard_*/{csv_name}") + glob.glob(f"{run_dir}/{arm}/{csv_name}"):
        with open(path) as f:
            for row in csv.DictReader(f):
                eid = row.get("example_id")
                if not eid:
                    continue
                rec = {}
                for k, v in row.items():
                    try:
                        rec[k] = float(v)
                    except (TypeError, ValueError):
                        pass
                out[eid] = rec  # last write wins (shards are disjoint objects anyway)
    return out


def mean_std_err(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def main(which):
    s = SETTINGS[which]
    pcol, scol, lcol, bcol = PSNR_COL[which], SSIM_COL[which], LPIPS_COL[which], BASELINE_COL[which]
    ctrl = load_arm(s["run_dir"], s["control"], s["csv_name"])
    ctrl_ids = set(ctrl)

    rows = []
    for arm, label, removes in s["arms"]:
        data = load_arm(s["run_dir"], arm, s["csv_name"])
        ids = sorted(set(data) & ctrl_ids) if arm != s["control"] else sorted(data)
        psnr = [data[i][pcol] for i in ids if pcol in data[i]]
        ssim = [data[i].get(scol) for i in ids if data[i].get(scol) is not None]
        lp = [data[i].get(lcol) for i in ids if data[i].get(lcol) is not None]
        pm, pse = mean_std_err(psnr)
        sm, _ = mean_std_err([x for x in ssim if x is not None])
        lm, _ = mean_std_err([x for x in lp if x is not None])
        # paired delta vs control (same objects)
        if arm == s["control"]:
            dmean, dse = 0.0, 0.0
        else:
            paired = [data[i][pcol] - ctrl[i][pcol] for i in ids
                      if pcol in data[i] and pcol in ctrl.get(i, {})]
            dmean, dse = mean_std_err(paired)
        rows.append(dict(arm=arm, label=label, removes=removes, n=len(psnr),
                         psnr=pm, psnr_se=pse, ssim=sm, lpips=lm,
                         dpsnr=dmean, dpsnr_se=dse))

    # baseline (feed-forward Splatter) reference from control CSV
    base = [ctrl[i][bcol] for i in ctrl if bcol in ctrl[i]]
    base_m, _ = mean_std_err(base)

    # ---- render ----
    hdr = f"# Component ablation ({which}) — leave-one-out from the full system\n\n"
    hdr += (f"Feed-forward Splatter baseline (same objects, no optimization): "
            f"**{base_m:.2f} dB** (n={len(base)})\n\n")
    tbl = ["| Arm | Removes | PSNR↑ | ΔPSNR (paired) | SSIM↑ | LPIPS↓ | n |",
           "|---|---|---|---|---|---|---|"]
    csv_rows = [["arm", "removes", "psnr", "psnr_se", "dpsnr_paired", "dpsnr_se", "ssim", "lpips", "n"]]
    for r in rows:
        dtxt = "—" if r["arm"] == s["control"] else f"{r['dpsnr']:+.2f} ± {r['dpsnr_se']:.2f}"
        tbl.append(f"| {r['label']} | {r['removes']} | {r['psnr']:.2f} ± {r['psnr_se']:.2f} "
                   f"| {dtxt} | {r['ssim']:.3f} | {r['lpips']:.3f} | {r['n']} |")
        csv_rows.append([r["arm"], r["removes"], f"{r['psnr']:.4f}", f"{r['psnr_se']:.4f}",
                         f"{r['dpsnr']:.4f}", f"{r['dpsnr_se']:.4f}", f"{r['ssim']:.4f}",
                         f"{r['lpips']:.4f}", r["n"]])
    out_md = hdr + "\n".join(tbl) + "\n"
    with open(f"{HERE}/ablation_table_{which}.md", "w") as f:
        f.write(out_md)
    with open(f"{HERE}/ablation_table_{which}.csv", "w") as f:
        csv.writer(f).writerows(csv_rows)
    print(out_md)
    print(f"[wrote ablation_table_{which}.md / .csv]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cars")
