"""Cumulative (nested) ablation for ShapeNet cars — the build-up complement to the LOO table
(mirrors the paper's cumulative Table 8, which is CO3D losses only).

Two ladders, each adds ONE component per rung; per-rung increment is the PAIRED per-object mean
(same PYTHONHASHSEED=0 init across arms). Because cumulative increments are order-dependent, we
show BOTH orderings of the optimization ladder — which is exactly why the per-component LOO
table is the cleaner "individual contribution" statistic.

Rungs are existing arms:
  feed-forward   = PSNR_novel_baseline column (no optimization)
  search_only    = multi-start latent+pose search & selection, NO gradient
  no_pose_opt    = + gradient-optimize the LATENT (pose frozen)
  no_latent_opt  = + gradient-optimize the POSE (latent frozen)
  control        = full (both optimized)
  mse_only_no_noise = pure-MSE objective (no LPIPS, no noise-reg)
  no_noise_reg   = MSE + LPIPS (no noise-reg)
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import csv, glob, math, os

RUN = f"{_EXT}/splatter-image-rebuttal/rebuttal/results/ablation_runs"
HERE = os.path.dirname(os.path.abspath(__file__))
PCOL, BCOL = "PSNR_novel_best", "PSNR_novel_baseline"


def load(arm, col=PCOL):
    out = {}
    for p in glob.glob(f"{RUN}/{arm}/shard_*/task_a_ours_so3.csv"):
        for row in csv.DictReader(open(p)):
            eid = row.get("example_id")
            try:
                out[eid] = float(row[col])
            except (KeyError, ValueError, TypeError):
                pass
    return out


def ms(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n)


def ladder(name, rungs):
    """rungs: list of (label, loader_dict). Print absolute PSNR + paired increment vs prev."""
    lines = [f"### {name}", "", "| rung (cumulative) | PSNR↑ | Δ vs previous (paired) | n |", "|---|---|---|---|"]
    prev = None
    for label, d in rungs:
        ids = sorted(d)
        m, se = ms([d[i] for i in ids])
        if prev is None:
            inc = "—"
        else:
            common = sorted(set(d) & set(prev))
            dm, dse = ms([d[i] - prev[i] for i in common])
            inc = f"{dm:+.2f} ± {dse:.2f}"
        lines.append(f"| {label} | {m:.2f} ± {se:.2f} | {inc} | {len(ids)} |")
        prev = d
    return "\n".join(lines) + "\n"


def main():
    control = load("control")
    feedfwd = load("control", BCOL)               # feed-forward baseline column
    search = load("search_only")
    lat = load("no_pose_opt")                      # latent opt on, pose off
    pose = load("no_latent_opt")                   # pose opt on, latent off
    mse = load("mse_only_no_noise")                # pure MSE
    mse_lpips = load("no_noise_reg")               # MSE + LPIPS

    out = ["# Cumulative ablation — ShapeNet cars / StyleGAN (build-up complement to the LOO table)\n"]
    out.append("Each rung adds one component; increments are paired per-object. Cumulative "
               "increments are **order-dependent** (compare the two optimization orderings below) "
               "— the per-component LOO table is the order-independent statistic.\n")

    out.append(ladder("A. Optimization build-up (losses = full), latent-first",
                      [("feed-forward lifter (no optimization)", feedfwd),
                       ("+ multi-start search & selection (no gradient)", search),
                       ("+ latent-code gradient opt (pose frozen)", lat),
                       ("+ pose gradient opt = full Infer3D", control)]))
    out.append("\n")
    out.append(ladder("A'. Same, pose-first (shows order-dependence)",
                      [("feed-forward lifter (no optimization)", feedfwd),
                       ("+ multi-start search & selection (no gradient)", search),
                       ("+ pose gradient opt (latent frozen)", pose),
                       ("+ latent-code gradient opt = full Infer3D", control)]))
    out.append("\n")
    out.append(ladder("B. Loss build-up (optimization = full) — cars analog of Table 8",
                      [("MSE only", mse),
                       ("+ LPIPS", mse_lpips),
                       ("+ noise-map regularizer = full Infer3D", control)]))

    txt = "\n".join(out) + "\n"
    with open(f"{HERE}/cumulative_table_cars.md", "w") as f:
        f.write(txt)
    print(txt)
    print("[wrote cumulative_table_cars.md]")


if __name__ == "__main__":
    main()
