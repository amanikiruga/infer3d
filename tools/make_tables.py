"""Assemble the Infer3D paper tables from the per-object evaluation CSVs.

Every number printed here is a mean over the FULL set of evaluated objects. No cell is
filtered, reweighted, or chosen to match a published value; the paper's number is shown
beside each cell for reference only and never influences what is computed.

Because the OOD/ICP per-object distributions are heavy tailed, each cell also reports the
median and a 95% CI, and each comparison is additionally reported as a *paired* per-object
delta (ours minus baseline on the same objects) -- the statistic the paper's claims rest on.

    python tools/make_tables.py                      # bundled per-object CSVs
    RESULTS_ROOT=/path/to/your/runs python tools/make_tables.py   # score your own

The per-object eval CSVs behind Tables 1 and 2 are bundled under `assets/eval_csvs/`
(179 KB), so this runs with no datasets, no weights and no GPU. Point RESULTS_ROOT at
your own eval output to score a fresh reproduction instead.
"""
import csv
import math
import os
import statistics as st

from infer3d import config

ROOT = os.getenv("RESULTS_ROOT", os.path.join(config.ASSETS_DIR, "eval_csvs"))

EVAL = "eval_output/{d}/results/{f}_with_icp.csv"
OURS = dict(CD="chamfer", PSNR="ours_icp_psnr", SSIM="ours_icp_ssim", LPIPS="ours_icp_lpips")
BASE = dict(CD="chamfer", PSNR="baseline_ood_icp_psnr", SSIM="baseline_ood_icp_ssim",
            LPIPS="baseline_ood_icp_lpips")

HYD_OOD = "checkpoints-icml-diffae-co3d-hydrants-baseline-se3-new-stylegan-10-depth-ood-final"
HYD_ID = "checkpoints-diffae-co3d-hydrants-baseline-se3-new-stylegan-10-depth-encoder-indist"
HYD_SG_OOD = "checkpoints-REPRO-hydrants-ood-SG"
HYD_SG_ID = "checkpoints-stylegan3-co3d-hydrants-w-inversion-baseline-se3-new-stylegan-10-depth-indist"
VAS_OOD = "checkpoints-icml-diffae-co3d-vases-baseline-se3-new-stylegan-10-depth-ood-fixed"
VAS_ID = "checkpoints-REPRO-vases-id"

TABLE1 = [
    ("Hydrants ID", "Splatter Image", HYD_ID, "baseline_ood", BASE,
     dict(CD=0.052, PSNR=21.48, SSIM=0.785, LPIPS=0.168)),
    ("Hydrants ID", "Infer3D (DiffAE)", HYD_ID, "ours", OURS,
     dict(CD=0.102, PSNR=20.55, SSIM=0.761, LPIPS=0.182)),
    ("Hydrants ID", "Infer3D (SG)", HYD_SG_ID, "ours", OURS,
     dict(CD=0.220, PSNR=17.73, SSIM=0.722, LPIPS=0.224)),
    ("Hydrants OOD", "Splatter Image", HYD_OOD, "baseline_ood", BASE,
     dict(CD=0.733, PSNR=15.73, SSIM=0.665, LPIPS=0.266)),
    ("Hydrants OOD", "Infer3D (DiffAE)", HYD_OOD, "ours", OURS,
     dict(CD=0.461, PSNR=18.46, SSIM=0.714, LPIPS=0.220)),
    ("Hydrants OOD", "Infer3D (SG)", HYD_SG_OOD, "ours", OURS,
     dict(CD=0.637, PSNR=17.08, SSIM=0.694, LPIPS=0.251)),
    ("Vases ID", "Splatter Image", VAS_ID, "baseline_ood", BASE,
     dict(CD=0.061, PSNR=20.82, SSIM=0.751, LPIPS=0.179)),
    ("Vases ID", "Infer3D (DiffAE)", VAS_ID, "ours", OURS,
     dict(CD=0.119, PSNR=19.65, SSIM=0.715, LPIPS=0.215)),
    ("Vases OOD", "Splatter Image", VAS_OOD, "baseline_ood", BASE,
     dict(CD=0.775, PSNR=15.01, SSIM=0.661, LPIPS=0.315)),
    ("Vases OOD", "Infer3D (DiffAE)", VAS_OOD, "ours", OURS,
     dict(CD=0.552, PSNR=17.65, SSIM=0.678, LPIPS=0.261)),
]

TABLE1_PAIRS = [
    ("Hydrants OOD (DiffAE)", HYD_OOD),
    ("Hydrants OOD (StyleGAN)", HYD_SG_OOD),
    ("Vases OOD (DiffAE)", VAS_OOD),
    ("Hydrants ID (DiffAE)", HYD_ID),
    ("Vases ID (DiffAE)", VAS_ID),
]

# ShapeNet-NMR (Table 2). Image metrics only; this pipeline computes no Chamfer.
#
# `_known_best` is the hypothesis the optimizer itself selected, by its decision loss
# against the INPUT image. `_best` is the maximum over the particle population of the
# GROUND-TRUTH novel-view PSNR -- oracle selection on the test set. The paper reports
# `_best`; it is shown here only to quantify how much of the published gain it accounts for.
SO3 = "checkpoints-stylegan3-shapenet-nmr-w-inversion-baseline/side_top_so3_results.csv"
SE3 = ("checkpoints-stylegan3-shapenet-nmr-w-inversion-baseline-se3-really/"
       "side_top_se3_results_HACK.csv")

TABLE2 = [
    ("OOD SO(3)", SO3, dict(PSNR=18.12, SSIM=0.847, LPIPS=0.157),
     dict(PSNR=20.67, SSIM=0.865, LPIPS=0.148)),
    ("OOD SE(3)", SE3, dict(PSNR=14.27, SSIM=0.771, LPIPS=0.308),
     dict(PSNR=16.30, SSIM=0.795, LPIPS=0.238)),
]

METRICS1 = ["CD", "PSNR", "SSIM", "LPIPS"]
METRICS2 = ["PSNR", "SSIM", "LPIPS"]
LOWER_BETTER = {"CD", "LPIPS"}


def load(path):
    p = os.path.join(ROOT, path)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def column(rows, col):
    """{example_id: value} for one column, skipping rows where it is absent/unparseable."""
    out = {}
    for r in rows:
        try:
            out[r.get("example_id") or r.get("object_id")] = float(r[col])
        except (TypeError, ValueError, KeyError):
            pass
    return out


def summarize(vals):
    if not vals:
        return None
    half = 1.96 * st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return st.mean(vals), st.median(vals), half, len(vals)


def fmt(stat, paper):
    m, med, half, n = stat
    return f"{m:7.3f}+-{half:5.3f} (med {med:6.3f}) [{paper:6.3f}] n={n:3d}"


def paired(ours, base, lower_better):
    ids = sorted(set(ours) & set(base))
    d = [ours[i] - base[i] for i in ids]
    if len(d) < 2:
        return None
    m = st.mean(d)
    half = 1.96 * st.stdev(d) / math.sqrt(len(d))
    wins = sum(1 for x in d if (x < 0 if lower_better else x > 0)) / len(d)
    sig = (m + half) < 0 if lower_better else (m - half) > 0
    return m, half, wins, len(d), sig


def table1():
    print("\n" + "=" * 100)
    print("TABLE 1 -- CO3D.  full-set mean +- 95% CI (median) [paper] n")
    print("=" * 100)
    for setting, method, d, f, cols, paper in TABLE1:
        rows = load(EVAL.format(d=d, f=f))
        if rows is None:
            print(f"{setting:13s} {method:17s} MISSING: {d}/{f}")
            continue
        for m in METRICS1:
            stat = summarize(list(column(rows, cols[m]).values()))
            if stat:
                print(f"{setting:13s} {method:17s} {m:5s} {fmt(stat, paper[m])}")
        print()

    print("-" * 100)
    print("TABLE 1 paired: Infer3D minus Splatter Image on the SAME objects")
    print("-" * 100)
    for label, d in TABLE1_PAIRS:
        o, b = load(EVAL.format(d=d, f="ours")), load(EVAL.format(d=d, f="baseline_ood"))
        if not o or not b:
            print(f"{label:26s} MISSING")
            continue
        parts, n = [], 0
        for m in METRICS1:
            p = paired(column(o, OURS[m]), column(b, BASE[m]), m in LOWER_BETTER)
            if p:
                delta, half, wins, n, sig = p
                parts.append(f"{m} {delta:+7.3f}+-{half:.3f}({wins:3.0%}{'*' if sig else ' '})")
        print(f"{label:26s} n={n:3d}  " + "  ".join(parts))
    print("  * = 95% CI excludes zero favouring Infer3D;  (..) = per-object win rate")


def table2():
    print("\n" + "=" * 100)
    print("TABLE 2 -- ShapeNet-NMR.  full-set mean +- 95% CI (median) [paper] n")
    print("=" * 100)
    for setting, path, paper_base, paper_ours in TABLE2:
        rows = load(path)
        if rows is None:
            print(f"{setting:12s} MISSING: {path}")
            continue
        for label, suffix, paper in [
            ("Splatter Img.", "_baseline", paper_base),
            ("Infer3D (SG)", "_known_best", paper_ours),
            ("  \\_ ORACLE-selected", "_best", paper_ours),
        ]:
            for m in METRICS2:
                stat = summarize(list(column(rows, f"{m}_novel{suffix}").values()))
                if stat:
                    print(f"{setting:12s} {label:21s} {m:5s} {fmt(stat, paper[m])}")
        base = column(rows, "PSNR_novel_baseline")
        for label, suffix in [("loss-selected", "_known_best"), ("ORACLE-selected", "_best")]:
            p = paired(column(rows, f"PSNR_novel{suffix}"), base, False)
            if p:
                delta, half, wins, n, sig = p
                print(f"{'':12s} paired PSNR ours-base [{label:15s}] {delta:+6.3f}+-{half:.3f}"
                      f"  win {wins:3.0%}  n={n}{'  *' if sig else ''}")
        print()
    print("NOTE: `_best` picks the particle by ground-truth novel-view PSNR (oracle on the")
    print("test set) and is the column the paper reports. `_known_best` is the hypothesis the")
    print("optimizer chose by its own loss, and is the only one usable as a result.")


if __name__ == "__main__":
    print(f"RESULTS_ROOT = {ROOT}")
    table1()
    table2()
