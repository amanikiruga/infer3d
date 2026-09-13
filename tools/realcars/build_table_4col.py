"""
Merge chamfer_direct.csv (ours, splatter-image baseline) with chamfer_lgm_sf3d.csv
(LGM, SF3D) into one 4-column table. Emit:
  - chamfer_table_4col.md  (drop into AGENT.md)
  - chamfer_table_4col.tex (drop into Overleaf)

Primary metric: asymmetric Chamfer pred->gt (m^2).
"""
import csv
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent / "results"


def read(csv_path):
    with open(csv_path) as f:
        rd = csv.DictReader(f)
        return {r["idx"]: r for r in rd}


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float('nan')


def main():
    ours_base = read(OUT_DIR / "chamfer_direct.csv")
    lgm_sf = read(OUT_DIR / "chamfer_lgm_sf3d.csv")

    idxs = sorted(set(ours_base) & set(lgm_sf))

    rows = []
    for idx in idxs:
        a = ours_base[idx]; b = lgm_sf[idx]
        rows.append({
            "idx": idx,
            "ours":  fnum(a["cd_ours_o2g"]),
            "spim":  fnum(a["cd_baseline_o2g"]),
            "lgm":   fnum(b["cd_lgm_p2g"]),
            "sf3d":  fnum(b["cd_sf3d_p2g"]),
        })

    methods = ["ours", "spim", "lgm", "sf3d"]
    pretty = {"ours": "Ours", "spim": "Splatter Image",
              "lgm": "LGM", "sf3d": "SF3D"}
    arr = {m: np.array([r[m] for r in rows]) for m in methods}

    def best_idx(r):
        vals = [(m, r[m]) for m in methods if not np.isnan(r[m])]
        return min(vals, key=lambda kv: kv[1])[0] if vals else None

    md = OUT_DIR / "chamfer_table_4col.md"
    with open(md, "w") as f:
        f.write("# RealCars Chamfer pred->gt (m^2, lower better) — 4-method\n\n")
        f.write("| idx | Ours | Splatter Image | LGM | SF3D | best |\n")
        f.write("|-----|------|----------------|-----|------|------|\n")
        for r in rows:
            best = best_idx(r)
            cells = []
            for m in methods:
                v = r[m]
                s = f"{v:.4f}" if not np.isnan(v) else "—"
                cells.append(f"**{s}**" if m == best else s)
            f.write(f"| {r['idx']} | " + " | ".join(cells) + f" | {pretty[best]} |\n")
        f.write("\n")
        for m in methods:
            v = arr[m]; v = v[~np.isnan(v)]
            f.write(f"- **{pretty[m]}** mean {v.mean():.4f} | median {np.median(v):.4f} | n={len(v)}\n")
        wins = {m: 0 for m in methods}
        for r in rows:
            b = best_idx(r)
            if b: wins[b] += 1
        f.write("\nWin counts: " + ", ".join(f"{pretty[m]} {wins[m]}/{len(rows)}" for m in methods) + "\n")

    tex = OUT_DIR / "chamfer_table_4col.tex"
    with open(tex, "w") as f:
        f.write("% RealCars 3D reconstruction — Chamfer pred->gt (m^2, lower is better)\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\caption{\\textbf{3D reconstruction accuracy on the RealCars subset.} "
                "Asymmetric Chamfer distance (pred$\\rightarrow$gt, m$^2$, lower is better) "
                "between each method's 3D output and a per-scene FastGS pseudo-ground-truth "
                "fit from the full multi-view ARKit capture. All methods take the same single "
                "$128{\\times}128$ SAM-cropped image as input. Predictions are pre-scaled to "
                "the GT diameter and aligned with FPFH+RANSAC$\\rightarrow$ICP before scoring. "
                "Best per row in \\textbf{bold}.}\n")
        f.write("\\label{tab:realcars_recon}\n")
        f.write("\\begin{tabular}{l" + "c" * len(rows) + "cc}\n\\toprule\n")
        f.write("Method & " + " & ".join(r["idx"] for r in rows) + " & Mean & Median \\\\\n\\midrule\n")
        for m in methods:
            label = pretty[m]
            cells = []
            for r in rows:
                v = r[m]
                if np.isnan(v):
                    cells.append("--")
                else:
                    best = best_idx(r)
                    cells.append(f"\\textbf{{{v:.3f}}}" if m == best else f"{v:.3f}")
            v = arr[m]; v = v[~np.isnan(v)]
            cells.append(f"{v.mean():.3f}")
            cells.append(f"{np.median(v):.3f}")
            f.write(label + " & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Compact variant (no per-scene columns) for tighter layouts
    tex2 = OUT_DIR / "chamfer_table_4col_compact.tex"
    with open(tex2, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{\\textbf{3D reconstruction accuracy on RealCars (n=20).} "
                "Asymmetric Chamfer distance pred$\\rightarrow$gt in m$^2$ (lower is better) "
                "against a FastGS multi-view pseudo-GT, after FPFH+RANSAC$\\rightarrow$ICP "
                "alignment with GT-diameter pre-scaling. All methods consume the same single "
                "$128{\\times}128$ SAM-cropped input. ``Wins'' counts scenes where a method "
                "is best.}\n")
        f.write("\\label{tab:realcars_recon_compact}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & Mean $\\downarrow$ & Median $\\downarrow$ & Wins $\\uparrow$ & n \\\\\n\\midrule\n")
        # bold the best mean
        means = {m: arr[m][~np.isnan(arr[m])].mean() for m in methods}
        best_mean = min(means, key=means.get)
        for m in methods:
            v = arr[m]; v = v[~np.isnan(v)]
            mean_s = f"{v.mean():.3f}"; med_s = f"{np.median(v):.3f}"
            if m == best_mean:
                mean_s = f"\\textbf{{{mean_s}}}"; med_s = f"\\textbf{{{med_s}}}"
            f.write(f"{pretty[m]} & {mean_s} & {med_s} & {wins[m]} & {len(v)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"Wrote {md}\nWrote {tex}\nWrote {tex2}")


if __name__ == "__main__":
    main()
