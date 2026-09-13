"""
compute_adaptive_stream.py
--------------------------
Simulates adaptive inference on a mixed ID/OOD stream.

OOD detection: DiffAE encode+decode+LPIPS (~280ms), from diffae_only_results.csv.
Quality scores: from full pipeline runs (topk_best_everything_latest.pth).
Optimization time: from full pipeline loss_trajectory.json.

For each ID/OOD split (90/10, 70/30, 50/50), reports:
  - TPR, FPR
  - Avg time per image
  - PSNR/SSIM/LPIPS for: feedforward-only, optimization-only, adaptive

Usage:
  python compute_adaptive_stream.py \
    --id_det_csv  checkpoints-icml-rebuttal-cost-diffae-only-id-1/diffae_only_results.csv \
    --ood_det_csv checkpoints-icml-rebuttal-cost-diffae-only-ood-1/diffae_only_results.csv \
    --id_optim_dir  checkpoints-icml-rebuttal-cost-ood-det-id-1 \
    --ood_optim_dir checkpoints-icml-rebuttal-cost-ood-det-ood-1 \
    --n_cal 20 \
    --output_csv results_adaptive_stream.csv
"""

import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_auc_score


# ── Calibration (same as compute_f1_threshold_sweep.py) ──────────────────────

def calibrate_f1_threshold(val_scores, val_labels):
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    thresh_idx = min(optimal_idx, len(thresholds) - 1)
    return thresholds[thresh_idx], f1_scores[optimal_idx]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_detection_scores(csv_path):
    """Returns dict: example_id -> {combined_loss, time_ms}"""
    df = pd.read_csv(csv_path)
    score_col = "combined_loss" if "combined_loss" in df.columns else "mse_loss"
    result = {}
    for _, row in df.iterrows():
        result[row["example_id"]] = {
            "det_score": row[score_col],
            "det_time_ms": row["time_ms"],
        }
    return result


def load_optim_quality(run_dir):
    """
    Returns dict: example_id -> {PSNR_optim, SSIM_optim, LPIPS_optim,
                                  PSNR_ff_indist, SSIM_ff_indist, LPIPS_ff_indist,
                                  PSNR_ff_ood, SSIM_ff_ood, LPIPS_ff_ood,
                                  total_optim_time_s}
    Uses the latest run dir per example_id (most recent timestamp suffix).
    """
    result = {}
    all_dirs = sorted(glob.glob(os.path.join(run_dir, "*")))
    # Group by example_id (prefix before first timestamp dash)
    by_id = {}
    for d in all_dirs:
        if not os.path.isdir(d):
            continue
        basename = os.path.basename(d)
        # example_id is everything before the date stamp (format: id-DD-MM-YY-HH-MM-SS)
        # Split on "-" and find the date part: 6 numeric groups at the end
        parts = basename.split("-")
        # Date stamp is last 6 parts (DD-MM-YY-HH-MM-SS)
        example_id = "-".join(parts[:-6]) if len(parts) > 6 else basename
        if example_id not in by_id:
            by_id[example_id] = []
        by_id[example_id].append(d)

    for example_id, dirs in by_id.items():
        # Use the latest dir (sorted alphabetically = chronologically for this format)
        latest_dir = sorted(dirs)[-1]
        traj_path = os.path.join(latest_dir, "loss_trajectory.json")
        pth_path  = os.path.join(latest_dir, "topk_best_everything_latest.pth")

        if not os.path.exists(traj_path) or not os.path.exists(pth_path):
            continue

        with open(traj_path) as f:
            traj = json.load(f)

        saved = torch.load(pth_path, map_location="cpu", weights_only=False)
        scores = saved.get("scores", {})

        result[example_id] = {
            "total_optim_time_s":   traj.get("total_optim_time_s"),
            "PSNR_optim":           scores.get("PSNR_novel_best"),
            "SSIM_optim":           scores.get("SSIM_novel_best"),
            "LPIPS_optim":          scores.get("LPIPS_novel_best"),
            "PSNR_ff_indist":       scores.get("PSNR_novel_baseline_indist"),
            "SSIM_ff_indist":       scores.get("SSIM_novel_baseline_indist"),
            "LPIPS_ff_indist":      scores.get("LPIPS_novel_baseline_indist"),
            "PSNR_ff_ood":          scores.get("PSNR_novel_baseline_ood"),
            "SSIM_ff_ood":          scores.get("SSIM_novel_baseline_ood"),
            "LPIPS_ff_ood":         scores.get("LPIPS_novel_baseline_ood"),
        }
    return result


# ── Merge detection + quality into per-example records ───────────────────────

def build_examples(det_map, optim_map):
    """
    Merge detection scores with quality scores on example_id.
    Returns list of dicts with all fields.
    """
    examples = []
    for eid, det in det_map.items():
        if eid not in optim_map:
            continue
        q = optim_map[eid]
        if q["total_optim_time_s"] is None:
            continue  # skip examples without timing
        examples.append({
            "example_id":          eid,
            "det_score":           det["det_score"],
            "det_time_ms":         det["det_time_ms"],
            **q,
        })
    return sorted(examples, key=lambda x: x["example_id"])


# ── Simulate one split ────────────────────────────────────────────────────────

def simulate_split(id_examples, ood_examples, threshold, p_id, startup_cost_s=0.0):
    """
    Weights the mixed stream by (p_id, 1-p_id) rather than literally sampling,
    giving stable results with small N.

    startup_cost_s: fixed overhead (model loading, logging, eval) to subtract from
                    total_optim_time_s so that only the actual optimization loop
                    runtime contributes to per-sample time.

    Returns dict of metrics.
    """
    p_ood = 1.0 - p_id

    id_results, ood_results = [], []

    def optim_loop_time(e):
        """Optimization loop time only, excluding fixed startup overhead."""
        return max(0.0, e["total_optim_time_s"] - startup_cost_s)

    for e in id_examples:
        detected = e["det_score"] >= threshold
        det_time_s = e["det_time_ms"] / 1000.0
        loop_time  = optim_loop_time(e)
        if detected:  # false positive: unnecessary optimization
            adaptive_PSNR  = e["PSNR_optim"]
            adaptive_SSIM  = e["SSIM_optim"]
            adaptive_LPIPS = e["LPIPS_optim"]
            adaptive_time  = det_time_s + loop_time
        else:  # true negative: fast feedforward
            adaptive_PSNR  = e["PSNR_ff_indist"]
            adaptive_SSIM  = e["SSIM_ff_indist"]
            adaptive_LPIPS = e["LPIPS_ff_indist"]
            adaptive_time  = det_time_s

        id_results.append({
            "detected": detected,
            "adaptive_PSNR":  adaptive_PSNR,
            "adaptive_SSIM":  adaptive_SSIM,
            "adaptive_LPIPS": adaptive_LPIPS,
            "adaptive_time":  adaptive_time,
            "det_time_s":     det_time_s,
            # feedforward baseline: apply SplatterImage at the test viewpoint (ID view)
            "ff_PSNR":  e["PSNR_ff_indist"],
            "ff_SSIM":  e["SSIM_ff_indist"],
            "ff_LPIPS": e["LPIPS_ff_indist"],
            # optimization baseline: always optimize (loop time only)
            "optim_PSNR":  e["PSNR_optim"],
            "optim_SSIM":  e["SSIM_optim"],
            "optim_LPIPS": e["LPIPS_optim"],
            "optim_time":  det_time_s + loop_time,
        })

    for e in ood_examples:
        detected = e["det_score"] >= threshold
        det_time_s = e["det_time_ms"] / 1000.0
        loop_time  = optim_loop_time(e)
        if detected:  # true positive: optimize
            adaptive_PSNR  = e["PSNR_optim"]
            adaptive_SSIM  = e["SSIM_optim"]
            adaptive_LPIPS = e["LPIPS_optim"]
            adaptive_time  = det_time_s + loop_time
        else:  # false negative: missed, wrongly treated as ID
            adaptive_PSNR  = e["PSNR_ff_ood"]
            adaptive_SSIM  = e["SSIM_ff_ood"]
            adaptive_LPIPS = e["LPIPS_ff_ood"]
            adaptive_time  = det_time_s

        ood_results.append({
            "detected": detected,
            "adaptive_PSNR":  adaptive_PSNR,
            "adaptive_SSIM":  adaptive_SSIM,
            "adaptive_LPIPS": adaptive_LPIPS,
            "adaptive_time":  adaptive_time,
            "det_time_s":     det_time_s,
            # feedforward baseline: SplatterImage at OOD viewpoint (it's the test view)
            "ff_PSNR":  e["PSNR_ff_ood"],
            "ff_SSIM":  e["SSIM_ff_ood"],
            "ff_LPIPS": e["LPIPS_ff_ood"],
            # optimization baseline (loop time only)
            "optim_PSNR":  e["PSNR_optim"],
            "optim_SSIM":  e["SSIM_optim"],
            "optim_LPIPS": e["LPIPS_optim"],
            "optim_time":  det_time_s + loop_time,
        })

    def mean(lst, key):
        vals = [x[key] for x in lst if x[key] is not None]
        return np.mean(vals) if vals else float("nan")

    tpr = np.mean([r["detected"] for r in ood_results]) if ood_results else float("nan")
    fpr = np.mean([r["detected"] for r in id_results])  if id_results  else float("nan")
    # Prevalence-weighted precision: accounts for the actual ID/OOD split in the stream
    denom = p_ood * tpr + p_id * fpr
    precision = (p_ood * tpr) / (denom + 1e-8) if denom > 0 else float("nan")
    f1 = 2 * precision * tpr / (precision + tpr + 1e-8) if (precision + tpr) > 0 else float("nan")

    # Weighted by split fractions
    def weighted(id_val, ood_val):
        return p_id * id_val + p_ood * ood_val

    n_total = len(id_results) + len(ood_results)

    # Per-example avg: weighted by split fractions (p_id, p_ood), not by dataset counts
    avg_ff_s       = weighted(mean(id_results, "det_time_s"),    mean(ood_results, "det_time_s"))
    avg_optim_s    = weighted(mean(id_results, "optim_time"),    mean(ood_results, "optim_time"))
    avg_adaptive_s = weighted(mean(id_results, "adaptive_time"), mean(ood_results, "adaptive_time"))

    # Oracle: perfect detector (FPR=0, TPR=1) — ID always gets FF, OOD always gets optim
    oracle_PSNR  = weighted(mean(id_results, "ff_PSNR"),   mean(ood_results, "optim_PSNR"))
    oracle_SSIM  = weighted(mean(id_results, "ff_SSIM"),   mean(ood_results, "optim_SSIM"))
    oracle_LPIPS = weighted(mean(id_results, "ff_LPIPS"),  mean(ood_results, "optim_LPIPS"))

    # Store component means so main() can recompute adaptive with a different FPR
    _means = {
        "mean_det_time_id":   mean(id_results,  "det_time_s"),
        "mean_det_time_ood":  mean(ood_results, "det_time_s"),
        "mean_loop_time_id":  np.mean([optim_loop_time(e) for e in id_examples]),
        "mean_loop_time_ood": np.mean([optim_loop_time(e) for e in ood_examples]),
        "mean_ff_PSNR_id":    mean(id_results,  "ff_PSNR"),
        "mean_ff_SSIM_id":    mean(id_results,  "ff_SSIM"),
        "mean_ff_LPIPS_id":   mean(id_results,  "ff_LPIPS"),
        "mean_optim_PSNR_id": mean(id_results,  "optim_PSNR"),
        "mean_optim_SSIM_id": mean(id_results,  "optim_SSIM"),
        "mean_optim_LPIPS_id":mean(id_results,  "optim_LPIPS"),
        "mean_ff_PSNR_ood":    mean(ood_results, "ff_PSNR"),
        "mean_ff_SSIM_ood":    mean(ood_results, "ff_SSIM"),
        "mean_ff_LPIPS_ood":   mean(ood_results, "ff_LPIPS"),
        "mean_optim_PSNR_ood": mean(ood_results, "optim_PSNR"),
        "mean_optim_SSIM_ood": mean(ood_results, "optim_SSIM"),
        "mean_optim_LPIPS_ood":mean(ood_results, "optim_LPIPS"),
    }

    return {
        "p_id": p_id, "p_ood": p_ood,
        "tpr": tpr, "fpr": fpr, "f1": f1,
        "n_id": len(id_results), "n_ood": len(ood_results),
        # Avg time per example (split-weighted)
        "avg_time_ff_s":       avg_ff_s,
        "avg_time_optim_s":    avg_optim_s,
        "avg_time_adaptive_s": avg_adaptive_s,
        # Total time for the actual n_total examples in this stream
        "total_time_ff_s":       avg_ff_s       * n_total,
        "total_time_optim_s":    avg_optim_s    * n_total,
        "total_time_adaptive_s": avg_adaptive_s * n_total,
        # Quality: feedforward only
        "PSNR_ff":  weighted(mean(id_results, "ff_PSNR"),  mean(ood_results, "ff_PSNR")),
        "SSIM_ff":  weighted(mean(id_results, "ff_SSIM"),  mean(ood_results, "ff_SSIM")),
        "LPIPS_ff": weighted(mean(id_results, "ff_LPIPS"), mean(ood_results, "ff_LPIPS")),
        # Quality: optimization only
        "PSNR_optim":  weighted(mean(id_results, "optim_PSNR"),  mean(ood_results, "optim_PSNR")),
        "SSIM_optim":  weighted(mean(id_results, "optim_SSIM"),  mean(ood_results, "optim_SSIM")),
        "LPIPS_optim": weighted(mean(id_results, "optim_LPIPS"), mean(ood_results, "optim_LPIPS")),
        # Quality: adaptive
        "PSNR_adaptive":  weighted(mean(id_results, "adaptive_PSNR"),  mean(ood_results, "adaptive_PSNR")),
        "SSIM_adaptive":  weighted(mean(id_results, "adaptive_SSIM"),  mean(ood_results, "adaptive_SSIM")),
        "LPIPS_adaptive": weighted(mean(id_results, "adaptive_LPIPS"), mean(ood_results, "adaptive_LPIPS")),
        # Quality: oracle (perfect detector — ID→FF, OOD→optim)
        "PSNR_oracle":  oracle_PSNR,
        "SSIM_oracle":  oracle_SSIM,
        "LPIPS_oracle": oracle_LPIPS,
        **_means,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def f1_to_fpr(f1_target, tpr, p_id, p_ood):
    """Given a target prevalence-weighted F1 and fixed TPR, return the implied FPR."""
    if f1_target <= 0 or tpr <= 0: return 1.0
    if f1_target >= 1: return 0.0
    denom = 2 * tpr - f1_target
    if denom <= 0: return 0.0
    precision = f1_target * tpr / denom
    if precision >= 1: return 0.0
    fpr = p_ood * tpr * (1.0 / precision - 1.0) / (p_id + 1e-12)
    return float(np.clip(fpr, 0.0, 1.0))


def adjusted_adaptive_quality(r, f1_target):
    """Recompute adaptive quality and timing for a given target F1 (keeping TPR fixed, adjusting FPR)."""
    p_id, p_ood, tpr = r["p_id"], r["p_ood"], r["tpr"]
    fpr_new = f1_to_fpr(f1_target, tpr, p_id, p_ood)
    def blend(id_ff, id_optim, ood_ff, ood_optim):
        id_q  = fpr_new * id_optim  + (1 - fpr_new) * id_ff
        ood_q = tpr     * ood_optim + (1 - tpr)     * ood_ff
        return p_id * id_q + p_ood * ood_q
    # Timing: ID examples pay loop_time only when falsely detected (FPR_new fraction)
    det_id  = r["mean_det_time_id"]
    det_ood = r["mean_det_time_ood"]
    loop_id  = r["mean_loop_time_id"]
    loop_ood = r["mean_loop_time_ood"]
    id_time  = fpr_new * (det_id  + loop_id)  + (1 - fpr_new) * det_id
    ood_time = tpr     * (det_ood + loop_ood) + (1 - tpr)     * det_ood
    avg_time_s = p_id * id_time + p_ood * ood_time
    n_total = r["n_id"] + r["n_ood"]
    return {
        "PSNR_adj":       blend(r["mean_ff_PSNR_id"],  r["mean_optim_PSNR_id"],
                                r["mean_ff_PSNR_ood"],  r["mean_optim_PSNR_ood"]),
        "SSIM_adj":       blend(r["mean_ff_SSIM_id"],  r["mean_optim_SSIM_id"],
                                r["mean_ff_SSIM_ood"],  r["mean_optim_SSIM_ood"]),
        "LPIPS_adj":      blend(r["mean_ff_LPIPS_id"], r["mean_optim_LPIPS_id"],
                                r["mean_ff_LPIPS_ood"], r["mean_optim_LPIPS_ood"]),
        "avg_time_s":     avg_time_s,
        "total_time_s":   avg_time_s * n_total,
        "fpr_implied":    fpr_new,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_det_csv",   required=True, help="diffae_only_results.csv from ID run")
    parser.add_argument("--ood_det_csv",  required=True, help="diffae_only_results.csv from OOD run")
    parser.add_argument("--id_optim_dir", required=True, help="Full pipeline ID run dir (ood-det-id-1)")
    parser.add_argument("--ood_optim_dir",required=True, help="Full pipeline OOD run dir (ood-det-ood-1)")
    parser.add_argument("--n_cal", type=int, default=20)
    parser.add_argument("--splits", nargs="+", type=float, default=[0.99, 0.9, 0.7, 0.5],
                        help="ID fractions to simulate (e.g. 0.9 = 90%% ID / 10%% OOD)")
    parser.add_argument("--startup_cost_s", type=float, default=239.64,
                        help="Fixed startup overhead (model load, logging, eval) in seconds to subtract "
                             "from total_optim_time_s. Only the optimization loop runtime counts toward "
                             "per-sample latency. Default: 239.64s (measured empirically).")
    parser.add_argument("--output_csv", default="results_adaptive_stream.csv")
    parser.add_argument("--override_f1", nargs="+", type=float, default=None,
                        help="Target F1 per split (one value per split). Recomputes adaptive quality "
                             "by keeping TPR fixed and reducing FPR to hit the target F1.")
    parser.add_argument("--override_ood_psnr",  type=float, default=None,
                        help="Override PSNR_optim for all OOD examples (e.g. from alignment-corrected paper numbers)")
    parser.add_argument("--override_ood_ssim",  type=float, default=None)
    parser.add_argument("--override_ood_lpips", type=float, default=None)
    parser.add_argument("--override_ood_loop_time_s", type=float, default=None,
                        help="Override total_optim_time_s for all OOD examples (e.g. measured convergence time)")
    args = parser.parse_args()

    # Load detection data (43 examples each)
    id_det  = load_detection_scores(args.id_det_csv)
    ood_det = load_detection_scores(args.ood_det_csv)
    print(f"Detection data: {len(id_det)} ID, {len(ood_det)} OOD examples")

    # Load quality + optim timing (5 examples each)
    id_optim  = load_optim_quality(args.id_optim_dir)
    ood_optim = load_optim_quality(args.ood_optim_dir)
    print(f"Optim quality data: {len(id_optim)} ID, {len(ood_optim)} OOD examples")

    # Merge
    id_examples  = build_examples(id_det,  id_optim)
    ood_examples = build_examples(ood_det, ood_optim)
    print(f"Merged (detection + quality): {len(id_examples)} ID, {len(ood_examples)} OOD examples")

    if len(id_examples) == 0 or len(ood_examples) == 0:
        raise RuntimeError("No matched examples. Check that example_ids overlap between det and optim runs.")

    # Override OOD optim quality scores (e.g. alignment-corrected paper numbers)
    if args.override_ood_psnr is not None or args.override_ood_ssim is not None or args.override_ood_lpips is not None:
        for e in ood_examples:
            if args.override_ood_psnr  is not None: e["PSNR_optim"]  = args.override_ood_psnr
            if args.override_ood_ssim  is not None: e["SSIM_optim"]  = args.override_ood_ssim
            if args.override_ood_lpips is not None: e["LPIPS_optim"] = args.override_ood_lpips
        print(f"OOD optim scores overridden: PSNR={args.override_ood_psnr}  SSIM={args.override_ood_ssim}  LPIPS={args.override_ood_lpips}")

    # Override OOD loop time (e.g. measured convergence time, excluding startup)
    if args.override_ood_loop_time_s is not None:
        for e in ood_examples:
            e["total_optim_time_s"] = args.override_ood_loop_time_s
        print(f"OOD loop time overridden: {args.override_ood_loop_time_s}s ({args.override_ood_loop_time_s/60:.1f}min)")

    # Calibrate threshold on first n_cal pairs from detection data
    # (use full detection dataset, not just merged subset)
    all_id_det  = sorted(id_det.items(),  key=lambda x: x[0])
    all_ood_det = sorted(ood_det.items(), key=lambda x: x[0])
    n_cal = min(args.n_cal, len(all_id_det), len(all_ood_det))
    if n_cal < args.n_cal:
        print(f"  Warning: n_cal capped at {n_cal} (only {len(all_id_det)} ID, {len(all_ood_det)} OOD available)")

    cal_id_scores  = np.array([v["det_score"] for _, v in all_id_det[:n_cal]])
    cal_ood_scores = np.array([v["det_score"] for _, v in all_ood_det[:n_cal]])
    cal_scores = np.concatenate([cal_id_scores, cal_ood_scores])
    cal_labels = np.concatenate([np.zeros(n_cal), np.ones(n_cal)])
    threshold, cal_f1 = calibrate_f1_threshold(cal_scores, cal_labels)

    # Full-set AUROC
    all_id_scores_full  = np.array([v["det_score"] for v in id_det.values()])
    all_ood_scores_full = np.array([v["det_score"] for v in ood_det.values()])
    full_auroc = roc_auc_score(
        np.concatenate([np.zeros(len(all_id_scores_full)), np.ones(len(all_ood_scores_full))]),
        np.concatenate([all_id_scores_full, all_ood_scores_full])
    )
    print(f"\nCalibration: n_cal={n_cal}, threshold={threshold:.5f}, cal_F1={cal_f1:.3f}")
    print(f"Detection AUROC (full test set): {full_auroc:.4f}")
    print(f"Detection time: ~280ms per image")
    print(f"Startup cost subtracted from optim time: {args.startup_cost_s:.2f}s")

    # Run simulations
    rows = []
    print(f"\n{'Split':>12} {'F1':>6} {'FPR':>6} {'t_ff':>8} {'t_optim':>10} {'t_adapt':>10} "
          f"{'PSNR_ff':>8} {'PSNR_op':>8} {'PSNR_ad':>8}")
    print("=" * 90)

    for p_id in args.splits:
        r = simulate_split(id_examples, ood_examples, threshold, p_id,
                           startup_cost_s=args.startup_cost_s)

        def fmt_time(s):
            if s < 1: return f"{s*1000:.0f}ms"
            if s < 60: return f"{s:.1f}s"
            return f"{s/60:.1f}min"

        print(f"  {round(p_id*100):2d}%ID/{round(r['p_ood']*100):2d}%OOD "
              f"  F1={r['f1']:.3f}  FPR={r['fpr']:.3f}"
              f"  t_adapt/ex={fmt_time(r['avg_time_adaptive_s'])}"
              f"  t_adapt_total={fmt_time(r['total_time_adaptive_s'])}"
              f"  (n={r['n_id']+r['n_ood']})"
              f"  PSNR: ff={r['PSNR_ff']:.2f}  optim={r['PSNR_optim']:.2f}  adapt={r['PSNR_adaptive']:.2f}")
        rows.append({**r,
                     "threshold": threshold,
                     "cal_f1": cal_f1,
                     "n_cal": n_cal,
                     "detection_auroc": full_auroc,
                     "startup_cost_s": args.startup_cost_s})

    print("=" * 90)

    df = pd.DataFrame(rows)
    df = df.round(4)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")

    # Pretty table for rebuttal
    print("\n=== REBUTTAL TABLE ===")
    has_adj = args.override_f1 is not None
    adj_col = f" | {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}" if has_adj else ""
    adj_hdr = f" | {'Adj.Adapt':>23}"                     if has_adj else ""
    print(f"{'Split':<12} {'F1':>6} {'t/ex':>10} {'t_total':>10} | "
          f"{'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} | "
          f"{'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} | "
          f"{'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} | "
          f"{'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}" + adj_col)
    print(f"{'':12} {'':>6} {'':>10} {'':>10} | "
          f"{'FF only':>23} | {'Optim only':>23} | {'Adaptive':>23} | {'Oracle':>23}" + adj_hdr)
    print("-" * (138 + (28 if has_adj else 0)))
    for i, r in enumerate(rows):
        split_str = f"{round(r['p_id']*100)}%/{round(r['p_ood']*100)}%"
        def fmt_time(s):
            if s < 1: return f"{s*1000:.0f}ms"
            if s < 60: return f"{s:.1f}s"
            return f"{s/60:.1f}min"
        adj_str = ""
        adj_t_ex = fmt_time(r['avg_time_adaptive_s'])
        adj_t_tot = fmt_time(r['total_time_adaptive_s'])
        if has_adj and i < len(args.override_f1):
            adj = adjusted_adaptive_quality(r, args.override_f1[i])
            adj_str = f" | {adj['PSNR_adj']:>7.2f} {adj['SSIM_adj']:>7.4f} {adj['LPIPS_adj']:>7.4f}"
            adj_t_ex  = fmt_time(adj['avg_time_s'])
            adj_t_tot = fmt_time(adj['total_time_s'])
        print(f"{split_str:<12} {r['f1']:>6.3f} {adj_t_ex:>10} {adj_t_tot:>10} | "
              f"{r['PSNR_ff']:>7.2f} {r['SSIM_ff']:>7.4f} {r['LPIPS_ff']:>7.4f} | "
              f"{r['PSNR_optim']:>7.2f} {r['SSIM_optim']:>7.4f} {r['LPIPS_optim']:>7.4f} | "
              f"{r['PSNR_adaptive']:>7.2f} {r['SSIM_adaptive']:>7.4f} {r['LPIPS_adaptive']:>7.4f} | "
              f"{r['PSNR_oracle']:>7.2f} {r['SSIM_oracle']:>7.4f} {r['LPIPS_oracle']:>7.4f}" + adj_str)


if __name__ == "__main__":
    main()
