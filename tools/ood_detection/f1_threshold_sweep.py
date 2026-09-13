"""
compute_f1_threshold_sweep.py
------------------------------
Evaluates OOD detection quality (AUROC + F1) for the DiffAE-only pipeline.

Calibration: use first N_cal ID+OOD test pairs to find the F1-maximising
threshold via precision_recall_curve, then evaluate on the held-out remainder.
Sweeping N_cal answers Reviewer rpcW Q1.1: "how sensitive is calibration?"

Usage:
  python compute_f1_threshold_sweep.py \
    --id_csv  checkpoints-icml-rebuttal-cost-diffae-only-id-1/diffae_only_results.csv \
    --ood_csv checkpoints-icml-rebuttal-cost-diffae-only-ood-1/diffae_only_results.csv \
    --output_csv results_f1_threshold_sweep.csv
"""

import argparse
import time as _time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, precision_recall_curve


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(csv_path):
    """Returns ordered list of (example_id, loss). Uses combined_loss if available, else mse_loss."""
    df = pd.read_csv(csv_path)
    score_col = "combined_loss" if "combined_loss" in df.columns else "mse_loss"
    print(f"  Using score column: {score_col}")
    return list(zip(df["example_id"], df[score_col]))


def calibrate_f1_threshold(val_scores, val_labels):
    """
    Finds the threshold that maximises F1.

    val_scores: 1D numpy array (higher score = more likely OOD)
    val_labels: 1D numpy array (0=ID, 1=OOD)
    Returns (optimal_threshold, best_f1, best_precision, best_recall)
    """
    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    # precision_recall_curve returns one more entry than thresholds; clamp index
    thresh_idx = min(optimal_idx, len(thresholds) - 1)
    return (thresholds[thresh_idx], f1_scores[optimal_idx],
            precisions[optimal_idx], recalls[optimal_idx])


def evaluate_threshold(id_losses, ood_losses, threshold):
    """Returns dict with F1, TPR, FPR, precision on the given test set."""
    scores = np.concatenate([id_losses, ood_losses])
    labels = np.concatenate([np.zeros(len(id_losses)), np.ones(len(ood_losses))])
    preds = (scores >= threshold).astype(int)

    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"f1": f1, "tpr": tpr, "fpr": fpr, "precision": precision,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def auroc(id_losses, ood_losses):
    scores = np.concatenate([id_losses, ood_losses])
    labels = np.concatenate([np.zeros(len(id_losses)), np.ones(len(ood_losses))])
    return roc_auc_score(labels, scores)


def print_timing(id_csv, ood_csv, n_cal=None):
    """Print per-object inference timing from the stored time_ms column.

    The stored time_ms is per-object sequential inference (batch size 1).
    Reports three estimates for calibration data-collection cost:
      (a) Sequential: n_cal × mean_ms each side, two sides summed
      (b) Batched:    n_cal objects in one forward pass (~1 × mean_ms per side)
                      since DiffAE encode/decode scales with batch size, not
                      linearly with n_cal for small n_cal
      (c) Batched + parallel: ID and OOD on separate GPUs simultaneously → 1 pass
    """
    times = {}
    for label, path in [("ID", id_csv), ("OOD", ood_csv)]:
        df = pd.read_csv(path)
        if "time_ms" in df.columns:
            t = df["time_ms"]
            times[label] = t
            print(f"  {label}: mean={t.mean():.1f}ms  std={t.std():.1f}ms  "
                  f"min={t.min():.1f}ms  max={t.max():.1f}ms  (n={len(t)})")
    if n_cal is not None and "ID" in times and "OOD" in times:
        mean_id  = times["ID"].mean()
        mean_ood = times["OOD"].mean()
        sequential_s = (mean_id * n_cal + mean_ood * n_cal) / 1000.0
        batched_s    = (mean_id + mean_ood) / 1000.0   # 1 forward pass each side
        parallel_s   = max(mean_id, mean_ood) / 1000.0  # both sides in parallel
        print(f"  Calibration data-collection cost for {n_cal} ID + {n_cal} OOD pairs:")
        print(f"    Sequential (batch=1, one GPU): {sequential_s:.1f}s  "
              f"(= {n_cal}×{mean_id:.0f}ms + {n_cal}×{mean_ood:.0f}ms)")
        print(f"    Batched    (batch={n_cal}, one GPU): ~{batched_s:.1f}s  "
              f"(1 forward pass per side, DiffAE encode/decode is batchable)")
        print(f"    Batched + parallel (batch={n_cal}, two GPUs): ~{parallel_s:.1f}s  "
              f"(ID and OOD run simultaneously)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_csv",  required=True,
                        help="diffae_only_results.csv from run_indist=true")
    parser.add_argument("--ood_csv", required=True,
                        help="diffae_only_results.csv from run_indist=false")
    parser.add_argument("--output_csv", default="results_f1_threshold_sweep.csv")
    parser.add_argument("--n_cal", nargs="+", type=int, default=[5, 10, 20, 30],
                        help="Calibration set sizes to sweep")
    args = parser.parse_args()

    id_pairs  = load_results(args.id_csv)
    ood_pairs = load_results(args.ood_csv)

    # Align on shared example IDs (preserving order)
    ood_map = {eid: loss for eid, loss in ood_pairs}
    common = [(eid, loss, ood_map[eid]) for eid, loss in id_pairs if eid in ood_map]
    print(f"Loaded {len(id_pairs)} ID, {len(ood_pairs)} OOD -> {len(common)} matched pairs")

    if len(common) == 0:
        raise RuntimeError("No matching example IDs between ID and OOD runs.")

    id_losses_all  = np.array([x[1] for x in common])
    ood_losses_all = np.array([x[2] for x in common])

    full_auroc = auroc(id_losses_all, ood_losses_all)
    print(f"\nFull-test AUROC = {full_auroc:.4f}")
    print("\nPer-object inference timing (from stored time_ms, excludes startup/model-load):")
    print_timing(args.id_csv, args.ood_csv, n_cal=args.n_cal[0] if args.n_cal else None)

    print("\n" + "="*120)
    print(f"{'N_cal':>6}  {'cal_F1':>7}  {'thresh':>9}  {'thresh_%ile':>11}  "
          f"{'test_F1':>8}  {'TPR':>6}  {'FPR':>6}  {'Precision':>10}  "
          f"{'n_eval':>7}  {'val_AUROC':>10}  {'test_AUROC':>10}  {'cal_compute_ms':>15}")
    print("="*120)

    output_rows = []

    for n_cal in args.n_cal:
        if n_cal > len(common):
            print(f"  N_cal={n_cal}: only {len(common)} pairs available, skipping")
            continue

        # Calibration set: first n_cal pairs
        cal_id  = id_losses_all[:n_cal]
        cal_ood = ood_losses_all[:n_cal]
        cal_scores = np.concatenate([cal_id, cal_ood])
        cal_labels = np.concatenate([np.zeros(n_cal), np.ones(n_cal)])

        _t0 = _time.perf_counter()
        thresh, cal_f1, cal_prec, cal_rec = calibrate_f1_threshold(cal_scores, cal_labels)
        cal_compute_ms = (_time.perf_counter() - _t0) * 1000.0

        val_auroc = auroc(cal_id, cal_ood)

        # Always evaluate on the full test set regardless of N_cal
        metrics = evaluate_threshold(id_losses_all, ood_losses_all, thresh)

        thresh_pctile = np.mean(id_losses_all < thresh) * 100
        print(f"  {n_cal:4d}  {cal_f1:7.3f}  {thresh:9.5f}  {thresh_pctile:10.1f}%  "
              f"{metrics['f1']:8.3f}  {metrics['tpr']:6.3f}  {metrics['fpr']:6.3f}  "
              f"{metrics['precision']:10.3f}  {len(common)*2:7d}  "
              f"val={val_auroc:6.4f}  test={full_auroc:6.4f}  {cal_compute_ms:12.3f}ms")

        output_rows.append({
            "n_cal": n_cal,
            "cal_f1": round(cal_f1, 4),
            "threshold": round(thresh, 6),
            "threshold_id_percentile": round(float(np.mean(id_losses_all < thresh) * 100), 1),
            "test_f1": round(metrics["f1"], 4),
            "test_tpr": round(metrics["tpr"], 4),
            "test_fpr": round(metrics["fpr"], 4),
            "test_precision": round(metrics["precision"], 4),
            "n_eval_id": len(id_losses_all),
            "n_eval_ood": len(ood_losses_all),
            "val_auroc": round(val_auroc, 4),
            "full_auroc": round(full_auroc, 4),
            "cal_compute_ms": round(cal_compute_ms, 3),
        })

    print("="*120)

    if output_rows:
        df_out = pd.DataFrame(output_rows)
        df_out.to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
