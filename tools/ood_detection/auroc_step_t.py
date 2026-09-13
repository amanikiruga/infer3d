"""
Compute AUROC of OOD detection at each step T using saved loss trajectories.

Usage:
  python compute_auroc_step_t.py \
    --id_dir  checkpoints-icml-rebuttal-cost-ood-det-id \
    --ood_dir checkpoints-icml-rebuttal-cost-ood-det-ood \
    --output_csv results_auroc_step_t.csv

Each run of diffae_co3d_encoder_depth.py saves a loss_trajectory.json under
  <prefix>/<example_id>/loss_trajectory.json

Timing columns:
  - step=0: step0_pinned_eval_time_ms (encode->decode->3DGS->render->loss, no proposals)
  - step T>0: optim_elapsed_time_s (time inside the optimization loop only, from iteration 1)

AUROC columns:
  - best_loss_AUROC: uses best_loss@T (best across all candidates at step T)
  - pinned_loss_AUROC: uses pinned_loss@T (directly-encoded candidate, no optimization)
"""
import argparse
import glob
import json
import os
import csv

import numpy as np
from sklearn.metrics import roc_auc_score


EVAL_STEPS = [0, 10, 30, 50, 100, 150, 300, 783]


def load_trajectories(directory):
    pattern = os.path.join(directory, "**", "loss_trajectory.json")
    paths = glob.glob(pattern, recursive=True)
    trajectories = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        trajectories.append(data)
    return trajectories


def get_entry_at_step(trajectory_list, target_step):
    """Return the trajectory entry at the closest iteration <= target_step."""
    candidates = [e for e in trajectory_list if e["iteration"] <= target_step]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["iteration"])


def safe_auroc(labels, scores):
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_dir", required=True)
    parser.add_argument("--ood_dir", required=True)
    parser.add_argument("--output_csv", default="results_auroc_step_t.csv")
    args = parser.parse_args()

    id_trajs = load_trajectories(args.id_dir)
    ood_trajs = load_trajectories(args.ood_dir)
    print(f"Loaded {len(id_trajs)} ID, {len(ood_trajs)} OOD trajectories")

    rows = []

    # Pre-compute per-example final best_loss (last trajectory entry) for % of final columns
    def get_final_losses(traj_list):
        finals = []
        for t in traj_list:
            entries = [e for e in t["trajectory"] if e.get("best_loss") is not None]
            if entries:
                finals.append(entries[-1]["best_loss"])
        return finals

    ood_final_losses = get_final_losses(ood_trajs)
    id_final_losses  = get_final_losses(id_trajs)
    mean_ood_final = float(np.mean(ood_final_losses)) if ood_final_losses else float("nan")
    mean_id_final  = float(np.mean(id_final_losses))  if id_final_losses  else float("nan")

    # ---- Step 0: feedforward-only detection (no optimization, no proposals) ----
    id_step0_losses = [t["step0_pinned_loss"] for t in id_trajs if "step0_pinned_loss" in t]
    ood_step0_losses = [t["step0_pinned_loss"] for t in ood_trajs if "step0_pinned_loss" in t]
    id_step0_times = [t["step0_pinned_eval_time_ms"] for t in id_trajs if "step0_pinned_eval_time_ms" in t]
    ood_step0_times = [t["step0_pinned_eval_time_ms"] for t in ood_trajs if "step0_pinned_eval_time_ms" in t]

    median_step0_time_ms = float(np.median(id_step0_times + ood_step0_times)) if (id_step0_times or ood_step0_times) else float("nan")
    labels_step0 = [0] * len(id_step0_losses) + [1] * len(ood_step0_losses)
    scores_step0 = id_step0_losses + ood_step0_losses
    auroc_step0 = safe_auroc(labels_step0, scores_step0)

    mean_ood_loss_s0 = float(np.mean(ood_step0_losses)) if ood_step0_losses else float("nan")
    mean_id_loss_s0  = float(np.mean(id_step0_losses))  if id_step0_losses  else float("nan")
    pct_ood_s0 = 100.0 * mean_ood_loss_s0 / mean_ood_final if mean_ood_final else float("nan")
    pct_id_s0  = 100.0 * mean_id_loss_s0  / mean_id_final  if mean_id_final  else float("nan")

    row0 = {
        "step_T": 0,
        "method": "feedforward_only",
        "median_wall_time_ms": round(median_step0_time_ms, 1),
        "mean_ood_best_loss": round(mean_ood_loss_s0, 5),
        "pct_of_final_ood":   round(pct_ood_s0, 1),
        "mean_id_best_loss":  round(mean_id_loss_s0, 5),
        "pct_of_final_id":    round(pct_id_s0, 1),
        "best_loss_AUROC": "n/a",
        "pinned_loss_AUROC": round(auroc_step0, 4) if not np.isnan(auroc_step0) else "nan",
        "n_id": len(id_step0_losses),
        "n_ood": len(ood_step0_losses),
    }
    rows.append(row0)
    print(f"Step  0 (ff)  | time={median_step0_time_ms:.0f}ms  | "
          f"OOD loss={mean_ood_loss_s0:.5f} ({pct_ood_s0:.1f}% of final)  "
          f"ID loss={mean_id_loss_s0:.5f} ({pct_id_s0:.1f}%)  "
          f"AUROC={auroc_step0:.4f}")

    # ---- Steps T > 0: optimization loop ----
    for step in [s for s in EVAL_STEPS if s > 0]:
        id_best, id_pinned, id_times = [], [], []
        for t in id_trajs:
            entry = get_entry_at_step(t["trajectory"], step)
            if entry is not None:
                id_best.append(entry["best_loss"])
                if entry.get("pinned_loss") is not None:
                    id_pinned.append(entry["pinned_loss"])
                if entry.get("optim_elapsed_time_s") is not None:
                    id_times.append(entry["optim_elapsed_time_s"])

        ood_best, ood_pinned, ood_times = [], [], []
        for t in ood_trajs:
            entry = get_entry_at_step(t["trajectory"], step)
            if entry is not None:
                ood_best.append(entry["best_loss"])
                if entry.get("pinned_loss") is not None:
                    ood_pinned.append(entry["pinned_loss"])
                if entry.get("optim_elapsed_time_s") is not None:
                    ood_times.append(entry["optim_elapsed_time_s"])

        median_time_s = float(np.median(id_times + ood_times)) if (id_times or ood_times) else float("nan")

        mean_ood_loss = float(np.mean(ood_best)) if ood_best else float("nan")
        mean_id_loss  = float(np.mean(id_best))  if id_best  else float("nan")
        pct_ood = 100.0 * mean_ood_loss / mean_ood_final if mean_ood_final else float("nan")
        pct_id  = 100.0 * mean_id_loss  / mean_id_final  if mean_id_final  else float("nan")

        labels_best = [0] * len(id_best) + [1] * len(ood_best)
        auroc_best = safe_auroc(labels_best, id_best + ood_best)

        labels_pinned = [0] * len(id_pinned) + [1] * len(ood_pinned)
        auroc_pinned = safe_auroc(labels_pinned, id_pinned + ood_pinned)

        row = {
            "step_T": step,
            "method": "optimization_loop",
            "median_wall_time_ms": round(median_time_s * 1000, 1),
            "mean_ood_best_loss": round(mean_ood_loss, 5),
            "pct_of_final_ood":   round(pct_ood, 1),
            "mean_id_best_loss":  round(mean_id_loss, 5),
            "pct_of_final_id":    round(pct_id, 1),
            "best_loss_AUROC": round(auroc_best, 4) if not np.isnan(auroc_best) else "nan",
            "pinned_loss_AUROC": round(auroc_pinned, 4) if not np.isnan(auroc_pinned) else "nan",
            "n_id": len(id_best),
            "n_ood": len(ood_best),
        }
        rows.append(row)
        print(f"Step {step:3d}        | time={median_time_s*1000:.0f}ms  | "
              f"OOD loss={mean_ood_loss:.5f} ({pct_ood:.1f}% of final)  "
              f"ID loss={mean_id_loss:.5f} ({pct_id:.1f}%)  "
              f"AUROC={auroc_best:.4f}")

    fieldnames = ["step_T", "method", "median_wall_time_ms",
                  "mean_ood_best_loss", "pct_of_final_ood",
                  "mean_id_best_loss",  "pct_of_final_id",
                  "best_loss_AUROC", "pinned_loss_AUROC", "n_id", "n_ood"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
