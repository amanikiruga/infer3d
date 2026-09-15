"""Assemble the RealCars table from the per-scene Chamfer distances.

Reads results/chamfer_direct.csv, written by eval_chamfer_direct.py. Reports the mean
and median over all scenes plus the paired per-scene difference, which is the statistic
the claim rests on: the absolute Chamfer varies a lot between scenes, the sign of the
difference much less.

    python experiments/realcars/score.py
"""
import csv
import math
import os
import statistics as st

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "chamfer_direct.csv")


def main():
    rows = list(csv.DictReader(open(CSV)))
    ours = [float(r["cd_ours_o2g"]) for r in rows]
    base = [float(r["cd_baseline_o2g"]) for r in rows]
    diff = [o - b for o, b in zip(ours, base)]
    half = 1.96 * st.stdev(diff) / math.sqrt(len(diff))

    print(f"RealCars, Chamfer pred->gt in m^2, n={len(rows)} scenes\n")
    print(f"{'method':22s} {'mean':>8s} {'median':>8s}")
    print(f"{'Splatter Image':22s} {st.mean(base):8.3f} {st.median(base):8.3f}")
    print(f"{'Infer3D':22s} {st.mean(ours):8.3f} {st.median(ours):8.3f}")
    print(f"\npaired Infer3D - Splatter: {st.mean(diff):+.3f} "
          f"[{st.mean(diff) - half:+.3f}, {st.mean(diff) + half:+.3f}]  "
          f"Infer3D better on {sum(1 for d in diff if d < 0)}/{len(diff)}")


if __name__ == "__main__":
    main()
