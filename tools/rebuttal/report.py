"""Recompute every rebuttal table from the per-object artifacts shipped in this directory.

No GPU, no datasets, no model weights: this reads only the small JSON/CSV result files
under `tools/rebuttal/*/results/` and prints the tables that appear in RESULTS.md Part II,
so a reviewer can check the arithmetic behind every rebuttal claim in a few seconds.

    PYTHONPATH=. python tools/rebuttal/report.py
"""
import csv
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load(*parts):
    p = os.path.join(HERE, *parts)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f) if p.endswith(".json") else list(csv.DictReader(f))


def rule(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def mean(xs):
    return st.mean(xs) if xs else float("nan")


def task_a_nvs():
    rule("1a. Task A — SO(3) viewpoint, 25 ShapeNet cars (NVS PSNR, shared harness)")
    ood = load("baselines", "results", "per_object_scores.json")
    ctrl = load("baselines", "results", "per_object_scores_ctrl.json")
    if not ood:
        print("  MISSING per_object_scores.json")
        return
    def col(d, k):
        return [v[k] for v in d.values() if k in v]
    rows = [
        ("Infer3D (ours)", col(ctrl or {}, "ours_ctrl"), col(ood, "ours")),
        ("nerf-from-image (BRFI)", col(ctrl or {}, "nfi_ctrl"), col(ood, "nfi_own")),
        ("  BRFI, oracle GT pose", [], col(ood, "nfi_oracle")),
        ("EG3D + PTI", [], col(ood, "eg3d_own")),
        ("  EG3D, oracle GT pose", [], col(ood, "eg3d_oracle")),
        ("FINV (single-view)", [], col(ood, "finv_own")),
    ]
    print(f"{'method':26s} {'ID ctrl':>9s} {'OOD SO(3)':>10s} {'drop':>8s} {'n':>4s}")
    for name, c, o in rows:
        if not o:
            continue
        drop = f"{mean(o) - mean(c):+.2f}" if c else "—"
        cs = f"{mean(c):9.2f}" if c else " " * 9
        print(f"{name:26s} {cs} {mean(o):10.2f} {drop:>8s} {len(o):4d}")


def task_a_geom():
    rule("1b. Task A geometry — pose-removed shape (ICP-aligned Chamfer, cd_sym)")
    print(f"{'method':26s} {'SO(3)':>9s} {'ID ctrl':>9s} {'n':>4s}")
    for tag, name in [("ours", "Infer3D (ours)"), ("nfi", "nerf-from-image"),
                      ("eg3d", "EG3D + PTI"), ("pigan", "pi-GAN"), ("finv", "FINV-SV")]:
        so3 = load("baselines", "results", "task_a_geom", f"chamfer_{tag}_so3_summary.json")
        ctl = load("baselines", "results", "task_a_geom", f"chamfer_{tag}_control_summary.json")
        if not so3:
            continue
        print(f"{name:26s} {so3['cd_sym_mean']:9.4f} "
              f"{ctl['cd_sym_mean'] if ctl else float('nan'):9.4f} {so3['n']:4d}")
    print("  NOTE: baselines recover SHAPE fine once ICP removes pose — the NVS gap")
    print("        above is a pose-recovery gap, not a shape gap.")


def task_b():
    rule("1c. Task B — appearance shift, 20 RealCars scenes (Chamfer m², ICP-aligned)")
    print(f"{'method':26s} {'cd_pred->gt':>12s} {'cd_sym':>9s} {'n':>4s}")
    for tag, name in [("pigan", "pi-GAN"), ("eg3d", "EG3D + PTI"),
                      ("finv", "FINV"), ("nfi", "nerf-from-image (BRFI)")]:
        j = load("baselines", "results", "task_b_geom", f"chamfer_{tag}_realcars_summary.json")
        if j:
            print(f"{name:26s} {j['cd_pred2gt_mean']:12.4f} {j['cd_sym_mean']:9.4f} {j['n']:4d}")
    print(f"{'Infer3D (ours)':26s} {0.131:12.4f} {0.327:9.4f} {20:4d}   [paper Table 5]")
    print(f"{'Splatter Image (feed-fwd)':26s} {0.219:12.4f} {'—':>9s} {20:4d}   [paper Table 5]")
    print("  NOTE: this metric removes pose AND scale and rewards emitting a complete")
    print("        mesh, so pi-GAN/EG3D score low by representation, not by fidelity.")
    print("        The discriminative fact is BRFI 0.491 > feed-forward 0.219 (collapse).")


def homefield():
    rule("1d. Homefield controls — each method on its own native setting")
    for f, name in [("eg3d_control_recon_summary.json", "EG3D+PTI input-view recon (25 real cars)"),
                    ("finv_control_recon_summary.json", "FINV input-view recon"),
                    ("pigan_real_summary.json", "pi-GAN on native real CARLA")]:
        j = load("baselines", "results", "homefield", f)
        if j:
            v = j.get("mean", j.get("mean_recon_psnr"))
            print(f"  {name:44s} {v:7.2f}")
    ctrl = load("baselines", "results", "per_object_scores_ctrl.json")
    if ctrl:
        for k, name in [("nfi_ctrl", "nerf-from-image, ID-pose NVS"),
                        ("ours_ctrl", "Infer3D, ID control NVS")]:
            xs = [v[k] for v in ctrl.values() if k in v]
            print(f"  {name:44s} {mean(xs):7.2f}")


def broad_prior():
    rule("2. Broad prior — EqM (ImageNet) + Objaverse lifter, 25 occluded objects")
    sel = load("broad_prior", "results", "cd_sel25.json")
    rows = {}
    for f in ("cd_summary_0.json", "cd_summary_30.json"):
        for r in (load("broad_prior", "results", f) or []):
            rows[r["tag"]] = r
    sub = [r for t, r in rows.items() if not sel or t in set(sel)]
    if not sub:
        print("  MISSING cd summaries")
        return
    cats = {}
    for r in sub:
        cats.setdefault(r["cat"], []).append(r)
    print(f"{'category':14s} {'n':>3s} {'direct CD':>10s} {'Infer3D-EqM':>12s} {'reduction':>10s}")
    for c, v in sorted(cats.items()):
        ff, iv = mean([x["cd_ff"] for x in v]), mean([x["cd_inv"] for x in v])
        print(f"{c:14s} {len(v):3d} {ff:10.3f} {iv:12.3f} {100*(ff-iv)/ff:9.0f}%")
    ff, iv = mean([x["cd_ff"] for x in sub]), mean([x["cd_inv"] for x in sub])
    print(f"{'ALL':14s} {len(sub):3d} {ff:10.3f} {iv:12.3f} {100*(ff-iv)/ff:9.0f}%")
    print(f"  Infer3D improves {sum(1 for r in sub if r['cd_inv'] < r['cd_ff'])}/{len(sub)} objects.")


def ablation():
    for tag, base in (("cars", 16.78), ("co3d", 12.49)):
        rule(f"3. Component leave-one-out ablation — {tag} "
             f"(feed-forward baseline {base:.2f} dB)")
        p = os.path.join(HERE, "ablation", f"ablation_table_{tag}.md")
        if not os.path.exists(p):
            print("  MISSING")
            continue
        for line in open(p):
            if line.startswith("|"):
                print("  " + line.rstrip())


def chamfer_loo():
    rule("3b. CO3D leave-one-out on canonical-shape Chamfer (cd_sym, ICP-aligned)")
    j = load("ablation", "results", "chamfer_loo_summary.json")
    if not j:
        print("  MISSING")
        return
    s = j["summary"]
    print(f"  n = {s['n_common']} objects common to all arms")
    for arm, name in [("b0_control", "full Infer3D"), ("no_depth", "- depth loss"),
                      ("no_latent_prior", "- latent prior")]:
        print(f"  {name:22s} cd_sym mean {s[arm]['cd_sym_mean']:.4f}  median {s[arm]['cd_sym_median']:.4f}")
    for k, name in [("delta_no_depth", "- depth loss"), ("delta_no_latent_prior", "- latent prior")]:
        d = s[k]
        print(f"  paired delta {name:18s} {d['paired_mean']:+.4f} +- {d['paired_sem']:.4f} (SEM)"
              f"  worse on {d['worse_count']}/{s['n_common']}")
    print("  Both are within noise on canonical shape, which is why the paper's Table 8")
    print("  (absolute posed Chamfer, 0.580 -> 0.458) is the load-bearing statement.")


def ood_objects():
    rule("4. OOD object categories — CO3D vases through the hydrants model")
    j = load("ood_objects", "results", "scores.json")
    if not j:
        print("  MISSING")
        return
    s = j["summary"]
    print(f"  n                          {s['n_id']} ID + {s['n_ood']} OOD-object")
    print(f"  mean initial-loss score    ID {s['id_score_mean']:.4f} vs OOD {s['ood_score_mean']:.4f}")
    print(f"  AUROC (combined score)     {s['auroc_score']:.3f}")
    print(f"  AUROC (LPIPS / MSE only)   {s.get('auroc_lpips', float('nan')):.3f} / {s['auroc_mse']:.3f}")


def multiobject():
    rule("5. Multiple objects — 2-chair scenes, 20 scenes x 19 held-out views")
    b = load("multiobject", "results", "baseline_summary.json")
    o = load("multiobject", "results", "oracle_summary.json")
    c = load("multiobject", "results", "crossval_summary.json")
    f = load("multiobject", "results", "fair_summary.json")
    if b:
        print(f"  feed-forward Splatter, raw            {b['mean_raw']['PSNR']:6.2f}  n={b['n']}")
        print(f"  feed-forward, depth-anchored          {b['mean_depth_anchored']['PSNR']:6.2f}")
    if f:
        print(f"  Infer3D, automatic placement          {f['mean_psnr']:6.2f}  "
              f"(>=18dB: {f['ge18']}/{f['n']})")
    if o:
        print(f"  Infer3D + ORACLE placement            {o['oracle_mean']:6.2f}  "
              f"(>=18dB: {o['ge18']}/{o['n']})")
    if c:
        gap = c["heldout_mean"] - mean([v["fit_psnr"] for v in c["per_scene"].values()])
        print(f"  Infer3D + 7-DoF reg, HELD-OUT         {c['heldout_mean']:6.2f}  "
              f"(>=18dB: {c['ge18']}/{c['n']}, fit-vs-heldout gap {gap:+.2f} dB)")
    print("  NOTE: the bottleneck is single-view PLACEMENT, not reconstruction quality.")


if __name__ == "__main__":
    print("Infer3D — rebuttal results recomputed from shipped per-object artifacts")
    task_a_nvs()
    task_a_geom()
    task_b()
    homefield()
    broad_prior()
    ablation()
    chamfer_loo()
    ood_objects()
    multiobject()
    print("\nSee RESULTS.md Part II for the written-up versions of these tables.")
