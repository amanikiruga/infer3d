# Results — every number in the paper and the rebuttal

This file lists every quantitative claim in the Infer3D paper and in the rebuttal, the
artifact it comes from, and whether it was independently re-derived in this repository.

Two conventions are used throughout and are worth stating once:

* **Full set, no selection.** Every number below is a mean over all evaluated
  objects/sequences. Nothing is filtered or chosen to match a target.
* **Paired comparisons.** Where a claim is "A beats B", the honest statistic is the
  per-object paired difference on the same objects, with a 95% CI. Those are given.

Status legend: **✅ reproduced** (re-derived here from per-object artifacts or re-run) ·
**⚠️ partial** (reproduces with a caveat that changes the reading) ·
**❌ does not reproduce** · **📄 recorded** (artifact shipped, not independently re-run here).

---

# Part I — Main paper

See [REPRODUCTION.md](REPRODUCTION.md) for the full audit, including the two
methodological problems it turned up (a table driver that fitted subsets to published
targets, and oracle particle selection on ShapeNet-NMR). Summary:

| Table | Claim | Status |
|---|---|---|
| 1 | CO3D OOD: Infer3D > Splatter Image | ✅ all OOD cells, all 4 metrics, CIs exclude 0; independently reproduced **from scratch** end-to-end (n=11 fresh runs, +3.27 dB, all 4 metrics significant) |
| 2 SE(3) | ShapeNet-NMR SE(3): +2.03 dB | ⚠️ honest gain +0.84 dB; ~1.6 dB was oracle selection |
| 2 SO(3) | ShapeNet-NMR SO(3): +2.55 dB | ❌ reverses: −2.17 dB under the standard protocol |
| 3 | RE10K fisheye: Infer3D ≫ CATSplat | ✅ +6.30 dB, 96% win, n=160 |
| 4 | Adaptive routing beats both fixed policies | ⚠️ feed-forward column exact; published "Adaptive" column matches *oracle* routing |
| 5 | RealCars Chamfer 0.131 vs 0.219 | ✅ exact on all headline rows |
| 6 | OOD AUROC 0.945 / 0.961 | ✅ 0.971 (Hydrants), 0.939 (Vases) |
| 7 | Calibration F1 0.965 / 0.944 | ✅ exact |
| 8 | Loss ablation ladder | ⚠️ not runnable as shipped (2 bugs, fixed); see Part II §3 for the rebuttal's ablation |
| 9 | Trivial TTT fails | ⚠️ holds qualitatively; published −3.32 dB gap is a cross-protocol artifact (honest: −0.90 dB) |

Headline paired deltas (Infer3D − feed-forward, same objects, full set):

| Cell | n | PSNR | SSIM | LPIPS | CD |
|---|---|---|---|---|---|
| CO3D Hydrants OOD (DiffAE) | 41 | **+1.88** (71% win) | +0.049 | −0.075 | −0.274 |
| CO3D Hydrants OOD — *from-scratch rerun* | 11 | **+3.27** (73% win) | +0.090 | −0.126 | −0.468 |
| CO3D Hydrants OOD (StyleGAN) | 29 | **+1.69** | +0.045 | −0.069 | −0.306 |
| CO3D Vases OOD (DiffAE) | 27 | **+0.76** | +0.076 | −0.041 | −0.155 |
| RE10K OOD fisheye | 160 | **+6.30** (96% win) | +0.286 | −0.137 | — |
| RealCars (CD only) | 20 | — | — | — | **−0.088** (80% win) |

---

# Part II — Rebuttal experiments

Code and artifacts: [`tools/rebuttal/`](tools/rebuttal/). Every number in this part was
recomputed here from the per-object artifacts shipped in that directory.

## 1. Optimization / inversion baselines (AC Priority 1, Reviewer akZb)

Four inversion methods were compared: **pi-GAN**, **3D-GAN-Inversion (EG3D+PTI)**,
**nerf-from-image (BRFI)** and **FINV**, on two OOD axes.

### Task A — SO(3) viewpoint, 25 ShapeNet cars

Novel-view PSNR at each method's **estimated** input pose (no ICP), shared harness.
Source: `tools/rebuttal/baselines/results/per_object_scores{,_ctrl}.json`. ✅ recomputed.

| Method | ID control | OOD SO(3) | ID → OOD drop | n |
|---|---|---|---|---|
| **Infer3D (ours)** | **20.72** | **19.23** | **−1.49 dB** | 25 |
| nerf-from-image (BRFI) | 19.85 | 14.28 | −5.57 dB | 25 |
| nerf-from-image, **oracle GT pose** | — | 17.66 | — | 25 |
| EG3D + PTI | — | 12.32 | — | 25 |
| EG3D + PTI, oracle GT pose | — | 12.48 | — | 16 |
| FINV (single-view) | — | 12.29 | — | 25 |

The reading: generic inversion collapses under SO(3) (12.3–14.3 dB) while Infer3D loses only
1.5 dB from its own ID control. Handing BRFI the ground-truth camera it fails to estimate
recovers it only to 17.66 — still below Infer3D's 19.23, which had to infer pose itself.
So pose initialization explains part of the gap, not all of it.

### Task A geometry — pose-removed shape control (ICP-aligned Chamfer)

Source: `baselines/results/task_a_geom/*_summary.json`. ✅ recomputed.

| Method | cd_sym SO(3) | cd_sym ID control |
|---|---|---|
| Infer3D (ours) | 0.0091 | 0.0077 |
| nerf-from-image | **0.0037** (n=24) | 0.0010 |
| EG3D + PTI | 0.0127 | 0.0086 |
| pi-GAN | 0.0075 | 0.0058 |
| FINV-SV | 0.0119 | 0.0096 |

**This is the honest and interesting control, and it does not favour us**: once pose is
removed by ICP, the baselines recover shape fine — BRFI's Chamfer is better than ours
(partly representation: ours is Gaussian centres from a single view, BRFI exports a clean
SDF mesh). The NVS gap in the table above is therefore a **pose-recovery** gap, not a
shape gap. That separation is exactly what was asked for, and it should be reported.

### Task B — appearance shift, 20 RealCars scenes (Chamfer m², ICP-aligned)

Source: `baselines/results/task_b_geom/*_summary.json` + paper Table 5. ✅ recomputed.

| Method | representation | cd_pred→gt | cd_sym |
|---|---|---|---|
| pi-GAN | CARLA / full mesh | 0.088 | 0.112 |
| **Infer3D (ours)** | SRN + DINOv2 / single-view splats | 0.131 | 0.327 |
| EG3D + PTI | ShapeNet / full mesh | 0.154 | 0.384 |
| FINV | / full mesh | 0.169 | 0.310 |
| Splatter Image (feed-forward) | SRN / single-view splats | 0.219 | — |
| SF3D | Objaverse / mesh | 0.242 | — |
| LGM | Objaverse / splats | 0.405 | — |
| nerf-from-image (BRFI) | shapenet_cars / SDF | 0.491 | 0.431 |

**Do not read this as "ours is best".** The metric removes pose and scale, and
cd_pred→gt rewards emitting a complete plausible car, so it interacts with the output
representation: pi-GAN (0.088) and EG3D (0.154) score well by producing a clean generic
full mesh that overlaps a real car after alignment, not by recovering the right car or
pose. The two defensible readings are: (i) BRFI — the direct NeRF-inversion analogue of
our method, same single-object setting — genuinely collapses to 0.491, *worse than the
feed-forward lifter it builds on*; (ii) among comparable single-view methods, ours
0.131 < Splatter 0.219 < LGM 0.405, which is the paper's Table 5 claim.

### Homefield controls (each method on its own native setting)

Source: `baselines/results/homefield/*.json`. ✅ recomputed.

| Control | value |
|---|---|
| EG3D+PTI input-view reconstruction, 25 real cars | 21.50 |
| FINV input-view reconstruction | 22.08 |
| pi-GAN on its native real CARLA images | 26.05 |
| nerf-from-image, ID pose NVS | 19.85 |
| Infer3D, ID control NVS | 20.72 |

Every baseline works in its own regime; the gap opens only under the OOD shift. This
rules out "we mis-ran the baselines" as an explanation.

### ⚠️ Provenance note on Infer3D's ShapeNet-Cars number

Three different figures for "Infer3D on 25 OOD SO(3) cars" appear across the artifacts,
because three harnesses were used. They are all real; they are not interchangeable:

| value | harness | use it for |
|---|---|---|
| **19.23** (ID ctrl 20.72) | shared baseline harness, identical renderer/metric for all methods | **comparisons against the baselines** (the table above) |
| 20.29 (ID 21.43) | ablation harness, `tools/rebuttal/ablation/` control arm | the leave-one-out ablation, §3 |
| 20.04 (ID 21.43) | paper harness | the paper's own tables |

The posted rebuttal quotes 21.43 / 20.29. Against the baselines the correct number is
**19.23 vs BRFI 14.28**, since only that pair shares a harness. The claim is unaffected
either way, but a reviewer comparing the two tables will notice, so the release states it.

## 2. Broad prior — EqM + Objaverse, occluded objects (Reviewer akZb Q2/Q5)

One frozen class-conditional **Equilibrium Matching** energy model (ImageNet) + one frozen
**Objaverse Splatter Image** lifter, across 25 partially-occluded objects in 6 categories.
Source: `tools/rebuttal/broad_prior/results/cd_summary_*.json` + `cd_sel25.json`.
✅ recomputed — matches the posted table exactly.

| Category | n | direct lifter CD ↓ | Infer3D-EqM CD ↓ | reduction |
|---|---|---|---|---|
| Teapot | 6 | 0.214 | **0.157** | 27% |
| Mug | 3 | 0.225 | **0.171** | 24% |
| Vase | 6 | 0.278 | **0.180** | 35% |
| Pineapple | 6 | 0.147 | **0.096** | 35% |
| Wine bottle | 3 | 0.145 | **0.109** | 25% |
| Helmet | 1 | 0.214 | **0.199** | 7% |
| **All** | **25** | **0.206** | **0.145** | **30%** |

Infer3D improves **25/25 objects**. This is the strongest evidence for the modularity
claim: neither the prior nor the lifter is category-specific here.

## 3. Component leave-one-out ablation (Reviewer uH4P W4)

Source: `tools/rebuttal/ablation/ablation_table_{cars,co3d}.md`. ✅ recomputed.
Identical examples and initializations per arm; Δ is the paired difference.

**ShapeNet Cars, StyleGAN prior, 25 OOD examples.** Feed-forward baseline **16.78 dB**.

| Removed component | PSNR ↑ | Δ paired | SSIM | LPIPS |
|---|---|---|---|---|
| — (full Infer3D) | **20.29 ± 0.41** | — | 0.860 | 0.145 |
| rendering-parameter (pose) optimization | 17.19 ± 0.45 | **−3.10 ± 0.42** | 0.820 | 0.187 |
| latent-code optimization | 18.02 ± 0.37 | **−2.27 ± 0.35** | 0.833 | 0.180 |
| both gradient optimizations (search only) | 16.21 ± 0.36 | **−4.08 ± 0.32** | 0.806 | 0.205 |
| MSE loss | 19.09 ± 0.40 | −1.20 ± 0.25 | 0.850 | 0.152 |
| LPIPS loss | 19.76 ± 0.48 | −0.53 ± 0.25 | 0.856 | 0.163 |
| noise-map regularizer | 20.17 ± 0.41 | −0.12 ± 0.15 | 0.860 | 0.144 |
| pose multi-start 30 → 10 | 19.97 ± 0.43 | −0.32 ± 0.17 | 0.857 | 0.151 |
| latent multi-start 20 → 2 | 19.49 ± 0.41 | −0.80 ± 0.28 | 0.849 | 0.155 |

Removing both gradient optimizations drops **below** the feed-forward baseline
(16.21 < 16.78): the multi-start search alone is not what produces the gain.

**CO3D Hydrants, DiffAE prior, 18 objects.** Feed-forward baseline **12.49 dB**.

| Removed component | PSNR ↑ | Δ paired |
|---|---|---|
| — (full Infer3D) | 15.79 ± 0.60 | — |
| latent-code optimization | 14.10 ± 0.29 | **−1.69 ± 0.50** |
| rendering-parameter optimization | 14.28 ± 0.51 | **−1.51 ± 0.48** |
| LPIPS loss | 15.10 ± 0.33 | −0.69 ± 0.58 |
| MSE loss | 15.29 ± 0.48 | −0.50 ± 0.36 |
| latent prior (w_reg) | 16.03 ± 0.51 | +0.24 ± 0.38 |
| depth loss | 16.14 ± 0.57 | +0.35 ± 0.44 |

⚠️ On CO3D **novel-view PSNR**, removing the depth loss and the latent prior *helps*
slightly (both within noise). The same holds on **canonical-shape Chamfer** (n=17 objects
common to all arms, ICP-aligned cd_sym):

| arm | cd_sym mean | paired Δ vs full |
|---|---|---|
| full Infer3D | 0.2178 | — |
| − depth loss | 0.2228 | +0.0050 ± 0.0304 (SEM), worse on 8/17 |
| − latent prior | 0.2035 | −0.0143 ± 0.0250 (SEM), worse on 9/17 |

Both are within noise once pose and scale are removed by ICP. Their benefit is therefore
to **absolute pose/scale recovery of the posed reconstruction**, which neither rendered
PSNR nor gauge-free Chamfer tests — it shows up only in the paper's Table 8 (absolute
posed Chamfer, 0.580 → 0.458), which is the load-bearing statement for these two terms.
The paper reports only that last measurement; all three should be stated.

## 4. OOD object categories (Reviewer akZb Q2)

Feeding CO3D **vases** through the **hydrants** model — a content shift the prior cannot
represent. Source: `tools/rebuttal/ood_objects/results/scores.json`. ✅ recomputed.

| quantity | value |
|---|---|
| n | 30 ID + 30 OOD-object |
| mean initial-loss score, ID | 0.0511 |
| mean initial-loss score, OOD object | 0.2424 |
| **AUROC (combined score)** | **0.988** |
| AUROC (LPIPS only) | 0.988 |
| AUROC (MSE only) | 0.959 |

The prior "snaps" each vase onto the nearest hydrant and the residual exposes it: Infer3D
**fails loudly** where a feed-forward model fails silently, at feed-forward cost.

## 5. Multiple objects / compositionality (Reviewer akZb Q3)

Two-chair scenes from a single image, with prior and lifter trained only on single
chairs; 20 scenes × 19 held-out views.
Source: `tools/rebuttal/multiobject/results/*.json`. ✅ recomputed.

| Setting | PSNR | SSIM | LPIPS | ≥18 dB |
|---|---|---|---|---|
| Feed-forward Splatter on the 2-chair image (raw) | 13.39 | 0.794 | 0.306 | — |
| Feed-forward, depth-anchored variant | 14.25 | 0.648 | 0.381 | — |
| Infer3D, automatic single-view placement | 14.17 | 0.728 | 0.261 | 3/20 |
| Infer3D + per-object **oracle** placement | 18.42 | — | — | 11/20 |
| Infer3D + per-object 7-DoF registration, **cross-validated held-out** | **18.24** | 0.80 | 0.19 | 10/20 |

The fit-vs-held-out gap is only 0.24 dB, so the registration generalizes rather than
overfits. The honest conclusion: **the bottleneck is single-view object placement, not
reconstruction quality** — with placement solved, the unchanged reconstructions reach
18.2–18.4 dB.

## 6. In-distribution trade-off (AC Priority 2, Reviewers XsZX / uH4P)

Optimizer-only ID performance vs the feed-forward lifter:

| ID benchmark | feed-forward | Infer3D optimization | difference |
|---|---|---|---|
| CO3D Hydrants | 21.48 | 20.55 | −0.93 dB |
| ShapeNet (11 cat.) | 24.27 | 21.61 | −2.66 dB |
| RealEstate10K | 25.41 | 24.03 | −1.38 dB |
| ShapeNet Cars | 23.81 | 21.43 | −2.38 dB |

The gap is **not uniform quality loss**: it is concentrated in a minority of objects where
inverting the 2D prior collapses to a wrong mode (DiffAE 4/18 objects > 4 dB; median gap
elsewhere 1.7 dB; StyleGAN weaker at 14/18). Split by view, the feed-forward advantage is
**+5.7 dB near the input view** but only **+0.9 dB at the most occluded view** — i.e. the
gap is largely the feed-forward model copying visible input detail, and Infer3D recovers
the underlying 3D nearly as well. Those failure objects are exactly the high-initial-loss
cases the detector flags (AUROC 0.97), so the router sends them to the feed-forward path.

Analysis code: `tools/rebuttal/analysis/`.

## 7. Optimization dynamics, cost, and Algorithm 1 (AC Priorities 3–4)

Loss vs iteration, 25 OOD ShapeNet Cars (📄 recorded, from the rebuttal):

| iteration | MSE ↓ | λ_MSE | LPIPS ↓ | λ_LPIPS | L_recon |
|---|---|---|---|---|---|
| 0 | 0.0185 | 1 | 0.1717 | 0.5 | 0.1044 |
| 10 | 0.0149 | 1 | 0.1409 | 0.5 | 0.0854 |
| 50 | 0.0085 | 1 | 0.1030 | 0.5 | 0.0600 |
| 150 | 0.0066 | 1 | 0.0828 | 0.5 | 0.0480 |
| 300 | 0.0060 | 1 | 0.0718 | 0.5 | 0.0419 |
| 500 | 0.0047 | 10 | 0.0617 | 2 | 0.1704 |
| 782 | **0.0026** | 2 | **0.0571** | 2 | 0.1194 |

MSE falls 86% and LPIPS 67%. Cost on one 80 GB A100:

| configuration | latency | CO3D Hydrants OOD PSNR |
|---|---|---|
| Splatter Image (feed-forward) | 56 ms | 15.73 |
| OOD detection | 286 ms | — |
| Infer3D, 100 steps | 1.2 min | 16.98 |
| Infer3D, converged | 2.8 min | 18.46 |

Peak memory 48 GB (37.5 GB for the default search). Detection latency was re-measured
here at **283 ms** (n=43), matching the quoted 286 ms. ✅

⚠️ **Convergence wall-clock does not reproduce.** Running the full 783-step schedule end
to end on one H100 from the documented command took **1201 s (20 min)** for one CO3D
Hydrants object, against the quoted 2.8 min on an A100. Per-step cost falls steeply as
particles are pruned (≈41 s for the first 600-particle step → 1.6 steps/s after t=302),
so the 600-hypothesis search dominates and the quoted figure is hard to reach at this
schedule. The step-0 detection latency, which is the number the adaptive system depends
on, does reproduce. The same discrepancy shows up in the Table 4 runtime column above.

**Algorithm 1 constants** (all fixed, iteration-indexed, identical across datasets) —
`tools/rebuttal/algorithm/`:

| symbol | value / schedule | role |
|---|---|---|
| R | **600** = 30 rotations × 20 latents | initial joint hypotheses |
| N | **783** (steps 0–782) | total optimization steps |
| z₁, θ₁ | E(I), identity pose when an encoder is available | encoder warm start replacing the first random hypothesis |
| k_t | 600 → 32 (t=3) → 10 (t=122) → 5 (t=302) | active particle population |
| λ_prior | **0.001**, constant | latent-prior weight |
| λ_MSE(t) | 1 → 10 (t=302) → 2 (t=732) | weight in L_recon |
| λ_LPIPS(t) | 0.5 → 2 (t=302) | weight in L_recon |
| η_z | 0.01, 5% cosine ramp-up, final 25% ramp-down | latent learning rate |
| η_θ | 0.01, constant | rendering-parameter learning rate |
| B | **10** | max particles optimized simultaneously per GPU step |

⚠️ Verified against the code during the audit: these schedules live in a six-stage table
inside `tools/optimize_co3d_diffae.py` that writes directly into module globals, so the
*effective* loss weights are the schedule's, not the config's. The paper should say so.

## 8. Real-world ground-truth geometry (Reviewer uH4P W3)

The RealCars evaluation is quantitative against measured geometry, not just plausibility:
single-image reconstructions are compared to a metric-scale multi-view pseudo-GT (Gaussian
Splatting fit to the full ARKit capture with LiDAR-grade poses) by ICP-aligned Chamfer in
m². Protocol and code: `tools/realcars/`; the written response is
`tools/rebuttal/realworld_gt/RESPONSE_REALWORLD_GT.md`. Numbers are Table 5 above. ✅

---

## Reproducing any of this

Everything that can be checked without a GPU, in one command:

```bash
scripts/verify.sh          # = make_tables.py + rebuttal/report.py
```

Individually:

```bash
PYTHONPATH=. python3 tools/make_tables.py                # paper Tables 1 & 2
PYTHONPATH=. python3 tools/rebuttal/report.py            # all of Part II
PYTHONPATH=. python  tools/realcars/build_table_4col.py  # Table 5 (needs numpy)
PYTHONPATH=. python  tools/re10k/make_table3.py <table3_per_seq.json>   # Table 3
```

The first two need only the Python standard library and the result files bundled in the
repository. Table 3's `table3_per_seq.json` is produced by `tools/re10k/eval_table3.py`
and is not bundled (it requires the CATSplat lifter); `scripts/reproduce_table3.sh` runs
that end to end. To re-run an experiment rather than re-score it, see the README and
`tools/rebuttal/README.md`.
