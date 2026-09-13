# Reproduction status — Infer3D (main paper)

This is the audit of the **main paper's** tables. For the rebuttal experiments — the
inversion baselines, the EqM/Objaverse broad-prior study, the component ablation, OOD
object categories, multi-object scenes, and the dynamics analysis — see
[RESULTS.md](RESULTS.md) Part II, which is recomputed by
`python tools/rebuttal/report.py`. [RESULTS.md](RESULTS.md) also carries a one-page
summary of everything below.

Audited 2026-09-10 by re-deriving every table from the underlying per-object artifacts and
by running the clean repo end-to-end. The per-object CSVs behind Tables 1 and 2, and
behind the eval-variant A/B below, are bundled in `assets/eval_csvs/` so every number
here can be rechecked with `python tools/make_tables.py` (no GPU, no datasets). **Every number below is a mean over the full set of
evaluated objects. Nothing is subset-selected.** The published value is quoted in
brackets for reference only.

## Read this first: how the previously-reported numbers were produced

The earlier version of this file reported that all of Tables 1, 2 and 3 reproduced. That
conclusion was an artifact of the tooling, not a result:

* `tools/make_tables.py` contained a `best_subset()` routine that searched over subsets of
  the test set (any subset of ≥45% of objects) and returned **the subset whose mean best
  matched a hardcoded paper target**. `tools/re10k/make_table3.py` did the same by greedy
  backward elimination. A procedure like this reports the target back to you almost
  regardless of the data, so it is not evidence of anything.
* Both drivers have been replaced with honest ones: full-set means, 95% CIs, medians, and
  **paired per-object deltas**, which is the statistic the paper's claims actually rest on.

A second, independent issue affects ShapeNet-NMR only: the column the paper reports
(`*_novel_best`) is the maximum over the particle population of the **ground-truth**
novel-view PSNR — oracle selection on the test set. The legitimate column is
`*_novel_known_best`, the hypothesis the optimizer picked using its own loss against the
input image. Both are now printed side by side.

## Summary

| Table | What | Verdict |
|---|---|---|
| 1 | CO3D Hydrants/Vases | **Reproduces.** Every OOD cell favours Infer3D on all 4 metrics with 95% CIs excluding zero. Effect sizes smaller than published. |
| 2 | ShapeNet-NMR SE(3) | **Reproduces weakly.** +0.84 dB honest vs +2.03 dB published; ~1.6 dB of the published gain is oracle selection. |
| 2 | ShapeNet-NMR SO(3) | **Does not reproduce.** Infer3D is 2.17 dB *worse* than the baseline under the standard protocol. |
| 3 | RE10K fisheye | **Reproduces strongly.** +6.30 dB over CATSplat, 96% win rate, n=160, no selection. |
| 4 | Adaptive inference | **Partial.** Feed-forward column exact; the published "Adaptive" column matches *oracle* routing; F1 column and runtimes do not reproduce. |
| 5 | RealCars Chamfer | **Reproduces exactly** on the headline rows. |
| 6 | OOD detection AUROC | **Reproduces** for Hydrants and Vases. RE10K cell and the "Image alignment" baseline row not reproducible from available artifacts. |
| 7 | Calibration sensitivity | **Reproduces** (F1 values match exactly). |
| 8 | Loss ablation | **Not runnable as shipped** (two bugs, now fixed). 4-object pilot does not show the published monotone ladder under honest selection. |
| 9 | Trivial test-time training | **Qualitatively reproduces**, published gap inflated ~4-8x by a cross-protocol comparison. |

---

## Table 1 — CO3D (Hydrants, Vases)

Not affected by the oracle-selection issue: the ICP eval rebuilds splats from
`best_input_image`/`best_rotation_matrix`, which are selected by the optimizer's decision
loss against the input image.

Paired per-object delta, **Infer3D minus Splatter Image on the same objects**, full set:

| Cell | n | PSNR | SSIM | LPIPS | CD |
|---|---|---|---|---|---|
| Hydrants OOD (DiffAE) | 36 | **+1.96** ±1.20 (72%) | +0.054 ±0.022 (81%) | −0.080 ±0.024 (86%) | −0.366 ±0.175 (83%) |
| Hydrants OOD (StyleGAN) | 29 | **+1.69** ±1.02 (69%) | +0.045 ±0.019 (79%) | −0.069 ±0.026 (79%) | −0.306 ±0.197 (69%) |
| Vases OOD (DiffAE) | 27 | **+0.76** ±0.66 (63%) | +0.076 ±0.045 (74%) | −0.041 ±0.022 (81%) | −0.155 ±0.118 (70%) |
| Hydrants ID (DiffAE) | 18 | −2.42 | −0.040 | +0.033 | +0.217 |
| Vases ID (DiffAE) | 27 | −2.58 | −0.037 | +0.059 | +0.445 |

All twelve OOD entries favour Infer3D with 95% CIs excluding zero. **The headline claim
holds.** The ID rows favour the feed-forward lifter, which the paper also reports.

Absolute full-set means vs published (`tools/make_tables.py`):

| Cell | Method | CD | PSNR | SSIM | LPIPS | n |
|---|---|---|---|---|---|---|
| Hyd OOD | Splatter | 0.907 [0.733] | 14.72 [15.73] | 0.665 [0.665] | 0.330 [0.266] | 38 |
| Hyd OOD | Infer3D DAE | 0.538 [0.461] | 16.81 [18.46] | 0.722 [0.714] | 0.246 [0.220] | 38 |
| Vas OOD | Splatter | 0.856 [0.775] | 15.18 [15.01] | 0.622 [0.661] | 0.333 [0.315] | 27 |
| Vas OOD | Infer3D DAE | 0.693 [0.552] | 16.11 [17.65] | 0.700 [0.678] | 0.291 [0.261] | 28 |
| Hyd ID | Infer3D DAE | 0.269 [0.102] | 20.58 [20.55] | 0.818 [0.761] | 0.146 [0.182] | 18 |
| Vas ID | Infer3D DAE | 0.535 [0.119] | 16.92 [19.65] | 0.718 [0.715] | 0.265 [0.215] | 27 |

The Vases ID row is the single largest discrepancy in Table 1. It was re-run from the
clean repo under both eval variants (see below): the eval reproduces its own cached
numbers exactly and neither variant approaches the published 19.65 / 0.119, so the
discrepancy originates in the optimization run behind those checkpoints.

The published Hydrants OOD gain is +2.73 dB; the honest one is +1.96 dB. For Vases,
+2.64 dB published vs +0.76 dB honest.

### The `regenerate_ours_splats` variant matters — resolved by direct A/B

Two versions of the routine that rebuilds optimized splats exist and disagree by 1-4 dB on
CO3D. `$INFER3D_OURS_REGEN` selects between them (`tools/eval/regenerate_ours.py`):

* `paper` — commit `e534f0e`. Rotates about the fixed point `[0,0,zgt]`; hands the lifter
  the OOD view's camera.
* `aligned` (**default**) — a later commit. Rotates about the empirical centroid of the
  predicted splats; hands the lifter view 0.

Before this audit the two eval entrypoints carried *separately drifted* copies of this
function; they now share one implementation.

**The A/B was run directly**: the full gs2mesh + ICP eval over the *same* cached Hydrants
OOD checkpoints under both variants, changing nothing else. Paired over the 38 objects
present in both:

| metric | `paper` | `aligned` | aligned − paper | 95% CI | aligned better on |
|---|---|---|---|---|---|
| PSNR | 16.447 | 17.519 | **+1.072** | [+0.29, +1.85] | 71% |
| SSIM | 0.720 | 0.734 | +0.014 | [+0.002, +0.027] | 63% |
| LPIPS | 0.249 | 0.229 | −0.020 | [−0.034, −0.006] | 68% |
| CD | 0.554 | 0.454 | −0.100 | [−0.173, −0.027] | 61% |

The same A/B was then run on an **ID** cell (Hydrants ID, same cached checkpoints,
n=18) to check whether the choice is cell-dependent. It is not:

| metric | `paper` | `aligned` | paper − aligned | 95% CI |
|---|---|---|---|---|
| PSNR | 20.575 | 20.594 | −0.019 | [−0.152, +0.114] |
| SSIM | 0.818 | 0.819 | −0.001 | [−0.002, +0.000] |
| LPIPS | 0.146 | 0.145 | +0.001 | [−0.001, +0.002] |
| CD | 0.269 | 0.281 | −0.012 | [−0.039, +0.015] |

**On ID the two variants are statistically indistinguishable** — every CI spans zero — and
both land on the published 20.55. That is what the geometry predicts: on ID the OOD
rotation is the identity, so the differing rotation centre cannot matter and the two code
paths coincide. (This corrects the claim in `indistribution-reproduction/README.md` that
the variants differ by 2-4 dB on ID; that comparison evidently included some other
difference, because a like-for-like A/B here shows no effect.)

A second ID cell (Vases ID, n=26) agrees — and if anything favours `aligned`:

| metric | `paper` | `aligned` | paper − aligned | 95% CI |
|---|---|---|---|---|
| PSNR | 16.508 | 17.024 | −0.516 | [−1.195, +0.163] |
| SSIM | 0.705 | 0.713 | −0.009 | [−0.021, +0.004] |
| LPIPS | 0.280 | 0.265 | +0.015 | [+0.001, +0.028] |
| CD | 0.578 | 0.505 | +0.073 | [−0.033, +0.180] |

**Result: `aligned` is the right choice everywhere** — it reproduces the published OOD
cells (CD 0.454 vs published 0.461) and is equal-or-better on both ID cells. It is the
default. There is no need to select a variant per cell.

Separately, this run pins down where the Vases ID discrepancy lives. Neither variant gets
close to the published Vases ID row (best is 17.02 PSNR / 0.505 CD against a published
19.65 / 0.119), and the fresh `aligned` run reproduces the cached CSV to three decimals
(CD 0.536 vs 0.535, PSNR 16.92 vs 16.92). So the eval code is deterministic and not at
fault: **the Vases ID gap is in the optimization run behind those checkpoints, not in the
evaluation.** For reference the frozen lifter scores 19.60 PSNR / 0.090 CD on the same
objects — close to the value published for Infer3D on this row.

Because the choice still moves the headline OOD effect, both are reported:

| Hydrants OOD, paired Infer3D − Splatter | PSNR | SSIM | LPIPS | CD | n |
|---|---|---|---|---|---|
| `aligned` (default, matches published OOD) | **+1.88** (71% win) | +0.049 | −0.075 | −0.274 | 41 |
| `paper` (conservative) | **+0.86** (58% win) | +0.036 | −0.056 | −0.180 | 38 |
| cached CSVs used for the table above | +1.96 | +0.054 | −0.080 | −0.366 | 36 |

The claim survives under either variant — Infer3D beats the frozen lifter on all four
metrics either way — but the effect is +1.9 dB under the convention behind the published
numbers and +0.9 dB under the other. The conservative number is stated here so the choice
is visible rather than buried.

## Fresh end-to-end run from the released code

The strongest check available: run the pipeline from scratch using only the commands in
the README, on an object never touched by the cached artifacts, and see whether the
headline claim appears.

Stage A (`tools/optimize_co3d_diffae.py`, 783 iterations, ~1200 s per object on one
H100) followed by Stage B (`tools/eval/run_eval_pipeline.py` → `PIPELINE COMPLETE`) on
the first twelve CO3D Hydrants OOD objects. Eleven survived meshing and were scored:

| metric | Infer3D | Splatter Image | paired Δ | 95% CI | win |
|---|---|---|---|---|---|
| PSNR ↑ | **16.826** | 13.555 | **+3.270** | [+1.159, +5.382] | 73% |
| SSIM ↑ | **0.693** | 0.604 | **+0.090** | [+0.043, +0.136] | 91% |
| LPIPS ↓ | **0.242** | 0.368 | **−0.126** | [−0.178, −0.074] | 91% |
| CD ↓ | **0.368** | 0.836 | **−0.468** | [−0.760, −0.176] | 82% |

**All four metrics favour Infer3D with 95% CIs excluding zero**, from nothing but the
released code, the frozen checkpoints and the bundled splits. The effect is larger here
(+3.27 dB) than in the cached-artifact analysis (+1.88 dB, n=41) — the two are consistent
given the CIs and the different object subsets. These per-object CSVs ship in
`assets/eval_csvs/fresh_run_hydrants_ood/`, so this table can be rechecked offline.

The run also independently confirms the published pruning schedule: the optimizer logged
`Applying configuration step 2 at 122 steps` and `step 3 at 302 steps`, matching
k_t: 600 → 32 (t=3) → 10 (t=122) → 5 (t=302).

### The oracle-selection gap, measured fresh

Extending the same fresh run to six objects gives an independent measurement of the
oracle-selection issue described above — this time on CO3D rather than ShapeNet-NMR:

| object | loss-selected (`known_best`) | oracle-selected (`best`) |
|---|---|---|
| 134_15451_31119 | 18.51 | 18.51 |
| 147_16374_32167 | 20.91 | 20.91 |
| 194_20925_42241 | **9.38** | **19.46** |
| 206_21797_45640 | 13.15 | 15.02 |
| 235_24641_51707 | 13.59 | 14.04 |
| 250_26767_54267 | 15.25 | 15.87 |
| **mean** | **15.13** | **17.30** |

Selecting the particle by ground-truth novel-view PSNR is worth **+2.17 dB** here, and
the two agree on only 2 of 6 objects. Object `194_20925_42241` is the extreme case: the
optimizer's own decision loss picked a particle 10 dB worse than the best available one,
which is precisely the failure mode the in-distribution analysis attributes to inversion
collapsing to a wrong mode.

This matters for how the tables are read. The CO3D Chamfer/novel-view pipeline
(`tools/eval/`) regenerates from the **loss-selected** hypothesis, so Table 1 is honest.
The ShapeNet-NMR CSVs report both columns and the paper quotes the oracle one — which is
why Table 2's published gain shrinks from +2.03 dB to +0.84 dB when the honest column is
used.

## Table 2 — ShapeNet-NMR

| Setting | Baseline | Infer3D (honest, `known_best`) | Infer3D (oracle, `best`) | published |
|---|---|---|---|---|
| SO(3) | 22.28 | 20.10 | 21.12 | 18.12 → 20.67 |
| SE(3) | 14.32 | 15.16 | 16.76 | 14.27 → 16.30 |

Paired PSNR, Infer3D minus baseline, same objects, full set:

| Setting | honest | oracle | published implies |
|---|---|---|---|
| SO(3) | **−2.17** ±0.44 (win 22%), n=310 | −1.15 ±0.40 (29%) | +2.55 |
| SE(3) | **+0.84** ±0.36 (win 56%), n=278 | +2.44 ±0.34 (82%) | +2.03 |

**SE(3) reproduces in direction** and is statistically significant, but the honest effect
is +0.84 dB, not +2.03 dB — roughly 1.6 dB of the published gain comes from oracle
particle selection.

**SO(3) does not reproduce.** Both `eval_baseline` and `eval_pred` score novel views in a
frame made relative to the OOD camera. Splatter Image emits camera-frame pixel-aligned
Gaussians, so under a *pure rotation* the OOD pose cancels and the baseline never has to
estimate it; Infer3D starts from a canonical generated image and must recover R, so it pays
a cost the baseline avoids. Under this (standard) protocol the baseline is only mildly
degraded — 24.27 ID → 22.28 OOD — and Infer3D is 2.17 dB behind it.

The published 18.12 corresponds to neither available protocol. The
`checkpoints-so3-baseline-probe` re-evaluation scores the baseline in *absolute* frame
(15.54 dB) while still scoring Infer3D in the relative frame; that is an apples-to-oranges
comparison, and it overlaps only 53 of the 310 SO(3) objects. The same CSV contains both
conventions on identical objects (`PSNR_novel_baseline` 15.54 vs
`PSNR_novel_baseline_relative` 21.75), which is direct evidence that the choice of frame,
not the method, drives that 6 dB.

A coherent honest statement of the result: a pure SO(3) camera rotation is absorbed by the
camera-frame output of the feed-forward lifter, so there is little for Infer3D to recover;
the benefit appears under SE(3), where translation cannot be absorbed.

**Chamfer cells are absent.** The NMR pipeline computes image metrics only; the paper's NMR
CD needs GT geometry that the SRN-rendered dataset does not ship.

### A note on the SO(3) runner itself

While purging hardcoded paths for release, `tools/optimize_nmr_so3.py` was found to point
its StyleGAN checkpoint at `…/stylegan3/~/training-runs/…` — a path containing a literal
`~` directory that does not exist. As shipped, the SO(3) runner could not have produced
any of the numbers above; the cached SO(3) CSVs came from the original research repo. It
now resolves the checkpoint through `config.STYLEGAN_CKPTS`. This does not change the
analysis (which used the cached per-object CSVs), but it does mean the SO(3) column had
never been re-run end-to-end from this package before.

## Table 3 — RE10K sensor / FOV shift

Full 160 sequences, no selection (`tools/re10k/make_table3.py`):

| Setting | Method | PSNR | SSIM | LPIPS | published PSNR |
|---|---|---|---|---|---|
| In-dist. | CATSplat | 22.98 ±0.81 | 0.761 | 0.182 | 25.41 |
| OOD Fisheye | CATSplat direct | 15.87 ±0.39 | 0.450 | 0.355 | 16.25 |
| OOD Fisheye | Equidistant (given TRUE FOV) | 19.18 ±0.48 | 0.608 | 0.266 | 19.24 |
| OOD Fisheye | **Infer3D (DAE), fully blind** | **22.17 ±0.68** | **0.736** | **0.218** | 23.33 |
| OOD Fisheye | oracle undistort (ceiling) | 22.67 ±0.74 | 0.752 | 0.212 | — |

Paired: Infer3D − CATSplat direct = **+6.30 dB** (CI +5.80…+6.80, **96%** win);
− Equidistant = **+2.99 dB** (87% win); − oracle = −0.50 dB, i.e. blind Infer3D lands
within half a dB of the oracle-undistortion ceiling. **This is the strongest and cleanest
result in the paper.**

Two caveats to disclose: (1) the OOD operating point (in_fov 75 / out_fov 94 / k 0.5 /
circle_scale 1.22) was *calibrated so the baseline rows match the published ones*, because
the paper's own fisheye-synthesis code was unavailable — so the OOD condition is a
reconstruction of the paper's, not the paper's; (2) the in-distribution row is 2.4 dB below
published, so the ID protocol differs somewhat.

## Table 4 — Adaptive inference

| Split | Splatter (ff) | Opt. only | Adaptive | oracle routing | Avg time |
|---|---|---|---|---|---|
| 99/1 | 22.41 [22.41] | 20.52 [20.53] | 21.94 [22.45] | **22.45** | 2.1 min [2.1 s] |
| 90/10 | 21.58 [21.58] | 20.24 [20.34] | 21.52 [22.07] | **22.00** | 3.2 min [19 s] |
| 70/30 | 19.76 [19.76] | 19.63 [19.92] | 20.60 [21.23] | **20.99** | 5.6 min [55 s] |
| 50/50 | 17.93 [17.93] | 19.02 [19.50] | 19.68 [20.41] | **19.99** | 7.9 min [1.4 m] |

The feed-forward column reproduces exactly on all four rows. Three problems:

1. **The published "Adaptive" column tracks oracle routing, not the calibrated router** —
   at 99/1 the published 22.45 equals the oracle column exactly. The published adaptive
   numbers appear to assume perfect OOD detection.
2. **The F1 column does not reproduce**: published 0.943/0.955/0.971/0.986 (rising with the
   OOD fraction) vs 0.171/0.689/0.888/0.943 measured. At a 1% OOD rate a 9.5% false-positive
   rate collapses F1 to 0.17; 0.943 is not attainable there with this detector.
3. **Runtimes are far faster in the paper** (19 s vs 3.2 min amortized at 90/10). Plausibly
   a different configuration — the paper cites B=10 parallel particles and ~2.8 min to
   convergence, whereas these runs imply ~16 min per OOD sample. Not resolvable from cached
   artifacts.

## Table 5 — RealCars Chamfer

Full n=20, no selection (`tools/realcars/build_table_4col.py`):

| Method | Mean CD [paper] | Median CD [paper] |
|---|---|---|
| **Infer3D** | **0.1309 [0.131]** | **0.0934 [0.093]** |
| Splatter Image | 0.2191 [0.219] | 0.1791 [0.179] |
| Splatter + DINOv2 + depth | 0.1517 [0.152] | 0.1341 [0.134] |
| LGM | 0.4910 [0.405] | 0.3023 [0.307] |
| SF3D | 0.2676 [0.242] | 0.2001 [0.206] |

Paired CD, Infer3D minus baseline: −0.088 vs Splatter (80% win), −0.360 vs LGM (75%),
−0.137 vs SF3D (90%); all significant. The three headline rows match the paper to three
decimals. The LGM and SF3D *means* do not (their medians do) — the table committed in the
research repo was built from an older `chamfer_lgm_sf3d.csv` than the one on disk. The
claim is unaffected.

## Tables 6 & 7 — OOD detection and calibration

| Cell | reproduced | published |
|---|---|---|
| Hydrants AUROC (step-0) | **0.971** (n=43+43) | 0.945 |
| Vases AUROC (combined loss) | **0.939** (n=28+28) | 0.961 |
| Vases AUROC (LPIPS only) | 0.999 | — |
| Detection latency | **283 ms** | "roughly 286 ms" |
| Calibration, 5 ID + 5 OOD | 0.6 s batched | "maximum of 500 ms" |
| Calibration F1 @ 10 objects | **0.965** | 0.965 |
| Calibration F1 @ 20 objects | **0.944** | 0.944 |

The Vases cell was computed fresh from this repo (`tools/ood_detection/`, ported during
this audit). Note the paper's "Num Val Objects" counts ID+OOD objects while the script's
`N_cal` counts pairs; matched that way the F1 values agree exactly.

Worth noting for the paper: **MSE alone is nearly uninformative as a detection score**
(Vases AUROC 0.53); the signal is essentially all LPIPS. Section 3.5 credits "MSE and
LPIPS" equally.

Not reproducible from available artifacts: the RE10K AUROC cell (0.997) and the entire
"Image alignment" baseline row (0.554 / 0.621 / 0.748), which is not implemented anywhere
in either repository.

## Table 8 — Loss ablation

**The ablation ladder could not be run as shipped.** The optimizer runs a six-stage
schedule that writes loss weights directly into module globals, and the stage at step 302
hardcoded `lambda_lpips: 2`, overriding whatever was configured — so "L_MSE only" was not
expressible, and setting `lambda_lpips=0` crashed with `UnboundLocalError: lpips_fn`
because that object was only built when the configured weight was non-zero. Both are fixed
here (default behaviour unchanged). The consequence for the paper is that the *effective*
loss weights are the schedule's, not the config's, so the rows are not simply "add one term".

A **4-object paired pilot** was then run — all four rungs on the identical objects. This is
a small sample and the metric is the optimizer's own novel-view PSNR/LPIPS, not the
ICP-aligned metrics the published table reports, so it is indicative only:

| config | PSNR (loss-selected) | PSNR (oracle) | LPIPS (loss-selected) | published PSNR / LPIPS |
|---|---|---|---|---|
| `L_MSE` | 14.27 | 15.08 | 0.349 | 14.85 / 0.482 |
| `+L_LPIPS` | 12.64 | 15.98 | 0.364 | 16.20 / 0.315 |
| `+L_prior` | **17.43** | 18.20 | 0.223 | 17.45 / 0.255 |
| `+L_depth` (full) | 12.96 | 17.46 | 0.349 | 18.51 / 0.218 |

Two things to note, both tentative at n=4. First, the published ladder is monotone, and the
**oracle** column here is much closer to monotone (15.08 → 15.98 → 18.20 → 17.46) than the
loss-selected column (14.27 → 12.64 → 17.43 → 12.96), which is consistent with the
oracle-selection issue that definitely affects Table 2. Second, the `+L_prior` rung
reproduces almost exactly (17.43 vs 17.45) while the two rungs on either side do not, and
adding the depth term *hurt* on these four objects.

A trustworthy Table 8 needs the full test set and the ICP metrics per configuration
(roughly 4 × 38 × 55 min of GPU plus four eval passes); that was out of budget here.
Also worth checking separately: the cached `-NO-depth` and `-NO-Z-loss` runs give
12.47 and 10.01 PSNR against 12.76 for the full objective on their 4 common objects,
i.e. the *ordering* there is consistent with the paper even though the pilot above is not.

## Table 9 — Trivial test-time training

The published row compares across protocols: the TTT number 12.41 is the non-ICP
`ours_psnr`, while the 15.73 it is set against is the ICP-aligned Splatter number from a
*different* 38-object run. On the same 19 objects:

| protocol | Splatter | TTT | paired TTT − Splatter |
|---|---|---|---|
| non-ICP | 13.31 | 12.41 | **−0.90** (CI −1.45…−0.35) |
| ICP | 13.95 | 13.55 | −0.40 (CI −1.42…+0.63, n.s.) |

**The qualitative claim holds** — naive test-time training does not improve on the frozen
baseline — but the published −3.32 dB gap is an artifact of the mismatched comparison; the
honest gap is −0.90 dB, or −0.40 dB and not significant after ICP alignment.

---

## Pipeline

1. **Stage A — test-time optimization** (`tools/optimize_*`): optimize the generator latent
   z and pose θ with the frozen lifter constraining the 3D. Writes per-scene
   `topk_best_everything_latest.pth` plus an aggregate CSV. ~55 min/object on an H100.
2. **Stage B — gs2mesh + Sim(3)-ICP eval** (`tools/eval`, CO3D only): splats → PLY → point
   cloud → ICP-align pred↔pseudo-GT → render novel views → per-object Chamfer + PSNR/SSIM/LPIPS.
3. **Tables**: `tools/make_tables.py` (1 & 2), `tools/re10k/make_table3.py` (3),
   `tools/ood_detection/` (4, 6, 7), `tools/realcars/build_table_4col.py` (5).

## Known gaps

* ShapeNet-NMR Chamfer (all Table 2 CD cells) — needs GT geometry the SRN-rendered dataset
  does not include.
* Table 6 RE10K cell and the "Image alignment" baseline row.
* Table 8 full ablation.
* Table 4's runtime column at the paper's B=10 parallel configuration.
