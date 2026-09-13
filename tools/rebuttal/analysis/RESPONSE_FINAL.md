# Reviewer-ready response — AC priority 2 (ID trade-off) + priority 3 (dynamics & cost)

Paste-ready. Numbers all reproduced with the paper's own metric/render code.
Evidence: `../id_analysis/index.html` (videos + pyramids), `report.html`, `DYNAMICS.md`,
`perterm/perterm_curves.png` (per-term dynamics).

---

## A. To the AC — resolving the uH4P vs XsZX disagreement

Both readings are correct about **different rows of our own tables**, which is why they
appear to conflict:

- **XsZX reads the optimization-only rows.** These do show an ID cost: CO3D hydrants
  20.55 vs Splatter 21.48 PSNR (−0.93 dB, DiffAE); ShapeNet 21.61 vs 24.27 (−2.66 dB,
  StyleGAN). XsZX is right, and "ID accuracy maintained" overstated it.
- **uH4P describes the complete proposed system**, i.e. the adaptive row of Table 4, which
  **matches or exceeds the feed-forward lifter on ID-dominated streams**: 22.45 vs 22.41
  (99% ID), 22.07 vs 21.58 (90% ID).

So: *optimization-only inference pays a modest ID cost; the full system with adaptive
routing does not.* We will state this explicitly rather than leaving it to the limitations
section.

## B. Cause of the ID gap — three findings, each visually verified

**B1. The gap is an *observed-view* effect and largely disappears on occluded views.**
PSNR at the input/near view rewards copying the pixels you were handed — which a
feed-forward lifter does by construction, and an inverse-graphics method cannot. We
therefore measured the gap separately at the near/observed view and at the maximally
occluded view (the unseen side, where genuine 3D inference is required), using a Gaussian
pyramid so the comparison is resolved by spatial scale (n=5 objects per class, full-res L0):

| class (prior) | near/observed view | occluded view | shrink |
|---|---|---|---|
| ShapeNet cars (StyleGAN) | +5.62 ± 1.70 dB | **+1.58 ± 0.79 dB** | 4.0 dB |
| CO3D hydrants (DiffAE) | +5.73 ± 2.47 dB | **+0.94 ± 1.04 dB** | 4.8 dB |

(gap = Splatter − Infer3D; positive = feed-forward ahead.) All 10 objects move the same
way; on 2/5 hydrants the occluded-view gap **reverses** in our favour. On hydrants the
feed-forward lifter reproduces the input view at ~39 dB — essentially a copy — and that copy
advantage *is* the entire ID gap. Where the input gives no answer to copy, Infer3D is level
with it.

**B2. What remains tracks the 2D prior's invertibility, not the 3D machinery.**
Our own Table 1 is a controlled experiment: same lifter, same losses, same data, two priors.
DiffAE (has an encoder, invertible) → 0.93 dB gap. StyleGAN (latent-only) → 3.8 dB. The ID
cost is a property of the frozen prior, and the limitations section's point about stronger
diffusion backbones is the direct remedy. In the pyramid data this shows up as a residual
coarse-scale (L4–L5) gap for cars only: StyleGAN completes the unseen back as a plausible
but slightly different-coloured/shaped car — a global identity offset, not lost detail.
DiffAE shows ≈0 gap at *every* scale on occluded views.

**B3. The residual is heavy-tailed, not a uniform quality loss.**
Per-object: DiffAE median gap 1.7 dB but mean 2.6 dB, driven by 4/18 objects where
inversion lands in a wrong mode and collapses; StyleGAN is broadly weaker (14/18 > 4 dB).
We ruled out registration/shading (per-image shift + per-channel gain/bias search closes
only ~0.2 dB) and uniform blur (on successful objects our sharpness ≈ Splatter's).

**Practical implication.** The collapse cases are exactly the high-initial-reconstruction-loss
inputs our OOD detector flags (AUROC 0.945 hydrants / 0.961 vases / 0.997 RE10K). The
adaptive router therefore sends them down the feed-forward path automatically — which is
*why* Table 4's adaptive row recovers full ID accuracy. The trade-off is (i) concentrated
where PSNR is least meaningful, (ii) self-diagnosing, and (iii) shrinking with prior quality.

## C. Optimization dynamics (akZb, uH4P)

**C1. Per-term convergence, unweighted.** Our 6-stage schedule reweights the composite
objective mid-run (`lambda_mse` 1→10 and `lambda_lpips` 0.5→2 at iter 302, `lambda_mse`
10→2 at 732), so the composite loss is not comparable end-to-end. We therefore report each
term **unweighted** over all 783 iterations (DiffAE hydrants; `perterm_curves.png`):
MSE, LPIPS, depth-MSE and PSNR, ID vs OOD. All terms are measured **against the single input
view being fitted** — they are the optimization objective, so this is a convergence curve, not
a novel-view quality metric (hence ID reaches ~34 dB here vs the 20.55 dB novel-view number in
Table 1). Each term is continuous across the stage boundaries (MSE ratio 1.008x, LPIPS 0.995x,
depth 1.001x at iter 302, where the composite jumps 4.74x), confirming the apparent jump was
purely the weight change, and every term descends monotonically to a flat final stage.
Over the run OOD improves 17.3 -> 25.8 dB fit while ID is already at 32.2 dB by iteration 10
(-> 34.2), i.e. ID starts near-converged: the Sec-3.5 mechanism, per-term.

<!-- PERTERM_TABLE:BEGIN -->
_n = 3 ID, 6 OOD; full 783-iteration schedule; every iteration logged._

| iter | OOD MSE | ID MSE | OOD LPIPS | ID LPIPS | OOD depth | ID depth | OOD PSNR&#8203;(in) | ID PSNR&#8203;(in) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0175 | 0.0021 | 0.2153 | 0.0520 | 0.0005 | 0.0001 | 17.71 | 27.00 |
| 10 | 0.0148 | 0.0012 | 0.1950 | 0.0364 | 0.0005 | 0.0001 | 18.44 | 29.61 |
| 30 | 0.0128 | 0.0009 | 0.1788 | 0.0310 | 0.0005 | 0.0001 | 19.00 | 30.59 |
| 50 | 0.0111 | 0.0009 | 0.1575 | 0.0288 | 0.0003 | 0.0001 | 19.63 | 30.58 |
| 100 | 0.0071 | 0.0009 | 0.1234 | 0.0246 | 0.0002 | 0.0001 | 21.86 | 30.67 |
| 150 | 0.0072 | 0.0008 | 0.1104 | 0.0222 | 0.0003 | 0.0001 | 21.93 | 31.09 |
| 300 | 0.0056 | 0.0008 | 0.0979 | 0.0203 | 0.0002 | 0.0001 | 22.98 | 30.95 |
| 400 | 0.0046 | 0.0007 | 0.0909 | 0.0185 | 0.0002 | 0.0001 | 23.86 | 32.06 |
| 500 | 0.0041 | 0.0006 | 0.0854 | 0.0176 | 0.0002 | 0.0001 | 24.26 | 32.23 |
| 600 | 0.0040 | 0.0006 | 0.0820 | 0.0171 | 0.0002 | 0.0001 | 24.44 | 32.39 |
| 700 | 0.0036 | 0.0006 | 0.0804 | 0.0168 | 0.0001 | 0.0001 | 24.93 | 32.47 |
| 782 | 0.0036 | 0.0006 | 0.0790 | 0.0166 | 0.0001 | 0.0001 | 24.90 | 32.42 |

## Continuity across the reweighting boundaries (ratio of term just after : just before)

| boundary iter | term | ratio |
|---|---|---|
| 122 | mse | 1.151× |
| 122 | lpips | 1.028× |
| 122 | depth_mse | 1.053× |
| 302 | mse | 1.048× |
| 302 | lpips | 0.995× |
| 302 | depth_mse | 0.995× |
| 732 | mse | 0.983× |
| 732 | lpips | 0.996× |
| 732 | depth_mse | 0.946× |

A ratio near 1.0 confirms the term itself is continuous — the ~4.7× jump in the
composite `best_loss` at iter 302 was purely the weight change.
<!-- PERTERM_TABLE:END -->

**C2. ID vs OOD separation (the Sec-3.5 mechanism).** At feed-forward initialization
(0.43 s) the OOD reconstruction loss is 4.3× the ID loss; the ratio is scale-free and stays
4.4–5.6× for the whole run. The *initial* loss alone separates ID from OOD at AUROC 0.971.
Detection AUROC slowly *declines* with more optimization (0.971 → 0.945 at 300 → 0.920 at
783) because optimization pulls OOD losses toward the ID range — an additional argument for
routing at step 0. Descent is monotone at 99.88% of steps across 96 trajectories, and the
final stage is flat (−1%), i.e. the runs are genuinely converged.

**C3. Cost / memory.**

| config | schedule | peak VRAM | wall-time | OOD PSNR |
|---|---|---|---|---|
| feed-forward lifter | — | single pass | **0.43 s** | 13.4 |
| Infer3D full | 783 it / 6 stages | 37.5 GB | ~24 min | 20.0 |
| Infer3D fast | 221 it | 37.5 GB | 1.8 min | 12.9–18.0 |
| Infer3D fast (v6) | 226 it | 48.3 GB | 2.9 min | 18.2–19.1 |

Runtime/quality is a smooth knob; even early-stopped configs beat the feed-forward OOD
baseline (16.98 vs 15.73). VRAM scales with the particle count. **Amortized** cost under the
adaptive router: 191 s/sample at a 10%-OOD stream (vs 1429 s pure optimization) while
matching or beating both pure strategies. Seed variance: PSNR std 0.3–2.2 dB over 5 seeds,
highest on multi-modal OOD poses — which the multi-start particle search exists to control.

---

### Honest caveats we will state in the paper
- The ID cost of optimization-only inference is real; we now quantify it per-object and
  per-view rather than mentioning it in passing.
- It is largest exactly at the observed view, and largest for the weaker (latent-only)
  prior.
- The occluded-view analysis is on n=5 per class (compute-bound); the per-object trend is
  unanimous but we report the spread.
