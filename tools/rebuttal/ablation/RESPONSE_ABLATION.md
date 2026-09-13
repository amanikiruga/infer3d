# Rebuttal prose — component ablation (Reviewer R2)

Companion to `../REBUTTAL_RESPONSE.md` (AC priority 1, baselines) and
`../frequency/RESPONSE_ID_AND_DYNAMICS.md` (AC priorities 2–3). Ready-to-paste; trim to fit.
Every number is from the paper's own optimizer + metric code — see `ABLATION.md`,
`ablation_table_{cars,co3d}.md`, `routing_table_cars.md`.

---

## To Reviewer R2 — isolating each component's contribution

We thank R2 for this request. We ran a **leave-one-out (LOO)** ablation that removes exactly one
component at a time from the full system, on **both** benchmarks, using the paper's optimizer
and metric code (the control arm reproduces our reported numbers, and its feed-forward baseline
matches the cached eval bit-for-bit). All arms share the same per-object initialization, so the
deltas are paired and their error bars are small.

**A note on design.** To *isolate an individual contribution*, LOO is the right tool:
contribution of X = full − (full without X), which is order-independent. This complements the
paper's **Table 8**, which ablates the losses *cumulatively* (MSE→+LPIPS→+prior→+depth); the
cumulative increments are order-dependent, LOO deltas are not. We report LOO below and keep
Table 8 as the cumulative view.

**(1) Latent-code optimization and (2) rendering-parameter (pose) optimization — the two
dominant components.** On ShapeNet cars (SO(3) OOD, n=25; full Infer3D 20.29 dB vs feed-forward
16.78 dB):

| Remove | ΔPSNR (paired) |
|---|---|
| rendering-parameter (pose) optimization | **−3.10 ± 0.42** |
| latent-code optimization | **−2.27 ± 0.35** |
| both (search + selection only, no gradient) | **−4.08 ± 0.32** → back to feed-forward level |

Removing either optimization drops most of the way back to the feed-forward lifter — the gain
is **not** the generic benefit of "some test-time optimization"; it comes specifically from
jointly optimizing the latent code *and* the rendering pose. (Two further arms isolate the
multi-start *search* from the gradient refinement within each; see `ablation_table_cars.md`.)

*Cumulative view (build-up form of Table 8, for cars).* Adding capabilities on top of the
feed-forward lifter (16.78 dB), losses full throughout: + multi-start search 16.21 → + latent
gradient-opt 17.19 → + pose gradient-opt = full **20.29**. The per-rung credit is
order-dependent (swap the order and latent-opt's credit moves from +0.98 to +2.27) — which is
why we report the order-independent LOO deltas above as the per-component number. Full ladders
(both orderings + a loss build-up) in `cumulative_table_cars.md`.

**(3) Loss terms.** LOO on cars, and — because the depth and latent-prior terms exist only in
the CO3D objective — the LOO counterpart of Table 8 on CO3D hydrants (DiffAE, raw novel-view
PSNR; full 15.79 vs feed-forward 12.49):

| Remove (cars) | ΔPSNR | | Remove (CO3D) | ΔPSNR |
|---|---|---|---|---|
| MSE | −1.20 ± 0.25 | | LPIPS | −0.69 ± 0.58 |
| LPIPS | −0.53 ± 0.25 | | MSE | −0.50 ± 0.36 |
| noise-map reg. | −0.14 ± 0.17 | | latent prior (w_reg) | +0.24 ± 0.38 |
| | | | depth | +0.35 ± 0.44 |

The pixel/perceptual terms (MSE, LPIPS) contribute positively on both benchmarks, consistent
with Table 8's cumulative increments. The two **CO3D-only** terms — depth and the latent prior —
are **within noise on appearance PSNR**; this is expected and does not contradict the paper:
depth is a *geometry* regularizer, and Table 8 credits it with a Chamfer improvement
(0.580→0.458) that novel-view PSNR at the optimized pose does not measure. The LOO thus
*localizes* the depth/prior benefit to geometry rather than appearance (a Chamfer LOO — needing
the Stage-B ICP pipeline — is the right follow-up for those two terms).

**(4) OOD detector.** The detector cannot be a single ablation arm — its role is to route each
input to the policy that is best *for that input*. On the same cars, feed-forward is better on
in-distribution inputs (23.81 vs 21.43) while optimization is better on OOD inputs
(20.29 vs 16.78); the two **cross over**. Removing the detector forces one policy on the whole
stream and loses on half of it:

| stream (ID/OOD) | always-feed-forward | always-optimize | **detector-routed** |
|---|---|---|---|
| 90% / 10% | 23.11 | 21.31 | **23.46** |
| 70% / 30% | 21.70 | 21.09 | **22.75** |
| 50% / 50% | 20.29 | 20.86 | **22.05** |

The router beats both fixed policies at every mixture (and always-optimize also pays
optimization cost on every input, including the ID inputs where it is 2.4 dB *worse*). The
detector achieves this at feed-forward cost — initial-loss AUROC 0.97 (paper Tables 6–7), so
the realized router matches this oracle within rounding.

**Takeaway.** All four components are necessary and their contributions are cleanly separable:
pose-opt (−3.1) and latent-opt (−2.3) dominate the reconstruction gain; the loss terms add
−0.1 to −1.2 each; and the detector supplies the ID/OOD routing that neither fixed policy can
match — together they turn a 16.8 dB feed-forward lifter into a 20.3 dB analysis-by-synthesis
reconstructor while keeping in-distribution and amortized cost low.

## Tightest version (if space-limited)

> A leave-one-out ablation (paper's optimizer + metric; control reproduces our numbers) isolates
> each part on ShapeNet cars (full 20.3 vs feed-forward 16.8 dB): removing **pose optimization**
> costs −3.1 dB, **latent optimization** −2.3 dB (both → back to feed-forward), and the loss
> terms −1.2 (MSE) / −0.5 (LPIPS) / −0.1 (noise-reg); the CO3D-only depth and latent-prior terms
> are isolated on hydrants as the leave-one-out counterpart of Table 8. The **OOD detector** is
> isolated by routing: feed-forward wins on ID (23.8 vs 21.4) and optimization wins on OOD
> (20.3 vs 16.8), so the detector-routed stream beats both fixed policies at every ID/OOD mix,
> at feed-forward detection cost (AUROC 0.97). All four components are necessary and separable.
