# Rebuttal prose — ID trade-off (AC priority 2) & optimization dynamics/cost (AC priority 3)

Companion to `../REBUTTAL_RESPONSE.md` (AC priority 1, the optimization-based baselines).
Grounded in `report.html` (qualitative evidence) + `FINDINGS.md` / `DYNAMICS.md`. Ready to
paste; trim to fit.

---

## To the AC — priority 2: the in-distribution trade-off (uH4P vs XsZX)

We thank the AC for flagging the disagreement. **XsZX is correct that an in-distribution
(ID) gap exists; "maintained" overstated it.** Reproducing the paper's ID cells and
analyzing them per-object (same metric/render code):

| prior | Infer3D | Splatter | median gap | mean gap | objects &gt;4 dB |
|---|---|---|---|---|---|
| DiffAE (encoder) | 20.6 | 23.0 | 1.7 dB | 2.6 dB | 4 / 18 |
| StyleGAN (latent-only) | 17.8 | 23.0 | 5.2 dB | 5.5 dB | 14 / 18 |

**The gap is not a uniform quality loss — it is concentrated in a minority of objects where
inverting the lightweight 2D prior lands in a wrong mode and the reconstruction collapses**
(a blurry blob), while on the remaining objects Infer3D matches the feed-forward lifter to
~1–2 dB (median DiffAE 1.7 dB). We verified this by inspection: error maps, per-object
breakdown, and image-content spectra (not just averaged numbers) — see the qualitative
report. The encoder-initialized DiffAE makes collapse rare (4/18) vs latent-only StyleGAN
(14/18), so the cost is a property of the 2D prior's inversion fidelity and shrinks with a
better prior. We also ruled out registration/shading (a per-image shift + gain/bias search
closes only ~0.2 dB) and uniform blur (on successful objects ours ≈ Splatter sharpness).

Crucially, the failure objects are exactly the **high-reconstruction-loss cases the OOD
detector flags** (Sec 3.5 / Table 6, AUROC 0.97 from the initial loss). The adaptive router
(Table 4) sends them through the feed-forward lifter, so the ID penalty is neutralized in
deployment. Geometry is also comparatively preserved (ID Chamfer gap small; OOD Chamfer
beats the feed-forward lifter).

## To the AC — priority 3, and Reviewer akZb: optimization dynamics

From logged runs (n=43 ID / 43 OOD; figure in the report), no new compute:
- **Convergence, ID vs OOD.** At feed-forward init the mean reconstruction loss is 0.030
  (ID) vs 0.132 (OOD) — a 4.3× separation; OOD descends monotonically (0.132→0.062 over 300
  steps), ID begins near-converged. Mean per-object monotonicity 99.6%.
- **Step count / schedule.** 783 iterations over 6 stages (multi-start rotation+latent
  search → prune to top-k → refine). ~24 min/sample at the default particle budget.
- **Cost & memory (uH4P).** Feed-forward 0.43 s; peak VRAM 37.5 GB (48–59 GB for larger
  search); runtime/quality is a smooth knob (18.6→2.7 min trades 21.5→15.3 dB).
- **Amortized (Table 4).** Because ID/failures are detectable from the initial loss at
  feed-forward cost (AUROC 0.97), the router runs optimization only on flagged inputs:
  191 s/sample at a 10%-OOD stream, matching/beating both pure strategies.

## To Reviewer uH4P
- **ID accuracy ("maintained").** There is a measurable ID gap, concentrated in inversion-
  failure objects (4/18 DiffAE), median ~1.7 dB otherwise; neutralized by adaptive routing.
- **Runtime / memory.** 0.43 s feed-forward; 37.5 GB peak; 1.8–24 min per operating point;
  amortized 191 s/sample at 10% OOD.

## Tightest version
> The in-distribution gap is not uniform: it is concentrated in a minority of objects where
> inverting the lightweight 2D prior collapses to a wrong mode (4/18 for encoder-DiffAE, more
> for StyleGAN); elsewhere Infer3D matches the feed-forward lifter to ~1–2 dB. Those failures
> are the high-loss cases the OOD detector flags (AUROC 0.97, 0.43 s) and the router sends to
> the feed-forward lifter, so the penalty is neutralized in deployment. Dynamics: ID init loss
> 4.3× below OOD; 783 iters/6 stages, 37.5 GB peak, amortized 191 s/sample at 10% OOD.

---
### Correction log (internal)
Two earlier drafts were wrong and were fixed by qualitative sanity-checking:
(1) "gap is lost fine detail" — too strong; (2) "gap is purely low-frequency coarse identity,
high-freq matched" — an artifact of the silhouette-dominated *error* power spectrum. The
per-object + image-content-spectrum + error-map analysis gives the failure-driven story above.
Lesson applied: trust the pixels, not one averaged statistic.
