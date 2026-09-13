# Rebuttal prose — OOD objects, OOD scope, and dataset-specificity (Reviewer akZb Q2 + weaknesses)

New experiment in this rebuttal: `detector_ood_objects.py` →
`rebuttal/results/ood_objects/{scores.json, montage_hydrants_id.png, montage_vases_oodobj.png}`
(n=30 per group, one encode–decode pass per object, everything frozen — the paper's
Sec-3.5 detector path, ~0.3 s/input).

---

## To Reviewer akZb — Q2: "How would the method work for OOD objects?"

The reviewer's premise is correct, and we state it plainly: **Infer3D cannot reconstruct
object categories the image prior has never seen** — the optimization searches the prior's
manifold, so an unseen category is outside the hypothesis space (we will make this
explicit in the Limitations). The right question for a deployed system is therefore
whether the method **knows** when this happens — and this is where our design gives an
answer feed-forward methods structurally cannot.

**New experiment (OOD-object detection).** We feed CO3D **vases** (test split, ordinary
in-distribution camera poses — only the *object category* is OOD) through the **hydrants**
model (DiffAE encoder → decoder → frozen lifter → render), and score each input with the
paper's OOD-detection signal: the initial reconstruction error (MSE + LPIPS) of the
one-pass hypothesis against the input.

| group (n=30 each) | initial recon error (MSE+LPIPS) |
|---|---|
| ID hydrants | **0.051** (median 0.041) |
| vases through hydrants model | **0.242** (median 0.234) — 4.7× |

**AUROC = 0.99** for detecting OOD *objects* — the same detector, at feed-forward cost,
with no retraining and no OOD data used for calibration. Qualitatively the failure mode is
exactly the manifold-snapping one expects: the hydrant prior "sees" a hydrant in every
vase (a white vase is decoded as a white hydrant with a dome and nozzle; a crystal vase as
a grey hydrant — montage in the supplement), and the large residual against the actual
input is what the detector picks up. So on OOD objects Infer3D degrades **detectably**:
the adaptive router (Sec. 3.5, Table 4) flags the input and can fall back to the
feed-forward lifter or abstain — whereas a feed-forward network silently produces
confident wrong geometry with no signal that anything went wrong. We believe
fail-loudly-vs-fail-silently is the practically important distinction under category
shift, and we thank the reviewer for prompting us to quantify it.

Finally, the hypothesis space grows directly with the prior: our framework is
prior-agnostic, and swapping in a broader generative prior (e.g., a large text-to-image
diffusion model) extends category coverage without changing the method — at higher
inference cost (Limitations already note this trade-off).

## To Reviewer akZb — "dataset-specific; no single model spanning datasets"

Two clarifications:

1. **Table 2 already uses a single prior spanning 11 categories.** The ShapeNet
   experiments use one class-conditional StyleGAN over all 11 NMR categories with a single
   multi-category lifter — not 11 separate models. The per-category priors on CO3D reflect
   the available pretrained assets (DiffAE/StyleGAN checkpoints per CO3D category), not an
   architectural constraint of the framework.
2. **This is also the operating regime of the entire optimization-based family the review
   compares us to**: pi-GAN, EG3D(+PTI inversion), nerf-from-image, and FINV's GET3D are
   all trained per category. Within that family, Infer3D is on the *more* general end
   (one 11-category prior in Table 2), and the framework itself is prior-, lifter-, and
   representation-agnostic (as the review's Significance section acknowledges) — a single
   broad-coverage image prior slots in without modification.

## To Reviewer akZb — "OOD scope is very limited"

We evaluate three OOD axes (pose/SE(3), sensor/fisheye-intrinsics, appearance/sim-to-real)
plus, in this rebuttal, category shift (above) and object-count shift (see the
compositionality response). We agree the scope is bounded by the prior's support and will
sharpen the Limitations paragraph accordingly: Infer3D robustifies reconstruction under
*rendering- and appearance-level* shifts of known categories, detects (rather than solves)
*content-level* shifts, and inherits the prior's coverage.

## Tightest version

> Infer3D cannot reconstruct categories its prior has never seen — no inversion method
> can — but it *detects* them: feeding CO3D vases through the hydrants model, the paper's
> initial-loss detector separates OOD objects from ID inputs at AUROC 0.99 (0.051 vs
> 0.242 mean error, n=30+30, feed-forward cost), so the router flags them instead of
> silently hallucinating as feed-forward networks do. On dataset-specificity: Table 2
> already uses one class-conditional prior spanning 11 ShapeNet categories, and every
> method in the cited inversion family (pi-GAN, EG3D, NFI, FINV/GET3D) is per-category;
> the framework is prior-agnostic, so broader priors extend coverage without changing the
> method.
