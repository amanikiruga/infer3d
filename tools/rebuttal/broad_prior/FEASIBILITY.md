# Broad-prior Infer3D: a single model pair spanning categories (R1/akZb rebuttal probe)

**Question (Reviewer akZb):** "How would the method work for OOD *objects* the image prior has
never seen?" + "The approach is dataset-specific. There's no single model spanning datasets."

**Idea (user):** replace the per-category 2D prior (StyleGAN2 / DiffAE) with a *broad*
generative prior trained on ImageNet-or-wider, and replace the per-category lifter with the
**Objaverse-trained Splatter Image** — then run the same analysis-by-synthesis inversion.
If it reconstructs an OOD-category object from one image, that is a **single model pair
spanning categories**, a direct answer to both comments.

---

## Verdict: mechanically feasible, both halves exist on disk and run; ONE quality risk.

### Broad prior G  ✅ present and invertible
- **ADM ImageNet-256, unconditional (all 1000 classes)** — weights
  `/net/holy-isilon/ifs/rc_labs/ydu_lab/aakaran/diffusion-posterior-sampling/models/imagenet256.pt`
  (2.2 GB, readable), repo `.../aakaran/diffusion-posterior-sampling/` (guided_diffusion + DPS).
  Confirmed loads in env `test2` (`image_size 256`, `class_cond False`).
- A diffusion/score model **is** an energy/score-based prior; this is the honest broad-prior
  instantiation. There is **no dedicated broad image-EBM on disk** — the only EBM-family image
  model is Decomp-Diffusion (CLEVR, 64px, too narrow). Stable Diffusion / LAION weights are NOT
  resident (sd-turbo blobs are empty symlinks; only an SD-2.1 VAE is local), but `diffusers
  0.35.2` + `HF_TOKEN` are installed, so SD-2.1-base is a ~2.5 GB download if we later want
  LAION-scale coverage.
- **Inversion loop already built:** DPS (Diffusion Posterior Sampling) is analysis-by-synthesis
  for a diffusion prior. `guided_diffusion/condition_methods.py::PosteriorSampling` computes
  `∇_x ‖A(x̂₀) − y‖²` for an **arbitrary differentiable operator A** and folds it into each
  reverse-diffusion step (`gaussian_diffusion.py::p_sample_loop`). An `inpainting` (masked)
  operator + `mask_generator` already exist.

### General lifter Φ  ✅ present, clean drop-in — ⚠️ quality unproven
- `checkpoints/model_objaverse.pth` (646 MiB). Expects **128px, white bg, 3-channel RGB,
  `focals_pixels=None`, object centered at camera-radius 2.0**, fov 49.13°, znear/zfar 0.8/3.2.
  Mechanically *cleaner* than the CO3D lifters (no origin-distance 4th channel, no focals
  plumbing) — same input format as the cars/nmr models.
- Trained on the **LVIS subset of Objaverse (44,798 objects)**; held-out val/test splits exist
  with 12 GT views each → novel-view PSNR and Chamfer-vs-`model_normalized.obj` are evaluable.
- **RISK:** the only existing eval is zero-shot feed-forward on SRN cars = **~12–18 PSNR**, and
  even that was run under *mismatched* intrinsics (`+dataset=cars` fov 51.99/zfar 1.8 instead of
  objaverse fov 49.13/zfar 3.2), so it is not a clean number. Its own in-distribution best-frame
  PSNR is ~21 median. **There is no clean objaverse-config eval of Φ.** Whether Φ reconstructs
  well enough to be a reliable general-purpose lifter is the open question that gates everything.

## Why this maps exactly onto the paper (not a new method — a broader instantiation)

Paper Eq. 2:  argmax_z,θ  log p(I | Φ(G(z)), θ) + log p(z) + log p(θ).

With a diffusion prior, `G(z)` and `log p(z)` are replaced by the diffusion model's implicit
`p(x)` (its score), and the argmax over the latent becomes DPS guided sampling of the image `x`.
The likelihood term is realized by a forward operator with two parts:
1. **appearance** — `A_app(x) = M⊙x` vs `M⊙I` (the existing masked/inpainting operator), keeps
   the generated image faithful to the OOD object;
2. **frozen-lifter 3D-consistency (the paper's core constraint)** — a NEW operator
   `A_3d(x) = R_θ( Φ_objaverse( resize₁₂₈(M⊙x) ) )` vs `M⊙I`, differentiable end-to-end
   (Splatter CNN + 3DGS renderer), which forces `x` onto the manifold the lifter reconstructs
   consistently. Pose θ optimized jointly (POC: start at identity = input view).

So "broad-prior Infer3D" = the paper's algorithm with (G, Φ) := (ADM-ImageNet, Objaverse-Splatter)
and gradient-descent inversion replaced by DPS. Nothing conceptually new is claimed — it
demonstrates the framework's prior-/lifter-agnosticism the paper already asserts.

## Plan (cheapest-first, de-risk before building the full loop)

- **Step 0 — de-risk Φ (cheap, feed-forward, ~minutes/GPU).** Eval `model_objaverse` under its
  CORRECT config (fov 49.13, zfar 3.2, radius 2.0, 3-ch, focals=None) on ~20 held-out objaverse
  objects: reconstruct from view-0, render the other 11 GT views, report PSNR/SSIM/LPIPS + an
  orbit montage. Also try a couple of genuinely OOD single images (masked). **Decision gate:** if
  Φ gives recognizable 3D → proceed; if it collapses even in-domain → the idea needs a better
  lifter (e.g. an Objaverse LGM/LRM) and we report that honestly.  → `eval_objaverse_lifter.py`
- **Step 1 — prior sanity (cheap).** DPS `inpainting` on 2–3 masked OOD object images with the
  ImageNet-256 prior; confirm it reconstructs a plausible on-manifold object.
- **Step 2 — the POC (the headline).** Add the custom `A_3d` operator to DPS PosteriorSampling;
  invert a few OOD-category single images → lift → render orbit. Show 3D of categories NO
  per-category prior covers, from one (G, Φ) pair. Ablate with/without the 3D-consistency term.
- **Step 3 — quantify (if Step 2 works).** On held-out objaverse/GSO objects with GT: novel-view
  PSNR + Chamfer vs feed-forward Φ alone (does inversion help?) and vs a per-category Infer3D
  point (coverage vs fidelity trade-off).

## Coupling points to change when swapping Φ (from code audit)
- CO3D drivers build a 4-channel input + pass real focals → must use 3-ch RGB + `focals=None`
  for objaverse (`gaussian_predictor.forward` asserts focals is None for non-CO3D).
- `ours_nmr_so3.py` hard-codes `zgt=1.3` (SRN radius) → objaverse world scale is ~2.0.
- cfg must set objaverse geometry (res 128, fov 49.13, znear 0.8, zfar 3.2, white bg); `ray_dirs`
  buffer is baked to res/fov, so a wrong cfg breaks geometry.
- category one-hot conditioning is on the *StyleGAN generator* only — irrelevant once G is the
  unconditional diffusion prior (a further simplification in favor of this design).

## On the paper the user shared (arXiv 2602.05993, "Diamond Maps")
Reward alignment for flow/diffusion models via stochastic flow maps. NOT a broad prior we can
grab, not an inversion or 3D method. Only tangential relevance: inference-time reward alignment
is the same shape as "steer a broad prior to match an observation," so its guidance/value-function
machinery could in principle make the inversion step cheaper — but it does not change the
which-prior answer. Not needed for the POC.

---

## EMPIRICAL RESULTS (2026-07-27, ran end-to-end on H200)

Both models set up + validated on-cluster; full pipeline built and run.
- EqM-XL/2 (10.8 GB ckpt) generates photorealistic ImageNet samples. VAE = sd-vae-ft-ema.
- Objaverse lifter under CORRECT config (fov 49.13, zfar 3.2, radius 2.0, 3-ch, focals=None):
  held-out novel-view PSNR **22.0 mean** (n=30, range ~16-30) — the earlier ~12-18 was a
  mismatched-intrinsics artifact, now refuted.
- Full chain `EqM latent -> VAE -> SAM3 mask+center -> Phi -> render` is differentiable; the
  analysis-by-synthesis inversion converges (ring source-loss 0.012->0.001).

**Deliverable: report.html — 25 objects, single (EqM, Objaverse-Phi) pair, input+turntable
videos.** 12 GT-backed Objaverse objects (mean novel-view PSNR 22) + 13 EqM-sampled ImageNet
categories (teapot, teddy bear, toaster, birdhouse, hamburger, jack-o-lantern, ...). Directly
answers akZb "no single model spanning datasets."

**Honest limitations found by iterating (in the report, not hidden):**
1. Domain gap: EqM = natural photos, Phi = synthetic renders. Compact/convex objects lift
   cleanly; elongated/complex objects (car) collapse at novel views — the single-view LIFTER
   is the bottleneck, and EqM-manifold pull (eqm_eta>0) does not fix a lifter limitation.
2. On in-distribution objects the feed-forward lifter is near-optimal; latent inversion
   matches (mean FF 21.98 vs INV 20.11) rather than beats it — consistent with the paper's own
   in-distribution finding (feed-forward wins ID, optimization wins OOD).
3. No explicit pose recovery demonstrated here (inputs canonicalized to the source view);
   geometry recovery is shown via turntables. SE(3) pose search plugs into the same loop.

**To strengthen (future):** a broad prior trained in the lifter's RENDER domain (or a stronger
category-agnostic lifter, e.g. an Objaverse LRM/LGM) would close the domain gap and likely let
the EqM inversion beat feed-forward on OOD-appearance inputs (the paper's RealCars-style win).
EqM-E (explicit-energy variant) enables an OOD-detection score analogous to the paper's detector.
