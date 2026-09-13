# Released broad-distribution EBMs we could download (web survey, 2026-07-27)

Goal: an ImageNet-or-wider **energy-based** prior G with public weights, to invert against
the Objaverse Splatter lifter (R1/akZb: OOD objects + "no single model spanning datasets").

## Ranked candidates

### 1. EqM — Equilibrium Matching (Wang & Du, 2025)  ← RECOMMENDED
- arXiv 2510.02300, "Generative Modeling with Implicit Energy-Based Models".
  **Yilun Du's own framework** (co-author of our paper) — perfect narrative fit.
- **ImageNet 256×256**, FID 1.90 (XL/2). Checkpoints RELEASED (Google Drive):
  EqM-B/2 (80 ep): drive.google.com/file/d/1kDZGOri7Hf4CgnJAdEDguWooY3al37T6
  EqM-XL/2 (1400 ep): drive.google.com/file/d/1AfMLAxz18hthaGmYvQjB6c1LMxSEGly6
- Repo github.com/raywang4/EqM (models.py, sample_gd.py, transport/ ⇒ SiT/DiT lineage:
  latent-space with SD-VAE, class-conditional w/ cfg 1.0–1.5 in the eval table; DiT-style
  models carry a null-class embedding → unconditional inversion possible).
- **Sampling IS gradient descent on the learned energy landscape** (vanilla GD / NAG-GD,
  η≈0.0017) — no time conditioning. Inversion is therefore maximally natural:
  min_x  E(x) + λ‖A(x) − y‖²  with one combined gradient loop — literally the paper's
  Eq. 2 with an EBM prior. Also has an explicit-energy variant (EqM-E, `--ebm dot`) and
  the paper demonstrates OOD detection + composition (matches our detector story).
- Cost: ~GB-scale download via gdown + sd-vae-ft-ema (VAE decode is differentiable, so
  A(x)=Render(Φ(dec(x))) still works end-to-end).

### 2. EGC — diffusion energy-based model (Guo et al., ICCV'23)
- arXiv 2304.02012. Joint p(x,y) EBM: classifier forward pass, generator backward pass.
- **ImageNet-256 checkpoints RELEASED** (SharePoint links in github.com/GuoQiushan/EGC):
  FID 6.05 / 78.97% top-1. Latent space (autoencoder_kl.pth).
- True "EBM trained on ImageNet" headline; explicit energy via the classifier logits.
- Cons: SharePoint downloads are awkward from a cluster; repo admits incomplete release;
  ByteDance code quality unknown; FID worse than EqM.

### 3. Hat EBM (Hill et al., NeurIPS'22)
- github.com/point0bar1/hat-ebm — pretrained models released, **ImageNet 128×128**
  (standard + large nets). EBM over (residual, generator-latent) pairs.
- Cons: TensorFlow 2 (our stack is torch), 128px, weaker samples.

### 4. OpenAI EBM (Du & Mordatch, 2019) — github.com/openai/ebm_code_release
- Conditional **ImageNet 128**, pretrained models downloadable. The classic citation.
- Cons: TensorFlow 1 — painful to stand up in 2026; low sample quality by today's bar.

### 5. Improved Contrastive Divergence (Du et al., ICML'21)
- github.com/yilundu/improved_contrastive_divergence — checkpoints via Dropbox are
  **CelebA-HQ compositional** etc., NOT ImageNet-broad. Good citation, wrong breadth.

### (already on disk, non-"EBM-labeled" fallback)
- **ADM ImageNet-256 unconditional + DPS** at
  `/net/holy-isilon/ifs/rc_labs/ydu_lab/aakaran/diffusion-posterior-sampling/` — a score
  model = energy gradient; zero download, inversion loop already built. Use as fallback /
  second prior for robustness of the claim.

## Recommendation
**EqM-XL/2** as the headline broad EBM prior: (i) genuinely an energy-landscape model —
inversion by plain gradient descent matches the paper's optimization-based inference
exactly, with no diffusion-time machinery; (ii) ImageNet-1k breadth at 256px, SOTA-level
FID; (iii) released checkpoints + torch code; (iv) it is the co-author's framework, so the
rebuttal line writes itself: "the framework is prior-agnostic; instantiating G with an
implicit EBM trained on ImageNet [EqM] and Φ with an Objaverse-trained lifter yields a
single model pair spanning categories." Keep ADM+DPS as the on-disk fallback.

## Sources
- https://github.com/raywang4/EqM · https://arxiv.org/abs/2510.02300 · https://raywang4.github.io/equilibrium_matching/
- https://github.com/GuoQiushan/EGC · https://arxiv.org/abs/2304.02012
- https://github.com/point0bar1/hat-ebm · NeurIPS'22 paper
- https://github.com/openai/ebm_code_release
- https://github.com/yilundu/improved_contrastive_divergence
- https://github.com/alexiglad/EBT (Energy-Based Transformers, 2025 — language/video-centric, no broad image-prior checkpoint use here)
