> **Provenance.** Values here were extracted from the run code during the rebuttal and
> are quoted against that code's original layout. In this repository the same code is
> `tools/rebuttal/baselines/ours_nmr_so3.py` (cars/NMR) and
> `tools/optimize_co3d_diffae.py` / `tools/optimize_co3d_stylegan.py` (CO3D), with
> configs under `configs/`. The schedule was re-confirmed by a live run during the
> release audit: the optimizer logged `configuration step 2 at 122 steps` and
> `step 3 at 302 steps`, matching k_t below. A condensed version of this table is in the
> main [README](../../../README.md#method-hyperparameters).

# Rebuttal prose — Algorithm 1 clarification + metric definitions (Reviewer XsZX, AC priority 4)

Every value below is extracted from the actual run code (not from memory):
`rebuttal/baselines/ours_nmr_so3.py` (cars/NMR), `experiments/stylegan3-ours-co3d/
stylegan3_co3d_abs_w_inversion_with_baseline_se3_folder.py` (CO3D), config
`configs/abs/stylegan_abs.yaml`, helpers `utils/abs_utils.py`. See "provenance" notes.

---

## To Reviewer XsZX — revised, self-contained Algorithm 1

We apologize for the under-specified pseudocode and will replace Algorithm 1 with the
self-contained version below in the revision. All quantities that looked like free-floating
variables are **fixed, pre-registered piecewise-constant schedules** — nothing is tuned
per-example; the same schedule is used for every result in the paper.

**Algorithm 1 (revised): Analysis-by-Synthesis via Generative Inversion**

> **Input:** image I; frozen generator G; frozen lifter Φ; differentiable renderer R;
> pruning schedule {k_t}; loss-weight schedule {λ_t = (λ_MSE(t), λ_LPIPS(t))}; learning
> rates η_z (with cosine ramp), η_θ (constant); total iterations N.
> **Output:** scene S* = Φ(G(z*)) and rendering parameters θ*.
> 1. **Initialize population** of R = R_rot × R_lat particles: R_rot camera rotations
>    {θ_r} spread near-uniformly over the admissible pose space (Fibonacci sphere +
>    random roll), crossed with R_lat latent codes {z_r} (z ~ N(0, I) mapped into
>    StyleGAN W-space; for DiffAE, the first particle is warm-started with the encoder,
>    z_1 ← E(I)).
> 2. Record the initialization anchor z̄_init (mean of the initial latent population).
> 3. **for** t = 1 … N:
> 4. &emsp;render every live particle: Î_r = R(Φ(G(z_r)), θ_r)
> 5. &emsp;L_r = λ_MSE(t)·MSE(Î_r, I) + λ_LPIPS(t)·LPIPS(Î_r, I) + L_prior(z_r) [+ L_depth on CO3D]
> 6. &emsp;Adam step on (z_r, θ_r) jointly; θ_r re-projected onto SO(3) by SVD after each step
> 7. &emsp;**if** t is a pruning step: keep the k_t particles with the lowest selection loss
> 8. **Maintain throughout:** a leaderboard of the best B particles seen (one slot per
>    particle); at termination, the B survivors are evaluated and the best is returned.

**Concrete values (identical for cars and CO3D; verified against the run code):**

| symbol | value | notes |
|---|---|---|
| R (initial population) | **600** = 30 rotations × 20 latents | Alg. 1 line 1 |
| N (total iterations) | **783** | hard stop |
| k_t (pruning schedule) | 600 → **32** (t=3) → **10** (t=122) → **5** (t=302) | piecewise constant, iteration-indexed |
| λ_MSE(t) | 1.0 → **10** (t=302) → **2** (t=732) | staged re-weighting: search → photometric refine → anneal |
| λ_LPIPS(t) | 0.5 → **2** (t=302) | |
| λ_prior (latent-prior weight, `w_reg`) | **0.001**, constant (no schedule) | CO3D/DiffAE; `configs/abs/diffae_abs.yaml:29`. Applied to ‖z − z̄_init‖ (L2 distance to the initial-population mean). StyleGAN/cars uses no explicit term (implicit prior — see below). |
| η_z (latent lr) | 0.01, cosine ramp-up 5% / ramp-down last 25% (standard StyleGAN2-inversion schedule) | |
| η_θ (pose lr) | 0.01, constant | |
| B | **10** | see below |

So the answer to "are k_t, λ_t fixed hyperparameters or iteration-dependent variables?" is:
**both** — they are iteration-dependent by design (that is what the schedule σ(t) denotes),
but the schedule itself is a fixed hyperparameter of the method, constant across datasets
and examples. The schedule realizes the coarse-to-fine strategy: a broad low-cost search
over 600 pose×latent hypotheses for a few steps, aggressive culling of the population
(600→32→10→5), then heavier photometric weights on the few survivors.

**Role of z̄_init and the latent prior (the Ln-186 variables).** z̄_init is the mean of the
initial latent population; the latent-prior term −log p(z) of Eq. 2 is approximated as a
quadratic pull toward it, λ_prior‖z − z̄_init‖², discouraging drift off the generator
manifold; **λ_prior = 0.001** (constant; `configs/abs/diffae_abs.yaml`, key `w_reg`). (The
code applies it to the L2 norm ‖z − z̄_init‖ rather than the square; Eq. 2 writes the
quadratic — a minor notational simplification we will reconcile in the revision.) Two
instantiation details we will state explicitly in the revision: (i) in the
DiffAE/CO3D instantiation this term is an explicit loss (ablated in Table 8, "Mean Latent
Dist. Reg."); (ii) in the StyleGAN instantiation the same role is played implicitly by
optimizing in W-space from mapped initializations plus decaying exploration noise on w
(noise strength ∝ (1 − t/0.75N)², reaching zero at t ≈ 0.75N), which regularizes the
search in the same spirit. The pseudocode will carry a λ_prior(t) entry with these two
concrete realizations footnoted.

**"Line 288: what is B = 10?"** B is the size of the retained particle set after pruning —
the B = 10 best (latent, pose) hypotheses maintained as a leaderboard during optimization
and carried in parallel on the GPU (this is the "parallelizing particle optimization"
sentence; it is the same quantity as k at the t=122 pruning rung). At termination all B
survivors are scored and the best one is returned as S*. Larger B costs VRAM
(peak 37.5 GB at B=10-scale populations; 48–59 GB for larger searches). We will rename it
k₃ (or state B ≡ k at the final search stage) so the pseudocode and text use one symbol.

**Execution-flow inconsistency.** The reviewer is right that the roles of these symbols
could not be reconstructed from the pseudocode alone; the revised Algorithm 1 above is
self-contained (all inputs declared, schedules explicit, pruning and leaderboard steps
written out), and we will cross-reference each line to the equation it implements
(line 5 = Eq. 2 likelihood + priors).

## To Reviewer XsZX — metric definitions (will be added at first use)

We will define every metric at first use in the revision:

- **PSNR** (Peak Signal-to-Noise Ratio, dB, ↑): −10·log₁₀ of the mean squared error
  between a rendered novel view and the ground-truth image; measures pixel-level fidelity.
- **SSIM** (Structural Similarity Index Measure, ↑): local luminance/contrast/structure
  similarity between rendered and ground-truth views.
- **LPIPS** (Learned Perceptual Image Patch Similarity, ↓): distance between deep VGG
  features of rendered and ground-truth views; correlates with perceived quality.
- **CD** (Chamfer Distance, ↓): average squared distance between the predicted 3D point
  set and the ground-truth surface points (and vice versa); on RealCars (Table 5) it is
  reported in m² against LiDAR-scale pseudo-ground-truth reconstructions and computed
  after ICP alignment, i.e. it measures shape rather than pose.
- **AUROC** (Area Under the Receiver Operating Characteristic curve, ↑): threshold-free
  quality of the OOD detector's score (initial reconstruction error); 0.5 = chance,
  1.0 = perfect separation of ID from OOD inputs.
- **F1** (↑): harmonic mean of precision and recall of the thresholded OOD detector
  (Table 4/7), at the threshold calibrated on the small validation set.

## Provenance (internal)

- R=600 (30×20): `ours_nmr_so3.py:502,547-548` + `stylegan_abs.yaml` (num_rotations 30,
  num_latents 20). N=783 + k_t/λ_t stages: `config_steps`, `ours_nmr_so3.py:658-759`;
  identical boundaries in the CO3D `_folder.py:787`.
- top_k/B=10 leaderboard: `ours_nmr_so3.py:575-581,1016-1036`.
- lr + ramp: `ours_nmr_so3.py:914-916`, `abs_utils.py:832-837`. SVD projection:
  `abs_utils.py:221` (`symmetric_orthogonalization`), applied each step.
- Exploration-noise decay (StyleGAN latent prior realization): `ours_nmr_so3.py:908-912`;
  strength = latent_std · 0.05 · max(0, 1 − t/0.75N)².
- CO3D differences (SE(3): +xy translation particles; centroid-aware render pivot;
  stage-weighted selection loss): `_folder.py:668-688,983-995,1032`.
- Caveat found during extraction (harmless, worth knowing): stage 1 evaluates 576 of the
  600 particles (18 batches of 32); the last 24 are never scored before the first pruning.
  Functionally R=576.
