> Evaluation protocol for the optimization-baseline comparison. Scripts implementing
> it are in `baselines/`; the resulting numbers are in [../../RESULTS.md](../../RESULTS.md)
> Part II §1.

# Rebuttal evaluation protocol — optimization-based baselines vs Infer3D

**Goal (AC request).** Show empirically whether Infer3D's *specific* design (2D generative
prior + frozen 3D lifter + particle search) advances over what optimization-based
reconstruction (GAN/NeRF inversion) already offers. Baselines: pi-GAN inversion,
3D-GAN-Inversion (EG3D+PTI+pose opt), nerf-from-image (bootstrapped radiance-field
inversion), FINV (filtered multi-start PTI). All use **pretrained generators only** (no
retraining) — same rule our method follows.

Two tasks, chosen to match the paper's tables exactly:

---

## Task A — SO(3) viewpoint generalization, ShapeNet cars

**Our setup (paper Table 2, restricted to the car class).** Prior (StyleGAN2) and lifter
(Splatter Image) trained on ShapeNet-NMR renders from the *side-top* elevation ring
(±15°). OOD input = render of the same object from a random SO(3) rotation
(pre-rendered `so3/` view folders). The method reconstructs 3D **in the frame of the
input camera**; novel-view metrics are computed on the 24 `test/`-ring views expressed
*relative to the input camera* (make_data_relative_to). Metrics: PSNR/SSIM/LPIPS,
128×128, white background, mean over objects.

**Our numbers (cached, car class, n=34):** Infer3D (SG) PSNR 21.25 / SSIM 0.875 /
LPIPS 0.135. Splatter Image at the OOD pose: PSNR 14.61 (n=22, corrected baseline).
Source: `checkpoints-stylegan3-shapenet-nmr-w-inversion-baseline/side_top_so3_results.csv`
+ `checkpoints-so3-baseline-probe/` filtered by
`rendering_scripts/vincent-renderer/object_ids_per_category_nmr.json`.

**Baseline runs.** Same 34 car objects, same SO(3) input image (identical PNG), same
relative target views, same metric code. Each baseline uses its own released pretrained
car generator:
- nerf-from-image → `shapenet_cars` (trained on SRN cars, full-sphere: its prior is NOT
  viewpoint-limited — favourable to the baseline).
- EG3D-PTI / 3D-GAN-Inversion → NVIDIA `shapenetcars128-64.pkl` (trained on SRN cars,
  full-sphere — favourable).
- pi-GAN → CARLA cars (full-sphere azimuth — favourable in pose coverage).
- FINV → GET3D/EG3D ShapeNet car generator, `num_train=1` (single view; off-label but
  the only single-view instantiation possible).

**Renderer-domain control (fairness).** The baselines' car priors were trained on SRN/
CARLA renders, not NMR renders. To separate "renderer gap" from "SO(3) pose gap", each
baseline is ALSO evaluated with the **canonical side-top input view** of the same
objects (an in-distribution-pose input). Reporting both isolates the pose-generalization
failure from any appearance-domain penalty.

**Pose handling per baseline (soft-constraint #1: failures must be inherent).**
- Baselines that estimate/optimize pose (nerf-from-image, EG3D-PTI, pi-GAN w/ pose opt
  enabled) run **without any GT pose** — same information our method gets.
- FINV requires a camera for its observation; it has no pose estimation. Two rows:
  (a) *as-usable*: canonical (side-top-like) camera assumption — what a user without GT
  pose must do; (b) *privileged*: GT SO(3) input pose provided. (b) is an upper bound
  the method cannot claim in practice; both reported, clearly labelled.

## Task B — Appearance-shift generalization, RealCars (synthetic→real)

**Our setup (paper Table 5).** Prior trained ONLY on synthetic SRN cars; lifter =
Splatter Image SRN-cars weights. 20 test frames (1/scene) from
`datasets/realcars/HQ339`, listed in
`experiments/neurips_submission/real_ood/realcars_test_paths.csv`. Input processing:
SAM2 mask → tight crop → 128×128; white-bg version (`rgb_128`) for pixel losses;
original-bg crop for DINOv2. Metric: **asymmetric Chamfer pred→gt (m²)** against
per-scene FastGS pseudo-GT cropped to a ball around the ARKit trajectory centroid;
alignment = diameter pre-scale + FPFH+RANSAC global + point-to-point ICP
(`eval_chamfer_direct.align`). Mean/median over the 20 frames.

**Our numbers (cached, verified = paper):** Ours 0.131/0.093 · Splatter 0.219/0.179 ·
LGM 0.405/0.307 · SF3D 0.242/0.206 (`results/chamfer_table_4col.md`).

**Baseline runs.** Same 20 frames. Input = the SAME SAM2-masked white-bg 128×128 crop
(`rgb_128`) our pixel-losses consumed. Baselines with real-image machinery may also use
the raw frame (noted per-run). Output 3D → point samples → identical align + chamfer
pred→gt. Chamfer needs geometry only, so canonical-frame reconstructions are fine (ICP
absorbs the unknown similarity transform — same treatment ours/LGM/SF3D got).
- nerf-from-image: two rows — `shapenet_cars` prior (matched training data: the fair
  head-to-head) and `p3d_car`/`imagenet_car` prior (real-photo prior = privileged, its
  best case).
- EG3D-PTI: shapenetcars generator; PTI weight-tuning is its mechanism for appearance
  adaptation — that is exactly what the AC wants tested.
- pi-GAN: CARLA generator, latent-only inversion (+pose opt).
- FINV: GET3D car generator, single view + PTI; camera = generator canonical default
  (its no-pose reality) and a privileged variant with an ARKit-derived pose.

## Reporting rules
1. Every number comes from the same metric code as ours (no re-implementations).
2. Per-method capability table: optimizes pose? has encoder init? prior trained on what?
   single-view native? what it inherently cannot do.
3. Qualitative videos for every method × task (novel-view orbits, input/recon
   side-by-side) — sanity-checked before trusting numbers.
4. Failure attribution: for each bad number, a stated cause backed by a visualization
   (e.g. "pose local minimum: orbit video shows back-front flip").
5. Any accommodation given to a baseline (pose, mask, init) is listed in the run notes;
   privileged rows are marked †.
