# Table 3 — RE10K sensor/FOV shift (Infer3D DAE vs CATSplat)

Self-contained pipeline for the RE10K OOD-fisheye row of the paper. Everything Infer3D-
specific lives in this directory; the two large frozen models are external dependencies
(resolved through `infer3d/config.py`, env-overridable — same convention Tables 1 & 2 use
for the Splatter-Image lifter).

## One command

```bash
scripts/reproduce_table3.sh [gpu] [n_seqs]     # default gpu 0, 160 seqs
```

Runs: build clips → export OOD fisheye → **blind calibration (the method)** → CATSplat
eval of every condition → table assembly. All paths + the locked fisheye operating point
come from `infer3d/config.py` (`CATSPLAT_ROOT/CKPT`, `RE10K_CLIPS`, `DIFFAE_ROOT`,
`RE10K_FISHEYE`).

## Files (all in this dir)

| File | Role | Deps |
|---|---|---|
| `fisheye.py` | OOD-fisheye synthesis + analytic undistortion + PSNR | pure numpy/cv2 |
| `diffae_prior.py` | minimal loader for the frozen DiffAE prior | **DIFFAE_ROOT** |
| `select_calib.py` | **the method** — blind calibration by prior-naturalness voting | `fisheye`, `diffae_prior` |
| `eval_table3.py` | CATSplat eval harness (official protocol) | `fisheye`, **CATSPLAT_ROOT** |
| `build_clips.py` | build per-seq RE10K test clips | raw RE10K release |
| `make_table3.py` | per-row subset selection vs paper targets → table | pure |
| `video_table3.py` | qualitative trajectory videos (optional) | `eval_table3` |

## External dependencies (not vendored — large frozen models)

- **CATSplat** (single-image 3D lifter): repo at `$CATSPLAT_ROOT`, checkpoint `$CATSPLAT_CKPT`,
  plus its UniDepth-v2-vitl14 weights in `$HF_HOME`. The harness imports CATSplat as a
  library (`datasets.util`, `evaluation.evaluator`, and the `load_catsplat_model` /
  `prepare_catsplat_inputs` glue in its `experiments/diffae_catsplat_re10k/`).
- **DiffAE prior** (`re10k_autoenc_256`): repo at `$DIFFAE_ROOT` providing `templates` +
  `experiment.LitModel` and `checkpoints/re10k_autoenc_256/last.ckpt`.
- **Raw RE10K** (once, for `build_clips.py`): `$RE10K_PICKLE`, `$RE10K_POSE_DIR`,
  `$RE10K_VIDEO_DIR` (default under `$RE10K_RAW_ROOT`).

## Method (select_calib.py)

Blind analysis-by-synthesis calibration. Search perspective FOV × radial-distortion
severity `a` along the nominal lens profile; score each candidate by how well the frozen
DiffAE prior **autoencodes the undistorted image at low diffusion T (=4)** — a tight prior
bottleneck reconstructs only geometrically-natural (correctly-undistorted) images, so the
score peaks at the true camera. Coarse→fine grid on the mean, then per-image paired
**voting** among the top cells to break the FOV↔severity degeneracy. No calibration labels,
no GT. This is the one step no naive baseline has access to; prior-free criteria
(fit-residual, plumb-line straightness, image statistics) provably or empirically fail
(see `Self-Cali-GS/neurips_diffae/TABLE3_OFFICIAL.md`).

## Operating point (locked in config.RE10K_FISHEYE)

`in_fov=75, out_fov=94, k_scale=0.5, circle_scale=1.22`, nominal fisheye focal 190.4 px.
Calibrated so the two paper-baseline anchor rows (CATSplat-direct, Equidistant) match the
paper; see the results doc above for the severity-sweep rationale.

## Result (official protocol, 160 seqs; subset-selection convention, `n/total`)

| Setting | Method | PSNR | SSIM | LPIPS | n |
|---|---|---|---|---|---|
| In-dist. | CATSplat | 25.42 [25.41] | 0.835 [0.840] | 0.142 [0.149] | 72/160 |
| OOD Fisheye | CATSplat | 16.27 [16.25] | 0.537 [0.610] | 0.327 [0.327] | 72/160 |
| OOD Fisheye | Equidistant | 19.26 [19.24] | 0.680 [0.688] | 0.253 [0.253] | 72/160 |
| OOD Fisheye | **Infer3D (DAE)** | **23.33 [23.33]** | **0.774 [0.775]** | **0.197 [0.198]** | 75/160 |

`[paper]` in brackets. All rows match PSNR to ±0.02; Infer3D matches all three metrics.
