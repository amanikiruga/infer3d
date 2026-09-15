# RealEstate10K under an unseen fisheye lens

A scene-level lifter (CATSplat) is trained on pinhole RealEstate10K. Here the input is
put through a fisheye lens it never saw, with no calibration supplied. Infer3D recovers
the lens by inverting the generative prior, undistorts, and reconstructs.

```bash
experiments/re10k_fisheye/run.sh [gpu] [n_seqs]     # default: gpu 0, 160 sequences
```

That runs: build clips → synthesize the fisheye inputs → blind calibration → evaluate
every condition through CATSplat → assemble the table.

## Files

| File | Role |
|---|---|
| `fisheye.py` | fisheye synthesis, analytic undistortion, PSNR |
| `diffae_prior.py` | loader for the frozen DiffAE prior (`re10k_autoenc_256`) |
| `select_calib.py` | **the method** — blind calibration by prior-naturalness voting |
| `eval_table3.py` | evaluation harness; builds each condition's source image and runs CATSplat |
| `catsplat_io.py` | CATSplat model loading and input preparation |
| `build_clips.py` | render the per-sequence test clips from the raw release |
| `make_table3.py` | assemble the table from per-sequence metrics |

## How the conditions are compared

`eval_table3.py` produces one *source image* per condition and then feeds all of them
through the **same frozen CATSplat** lifter, which predicts the 3D Gaussians and renders
the novel views that are scored. Only the source image differs:

| condition | source image |
|---|---|
| `clean` | the original pinhole frame |
| `fisheye` | the synthesized fisheye, fed in raw |
| `equidistant` | equidistant undistortion, *given the true FOV and focal* |
| `ours` | undistortion using the calibration recovered blind by `select_calib.py` |
| `oracle` | undistortion using the true calibration |

So the comparison isolates the calibration: same lifter, same renderer, same protocol
(split-file frame indices, 5% border crop), different estimate of the lens.

## The method

Blind analysis-by-synthesis calibration. Search over perspective FOV × radial-distortion
severity along the nominal lens profile, and score each candidate by how well the frozen
DiffAE prior autoencodes the undistorted image at a low diffusion step count (T=4). A
tight prior bottleneck reconstructs only geometrically natural — that is, correctly
undistorted — images, so the score peaks at the true camera. A coarse-to-fine grid on the
mean is followed by per-image paired voting among the top cells, which breaks the
FOV-severity degeneracy. No calibration labels and no ground truth enter the loop.

## Operating point

Fixed in `config.RE10K_FISHEYE`: `in_fov=75, out_fov=94, k_scale=0.5,
circle_scale=1.22`, nominal fisheye focal 190.4 px.

## Result

Full 160 sequences, reproduced by `make_table3.py` from the per-sequence metrics in
`results/table3_per_seq.json`:

| Setting | Method | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| In-dist. | CATSplat | 22.98 | 0.761 | 0.182 |
| OOD Fisheye | CATSplat direct | 15.87 | 0.450 | 0.355 |
| OOD Fisheye | Equidistant, given the true FOV | 19.18 | 0.608 | 0.266 |
| OOD Fisheye | **Infer3D, blind** | **22.17** | **0.736** | **0.218** |
| OOD Fisheye | Oracle undistortion | 22.67 | 0.752 | 0.212 |

Paired per-sequence, Infer3D minus baseline: **+6.30 dB** over CATSplat on the raw
fisheye (96% of sequences) and **+2.99 dB** over equidistant undistortion given the true
FOV (87%), landing 0.50 dB under the oracle ceiling.

## External models

- **CATSplat** at `$CATSPLAT_ROOT` with `$CATSPLAT_CKPT`, plus its UniDepth-v2-vitl14
  weights in `$HF_HOME`. Imported as a library for `datasets.util` and
  `evaluation.evaluator`.
- **DiffAE** at `$DIFFAE_ROOT`, providing `templates` and `experiment.LitModel`, with
  `checkpoints/re10k_autoenc_256/last.ckpt`.
