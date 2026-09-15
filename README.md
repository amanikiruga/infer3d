# Infer3D

Single-image 3D reconstruction that holds up when the input is out of distribution.

A feed-forward lifter (Splatter Image, CATSplat) turns one image into 3D Gaussians, and is
accurate as long as the input looks like its training data. Infer3D keeps that lifter
frozen and, at test time, searches the latent **z** of a frozen 2D generative prior and the
rendering parameters **θ** for the pair whose render best explains the observed image:

```
z ──► G (2D prior) ──► Φ (3D lifter) ──► R(·,θ) ──► rendered image
▲                                                        │
└────────────── gradient of ‖render − input‖ ────────────┘
```

The lifter only ever sees an image the prior generated, so the search cannot leave the
manifold where the lifter is valid. Nothing is trained; both networks stay frozen.

This repository contains the three main experiments and nothing else.

```
infer3d/                  the method
  config.py               all paths, one file
  model.py                the frozen Splatter-Image lifter
  renderer.py             differentiable Gaussian rasterization
  utils/optim.py          the search: particles, pruning, pose parameterization
  data/                   CO3D and SRN-Cars loaders
experiments/
  co3d_ood/               unseen camera poses     (CO3D hydrants, vases)
  re10k_fisheye/          unseen lens             (RealEstate10K)
  realcars/               unseen appearance       (real cars vs synthetic training)
configs/                  hydra configs
assets/                   the test splits, so object selection is fixed
```

## 1. Environment

```bash
conda create -n infer3d python=3.12 && conda activate infer3d
pip install -r requirements.txt
```

## 2. Two dependencies that are not on PyPI

The CUDA rasterizer, and PyTorch3D for ICP. Both need a compiler and a matching CUDA
toolkit; install them before going further, because they are the only fiddly part.

```bash
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## 3. Models and data

Put everything under one directory and point `INFER3D_EXTERN_ROOT` at it:

```bash
cp .env.example .env        # edit INFER3D_EXTERN_ROOT, then:
source .env
mkdir -p $INFER3D_EXTERN_ROOT && cd $INFER3D_EXTERN_ROOT
```

**The 2D prior** is a Diffusion Autoencoder. Clone the repo (we import its model
definitions, so weights alone are not enough) and drop the checkpoints in:

```bash
git clone https://github.com/phizaz/diffae
# then place the trained priors at:
#   diffae/checkpoints/co3d_hydrants_autoenc_128/last.ckpt
#   diffae/checkpoints/co3d_vases_autoenc_128/last.ckpt
#   diffae/checkpoints/re10k_autoenc_256/last.ckpt
```

**The 3D lifters.** Splatter Image for CO3D and RealCars, CATSplat for RE10K:

```bash
mkdir -p lifters      # co3d_hydrants.pth, co3d_vases.pth, srn_cars.pth
git clone https://github.com/szymanowiczs/splatter-image    # srn_cars weights
git clone https://github.com/kuai-lab/iccv25_CATSplat CATSplat   # + CATSplat.pth
```

**Meshing**, used by the Chamfer evaluations:

```bash
git clone https://github.com/yanivw12/gs2mesh
```

**CO3D.** Download only the two categories the experiments use, then preprocess:

```bash
git clone https://github.com/facebookresearch/co3d && cd co3d
python co3d/download_dataset.py --download_folder $INFER3D_EXTERN_ROOT/co3d_raw \
       --download_categories hydrant,vase
cd .. && CO3D_RAW_ROOT=$INFER3D_EXTERN_ROOT/co3d_raw \
       python experiments/co3d_ood/preprocess_co3d.py
```

Preprocessing crops and masks each sequence and writes `co3d_processed_1080/`. The full
CO3D release is 5.5 TB; these two categories are a small fraction of it.

**RealEstate10K.** The release provides per-sequence camera trajectories and the
YouTube ids they came from; the frames are extracted from the source videos. Arrange the
raw pieces as

```
$INFER3D_EXTERN_ROOT/re10k/
  RealEstate10K/test/<seq>.txt          per-sequence poses (first line = YouTube id)
  videos/<youtube_id>.mp4               source videos
  flash3d_anns/catsplat_test_256.pickle.gz   per-sequence frame indices
```

then render the per-sequence test clips the loader reads:

```bash
python experiments/re10k_fisheye/build_clips.py --out $RE10K_CLIPS
```

Each clip holds the frame at every pose timestamp in order, so clip frame *i* lines up
with pose *i*. The step skips sequences already built, so it is safe to re-run.

**RealCars** is 20 ARKit car captures with per-scene multi-view pseudo-ground-truth
(a Gaussian-Splatting fit to the full capture, meshed through gs2mesh). Extract to

```
$INFER3D_EXTERN_ROOT/realcars/HQ339/<scene>/      frame_*.jpg + frame_*.json
$INFER3D_EXTERN_ROOT/realcars_pseudo_gt/<00..19>/ per-scene pseudo-GT
```

`assets/realcars_test_paths.csv` pins which 20 scenes and which frame of each is the
single input view.

Check what resolved:

```bash
python -c "from infer3d import config as c; print(c.CO3D_DATASET_ROOT_HQ, c.DIFFAE_ROOT)"
```

## 4. Run

Each script runs one experiment end to end -- search, then meshing and scoring -- and
prints its table.

```bash
experiments/co3d_ood/run.sh hydrants 0     # category, gpu
experiments/re10k_fisheye/run.sh 0         # gpu
experiments/realcars/run.sh 0              # gpu
```

What each stage does:

| | units | stages |
|---|---|---|
| `co3d_ood` | 41 objects | search → gs2mesh → Sim(3)-ICP → Chamfer + novel views |
| `realcars` | 20 scenes | search → PLY → ICP against pseudo-GT → Chamfer |
| `re10k_fisheye` | 160 sequences | synthesize fisheye → blind calibration → undistort → CATSplat → render → score |

In every condition of the RE10K experiment the source image is fed through the same
frozen CATSplat lifter, which predicts the Gaussians and renders the novel views that
get scored; the conditions differ only in what that source image is (raw fisheye,
equidistant undistortion given the true FOV, blind undistortion, oracle undistortion).

All three loops skip work that already exists, so they can be interrupted and resumed.
To try a single unit first, pass `+general.maxsamples=1` to `co3d_ood/optimize.py`, or a
scene range to `realcars/run.sh 0 0 0`.

Every search uses the same fixed schedule: 600 initial hypotheses (30 rotations × 20
latents) pruned to 32, then 10, then 5 over 783 iterations.

## 5. Checking the numbers without running anything

The per-scene and per-sequence results behind two of the tables below ship with the repo
(11 KB), so those can be re-derived with no data, no GPU and no install:

```bash
python experiments/realcars/score.py
python experiments/re10k_fisheye/make_table3.py experiments/re10k_fisheye/results/table3_per_seq.json
```

Both print full-set means and the paired per-scene difference. CO3D has no equivalent
shortcut: its numbers come out of the meshing stage, so you have to run it.

## 6. What you should get

**CO3D, out-of-distribution camera poses** (hydrants, full test set, n=41). Novel views
after ICP alignment; Chamfer in the dataset's units.

| | PSNR | SSIM | LPIPS | Chamfer |
|---|---|---|---|---|
| Splatter Image (frozen lifter) | 15.79 | 0.691 | 0.300 | 0.727 |
| Infer3D | **17.67** | **0.740** | **0.225** | **0.453** |

Per-object variance is high, so the stable quantity is the paired difference on the same
objects: **+1.9 dB PSNR**, Infer3D ahead on 71% of objects.

**RealEstate10K, unseen fisheye lens** (n=160). Infer3D is given no calibration and
recovers the lens itself.

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| CATSplat on the raw fisheye | 15.87 | 0.450 | 0.355 |
| Equidistant undistortion, *given the true FOV* | 19.18 | 0.608 | 0.266 |
| Infer3D, blind | **22.17** | **0.736** | **0.218** |
| Oracle undistortion (ceiling) | 22.67 | 0.752 | 0.212 |

**RealCars, synthetic-to-real appearance** (n=20). Both priors saw only synthetic cars.
Chamfer in m² against the multi-view pseudo-ground-truth.

| | mean | median |
|---|---|---|
| Splatter Image | 0.219 | 0.179 |
| Infer3D | **0.131** | **0.093** |

## Notes

- `general.prefix` is relative to `$INFER3D_RUNS_ROOT`, and hydra changes directory at
  startup, so pass absolute paths for any file arguments.
- Keep `WANDB_MODE=disabled`. wandb is imported by the eval loops but nothing is logged,
  and version 0.22 crashes the ICP stage if it is live.
- The search schedule is fixed and identical across all three experiments: 600 initial
  hypotheses (30 rotations × 20 latents), pruned to 32, then 10, then 5 over 783
  iterations. It is defined in `experiments/co3d_ood/optimize.py`.
- `torch.load(..., weights_only=False)` is used for checkpoints that carry an OmegaConf
  config. Only load checkpoints you trust.

## Credit

Built on Splatter Image (Szymanowicz et al.), 3D Gaussian Splatting (Kerbl et al.),
Diffusion Autoencoders (Preechakul et al.), CATSplat, and gs2mesh. Each is under its own
license and is cloned rather than vendored here.
