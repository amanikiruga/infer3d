# Infer3D: 3D Reconstruction through Inverse Generative Modelling

Reference implementation and full experimental record.

Feed-forward single-view 3D models are accurate when the input matches their training
distribution and degrade out of it. Infer3D keeps a feed-forward lifter **frozen** as a
3D prior and, at test time, optimizes the latent **z** of a frozen 2D generative prior
together with the rendering parameters **θ** so that the rendered reconstruction explains
the observed image. Because the lifter only ever sees a generated in-distribution image,
the search stays on a valid, multi-view-consistent 3D manifold.

```
        z  ──►  G (frozen 2D prior)  ──►  Φ (frozen 3D lifter)  ──►  R(·, θ)  ──►  Î
        ▲                                                              │
        └──────────────  ∇ of  L(Î, I_OOD)  over (z, θ)  ◄─────────────┘
```

---

## Start here

Two commands, no GPU, no datasets, no model weights, no install — they recompute the
paper and rebuttal tables from per-object result files bundled in this repository:

```bash
PYTHONPATH=. python3 tools/make_tables.py        # paper Tables 1 & 2
PYTHONPATH=. python3 tools/rebuttal/report.py    # every rebuttal table
```

Both run on a bare Python 3 standard library in a few seconds. Then read:

* **[RESULTS.md](RESULTS.md)** — every quantitative claim in the paper and the rebuttal,
  the artifact behind it, and whether it was independently re-derived here.
* **[REPRODUCTION.md](REPRODUCTION.md)** — the reproduction audit, including the places
  where published numbers do **not** hold up and why.
* **[DATA.md](DATA.md)** — datasets and frozen checkpoints, and where to get them.

We report what reproduced and what did not. Three findings are stated up front because
they change how the paper's tables should be read:

1. The original table driver selected the subset of test objects whose mean best matched
   the published target. It has been replaced with full-set means and paired statistics.
2. On ShapeNet-NMR the published "ours" column selects the best particle by
   **ground-truth** novel-view PSNR. The honest, loss-selected column is reported beside it.
3. Under the standard evaluation protocol the ShapeNet-NMR **SO(3)** result reverses.
   The SE(3), CO3D, RE10K and RealCars results hold.

## Layout

```
README.md                 this file
RESULTS.md                every number in the paper + rebuttal, with provenance
REPRODUCTION.md           audit of the main paper's tables
DATA.md                   datasets and frozen checkpoints
.env.example              copy to .env and edit the roots

infer3d/                  importable core library
├── config.py             ALL paths and checkpoints (env-overridable) — edit this first
├── model.py              GaussianSplatPredictor, the frozen Splatter-Image lifter
├── renderer.py           differentiable Gaussian rasterization
├── generators.py         DiffAE / StyleGAN priors (frozen)
├── data/                 CO3D and ShapeNet-NMR datasets
└── utils/                geometry, losses, the optimization loop (optim.py)

tools/
├── check_config.py       report which configured paths exist and what they unlock
├── smoke_test.py         ~30 s end-to-end check: build + load + forward + CUDA render
├── optimize_co3d_diffae.py     test-time optimization, CO3D, DiffAE prior
├── optimize_co3d_stylegan.py   test-time optimization, CO3D, StyleGAN prior
├── optimize_nmr_se3.py         ShapeNet-NMR, OOD SE(3)
├── optimize_nmr_so3.py         ShapeNet-NMR, OOD SO(3)
├── eval/                 CO3D gs2mesh + Sim(3)-ICP eval → Chamfer + novel-view metrics
├── ood_detection/        Tables 4, 6, 7 — detector, AUROC, F1 sweep, adaptive stream
├── realcars/             Table 5 — RealCars Chamfer against ARKit pseudo-GT
├── re10k/                Table 3 — fisheye synthesis, blind calibration, eval
├── rebuttal/             everything added during review — see its README
└── make_tables.py        Tables 1 & 2 from per-object CSVs (full-set, no selection)

configs/                  hydra configs (abs_config + abs/* + dataset/*)
assets/                   splits, intrinsics, test-path CSVs, bundled per-object eval CSVs
scripts/                  verify.sh (no GPU) + one-command reproduce_*.sh wrappers
```

## Install (only needed to re-run experiments)

Datasets and frozen model weights are described in **[DATA.md](DATA.md)**.

```bash
conda create -n infer3d python=3.12 && conda activate infer3d
pip install -r requirements.txt
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
```

The 2D priors (DiffAE, StyleGAN3), the CATSplat lifter used for RE10K, and the four
inversion baselines compared in the rebuttal are external repositories under their own
licenses; clone them and point `infer3d/config.py` at them.

## Configure

Every path lives in [`infer3d/config.py`](infer3d/config.py) and is overridable by
environment variable. Copy `.env.example` → `.env`, edit the roots, `source .env`.

| What | Env var |
|---|---|
| CO3D (preprocessed) | `CO3D_DATASET_ROOT` |
| ShapeNet-NMR, and its SE(3) renders | `SHAPENET_NMR_ROOT`, `SHAPENET_NMR_SE3_ROOT` |
| DiffAE / StyleGAN3 repos + checkpoints | `DIFFAE_ROOT`, `STYLEGAN3_ROOT` |
| Trained Splatter-Image lifters | `SPLATTER_REPO_ROOT` |
| CATSplat (Table 3) | `CATSPLAT_ROOT`, `CATSPLAT_CKPT` |
| RealCars + gs2mesh (Table 5) | `REALCARS_ROOT`, `GS2MESH_INPUT` |
| Rebuttal baselines + intermediates | `BASELINES_ROOT`, `REBUTTAL_ASSETS` |
| **Where runs are written** | `INFER3D_RUNS_ROOT` (default `./runs`, must be writable) |

Only `INFER3D_RUNS_ROOT` is written to; every other root may be read-only. To see which
paths resolve on your machine and what each one unlocks:

```bash
PYTHONPATH=. python tools/check_config.py
```

## Running the method

```bash
source .env
RUNS=${INFER3D_RUNS_ROOT:-$(pwd)/runs}           # where runs land; default ./runs
PYTHONPATH=. python tools/smoke_test.py          # -> "SMOKE TEST PASSED"

# Stage A — test-time optimization (CO3D Hydrants, OOD SE(3), DiffAE prior)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tools/optimize_co3d_diffae.py \
    abs=diffae_abs +dataset=hydrants general.run_indist=false \
    general.prefix=hydrants-ood \
    opt.pretrained_ckpt=$SPLATTER_REPO_ROOT/experiments_out/2025-10-07/16-13-13/model_latest.pth \
    +general.test_imgs_csv_path=$(pwd)/assets/co3d_test_paths_1080.csv

# Stage B — Chamfer + novel-view metrics via gs2mesh + Sim(3)-ICP
PYTHONPATH=. WANDB_MODE=disabled python tools/eval/run_eval_pipeline.py \
    --checkpoint_dir $RUNS/hydrants-ood \
    --output_base_dir $RUNS/eval \
    --dataset_name hydrants \
    --pretrained_ckpt $SPLATTER_REPO_ROOT/experiments_out/2025-10-07/16-13-13/model_latest.pth
```

Budget roughly **20 minutes of H100 time per CO3D object** for Stage A — measured end to
end from the command above: 1201 s for the full 783-iteration schedule. Per-step cost
falls sharply as particles are pruned (≈41 s for the first 600-particle step, 4.1 s/step
by t=122, 1.6 steps/s after t=302), so the early search dominates. Stage B (gs2mesh +
ICP) is CPU-bound and takes longer than Stage A. `scripts/` wraps the common cases.

### Method hyperparameters

All fixed, iteration-indexed, and identical across datasets (full table in
`tools/rebuttal/algorithm/`): **R = 600** initial hypotheses (30 rotations × 20 latents),
**N = 783** steps, pruning **k_t: 600 → 32 (t=3) → 10 (t=122) → 5 (t=302)**,
λ_MSE 1 → 10 (t=302) → 2 (t=732), λ_LPIPS 0.5 → 2 (t=302), λ_prior = 0.001,
learning rates 0.01, **B = 10** particles optimized per GPU step. These live in a
six-stage schedule inside the optimizer that writes into module globals, so they — not
the config values — are the effective weights.

### Things that will bite you

* `general.prefix` is relative to `$INFER3D_RUNS_ROOT`, not the cwd. Hydra `chdir`s into
  its own run directory, so pass **absolute paths** for csv/checkpoint arguments.
* Always set `WANDB_MODE=disabled`; a wandb-0.22 bug crashes the ICP stage.
* `general.total_splits` shards the test set across processes and defaults to `1`. An
  empty shard now raises instead of silently exiting 0.
* `$INFER3D_OURS_REGEN` picks how optimized splats are rebuilt at eval time (`aligned`,
  the default, or `paper`). This moves CO3D OOD PSNR by about 1 dB — see REPRODUCTION.md.
* `torch.load` uses `weights_only=False` for the trusted OmegaConf-bearing checkpoints.

## What is not included

Stated plainly so nothing is assumed present:

* **Model weights and datasets.** All frozen checkpoints (lifters, DiffAE/StyleGAN priors,
  CATSplat, EqM) and all datasets are external — see [DATA.md](DATA.md). The repository
  ships code and per-object result files, not weights.
* **Third-party baseline repositories.** pi-GAN, 3D-GAN-Inversion, nerf-from-image, FINV
  and EqM are used under their own licenses and must be cloned from upstream; only our
  harness code around them is here (`tools/rebuttal/baselines/`).
* **Qualitative figure generation.** The paper's qualitative figures (the viewpoint
  heat-map, the montages, the orbit videos) were produced by separate visualization
  scripts and large rendered media that are not carried over. Everything *quantitative*
  is here and recomputable.
* **Intermediate artifacts for re-running the rebuttal baselines.** Rendered orbits and
  GT point clouds for the 25-car and 20-scene benchmarks are large and live outside the
  repo (`REBUTTAL_ASSETS`); the per-object *scores* they produced do ship, which is what
  `tools/rebuttal/report.py` reads.

## Citation

Please cite the Infer3D paper. This work builds on Splatter Image (Szymanowicz et al.),
3D Gaussian Splatting (Kerbl et al.), Diffusion Autoencoders (Preechakul et al.),
StyleGAN2 (Karras et al.), CATSplat, and — for the rebuttal comparisons — pi-GAN
(Chan et al.), 3D GAN Inversion (Ko et al.), Bootstrapped Radiance Field Inversion
(Pavllo et al.), FINV (Sun et al.) and Equilibrium Matching (Wang and Du).
