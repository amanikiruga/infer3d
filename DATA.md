# Datasets and checkpoints

None of this is needed to **check** the reported numbers — `scripts/verify.sh` recomputes
every table from per-object result files bundled in the repository. This page is for
re-running the experiments.

Everything is located through [`infer3d/config.py`](infer3d/config.py); each row below
names the environment variable that overrides it.

## Datasets

| Dataset | Used by | Env var | Source |
|---|---|---|---|
| **CO3D** (Hydrants, Vases), preprocessed to per-sequence Gaussian-Splatting format | Tables 1, 4, 6, 7, 8, 9 | `CO3D_DATASET_ROOT` | [CO3D](https://github.com/facebookresearch/co3d), then the Splatter-Image preprocessing |
| **ShapeNet-NMR**, SRN-rendered, 11 categories | Table 2 | `SHAPENET_NMR_ROOT` | [NMR renders](https://github.com/autonomousvision/occupancy_networks) / SRN protocol |
| **ShapeNet-NMR SE(3)** — the same objects re-rendered under SE(3)-perturbed cameras | Table 2 OOD SE(3) | `SHAPENET_NMR_SE3_ROOT` | rendered by us with the SRN camera conventions |
| **ShapeNetCore.v2** raw meshes | Chamfer ground truth for the rebuttal's Task A | `SHAPENET_CORE_V2` | [ShapeNet](https://shapenet.org) (registration required) |
| **RealEstate10K**, per-sequence test clips at 256px | Table 3 | `RE10K_CLIPS` | [RealEstate10K](https://google.github.io/realestate10k/); build clips with `tools/re10k/build_clips.py` |
| **RealCars** — 20 ARKit captures with LiDAR poses | Table 5 | `REALCARS_ROOT` | captured by us |
| **Objaverse** subset, 25 partially-occluded objects, 6 categories | rebuttal broad-prior | — (ids in `tools/rebuttal/broad_prior/results/cd_sel25.json`) | [Objaverse](https://objaverse.allenai.org) |

The two ShapeNet-NMR splits that the code reads (`assets/nmr_splits.json`,
`assets/nmr_test_split.json`), the per-view intrinsics
(`assets/shapenet_nmr_intrins_*.json`) and the CO3D test-path CSVs
(`assets/co3d_{,vases_}test_paths_1080.csv`) ship with the repository, so the exact
object and view selection is fixed and does not have to be reconstructed.

## Frozen models

Infer3D trains nothing at test time. It needs one **2D generative prior** and one
**3D lifter** per setting, both frozen.

### 3D lifters — Splatter Image (`SPLATTER_REPO_ROOT`)

| Role | File | Size |
|---|---|---|
| CO3D Hydrants (Table 1) | `experiments_out/2025-10-07/16-13-13/model_latest.pth` | 729 MB |
| CO3D Vases (Table 1) | `experiments_out/2026-01-15/14-55-34/model_latest.pth` | 778 MB |
| ShapeNet-NMR, all 11 categories (Table 2) | `experiments_out/2025-08-06/12-11-04/model_latest.pth` | 779 MB |
| SRN Cars (Table 5 / RealCars) | official Splatter-Image release | — |
| RE10K (Table 3) | **CATSplat** — `CATSPLAT_CKPT` | — |
| Objaverse (rebuttal broad-prior) | Objaverse Splatter Image | — |

Paths are set in `config.LIFTER_CKPTS`. A single ShapeNet-NMR lifter covers all 11
categories; the per-category CO3D lifters reflect the available pretrained assets, not a
requirement of the method.

### 2D priors

| Prior | Config | Checkpoint |
|---|---|---|
| DiffAE, CO3D Hydrants | `co3d_hydrants_autoenc_128` | `$DIFFAE_ROOT/checkpoints/<conf>/last.ckpt` (2.4 GB) |
| DiffAE, CO3D Vases | `co3d_vases_autoenc_128` | as above |
| DiffAE, ShapeNet-NMR | `nmr_train_autoenc` | as above |
| DiffAE, RE10K | `re10k_autoenc_256` | as above |
| StyleGAN2, CO3D Hydrants | — | `$STYLEGAN3_ROOT/training-runs/00020-.../network-snapshot-006600.pkl` (349 MB) |
| StyleGAN2, ShapeNet-NMR (class-conditional, 11 classes) | — | `$STYLEGAN3_ROOT/training-runs/00004-.../network-snapshot-020889.pkl` (360 MB) |
| Equilibrium Matching, ImageNet (rebuttal) | released EqM checkpoint | `$BASELINES_ROOT/eqm/EqM/pretrained_models/` |

Names are in `config.DIFFAE_CONF_NAMES` and `config.STYLEGAN_CKPTS`. DiffAE and
StyleGAN3 are external repositories: clone them and point `DIFFAE_ROOT` / `STYLEGAN3_ROOT`
at the clone, since the code imports their model definitions, not just their weights.

### Rebuttal baselines (`BASELINES_ROOT`)

Only needed to re-run `tools/rebuttal/baselines/`. Clone each at its released commit with
its released weights: `nerf-from-image/` (BRFI), `3D-GAN-Inversion/` (EG3D+PTI),
`pi-GAN/`, `FINV/`, `eqm/EqM/`. They are third-party, under their own licenses, and are
not redistributed here.

## Disk budget

Roughly 25 GB of frozen model weights, plus the datasets (CO3D and ShapeNet-NMR dominate;
tens to hundreds of GB depending on how much of each you fetch). Stage A writes about
150–350 MB of optimization checkpoints per run directory, and Stage B's gs2mesh
intermediates are several GB per evaluated set — point `INFER3D_RUNS_ROOT` at scratch.
