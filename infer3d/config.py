"""Central path and checkpoint configuration for Infer3D.

Every path the code needs lives here, and every value can be overridden with an
environment variable, so nothing in the rest of the repository hardcodes a location.
The defaults point at the layout used to produce the paper numbers; set the variables
(or edit this file, or copy `.env.example` to `.env`) to match your machine.

Nothing here is needed to CHECK the reported numbers -- `scripts/verify.sh` recomputes
every table from per-object result files bundled in the repository. These paths matter
only when re-running experiments. See DATA.md for what each dataset and checkpoint is
and where to obtain it.

Environment variables, grouped as below:

    INFER3D_EXTERN_ROOT    parent directory holding datasets/, diffae/, stylegan3/, ...
    INFER3D_ASSETS_DIR     bundled splits/intrinsics/CSVs           (default ./assets)
    INFER3D_RUNS_ROOT      where runs and eval outputs are written  (default ./runs)

    CO3D_DATASET_ROOT, CO3D_DATASET_ROOT_HQ, SHAPENET_NMR_ROOT,
    SHAPENET_NMR_SE3_ROOT, SHAPENET_CORE_V2, RE10K_CLIPS, REALCARS_ROOT
    DIFFAE_ROOT, STYLEGAN3_ROOT, CATSPLAT_ROOT, CATSPLAT_CKPT
    SPLATTER_REPO_ROOT, SPLATTER_OUT_ROOT
    GS2MESH_ROOT, GS2MESH_INPUT, DINO_PCA_BASIS
    BASELINES_ROOT, REBUTTAL_ASSETS

Only INFER3D_RUNS_ROOT is ever written to; every other root may be read-only.
"""
import os

# --- repository-relative roots ----------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR  = os.getenv("INFER3D_ASSETS_DIR", os.path.join(REPO_ROOT, "assets"))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")   # hydra config root; keep repo-relative

# Where this repository writes its own output. Defaults inside the repo so a fresh
# clone works, but point it at scratch: Stage B's intermediates run to several GB.
RUNS_ROOT   = os.getenv("INFER3D_RUNS_ROOT", os.path.join(REPO_ROOT, "runs"))

# Everything external (datasets, generator repos, frozen weights) hangs off one root.
EXTERN_ROOT = os.getenv("INFER3D_EXTERN_ROOT",
                        "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/akiruga")

# --- datasets ----------------------------------------------------------------
CO3D_DATASET_ROOT     = os.getenv("CO3D_DATASET_ROOT",
                                  f"{EXTERN_ROOT}/datasets/co3d_processed")
# 1080p CO3D, used when a runner is invoked with general.use_hq=true.
CO3D_DATASET_ROOT_HQ  = os.getenv("CO3D_DATASET_ROOT_HQ",
                                  f"{EXTERN_ROOT}/datasets/co3d_processed_1080")
SHAPENET_NMR_ROOT     = os.getenv("SHAPENET_NMR_ROOT",
                                  f"{EXTERN_ROOT}/datasets/srn_nmr_classes")
SHAPENET_NMR_SE3_ROOT = os.getenv("SHAPENET_NMR_SE3_ROOT",
                                  f"{EXTERN_ROOT}/datasets/srn_nmr_se3")
SHAPENET_CORE_V2      = os.getenv("SHAPENET_CORE_V2",
                                  f"{EXTERN_ROOT}/datasets/ShapeNetCore.v2")
RE10K_CLIPS           = os.getenv("RE10K_CLIPS",
                                  f"{EXTERN_ROOT}/datasets/re10k_clips_test_256")
REALCARS_ROOT         = os.getenv("REALCARS_ROOT",
                                  f"{EXTERN_ROOT}/datasets/realcars/HQ339")

# --- external generator repositories -----------------------------------------
# Cloned, not pip-installed: the code imports their model definitions, not just weights.
DIFFAE_ROOT    = os.getenv("DIFFAE_ROOT",    f"{EXTERN_ROOT}/diffae")
STYLEGAN3_ROOT = os.getenv("STYLEGAN3_ROOT", f"{EXTERN_ROOT}/stylegan3")

# --- frozen 3D lifters (Splatter Image) --------------------------------------
# The original splatter-image repo holds the trained lifters under experiments_out/
# and the cached per-object eval CSVs. Read-only is fine.
SPLATTER_REPO_ROOT = os.getenv("SPLATTER_REPO_ROOT", f"{EXTERN_ROOT}/splatter-image")
SPLATTER_OUT_ROOT  = os.getenv("SPLATTER_OUT_ROOT", f"{SPLATTER_REPO_ROOT}/experiments_out")
LIFTER_CKPTS = {
    "hydrants":     f"{SPLATTER_OUT_ROOT}/2025-10-07/16-13-13/model_latest.pth",
    "vases":        f"{SPLATTER_OUT_ROOT}/2026-01-15/14-55-34/model_latest.pth",
    "shapenet_nmr": f"{SPLATTER_OUT_ROOT}/2025-08-06/12-11-04/model_latest.pth",
}

# --- frozen 2D generative priors ---------------------------------------------
DIFFAE_CONF_NAMES = {                       # resolved inside diffae/templates.py
    "hydrants":     "co3d_hydrants_autoenc_128",
    "vases":        "co3d_vases_autoenc_128",
    "shapenet_nmr": "nmr_train_autoenc",
}
STYLEGAN_CKPTS = {
    "hydrants":     f"{STYLEGAN3_ROOT}/training-runs/"
                    f"00020-stylegan2-co3d_hydrant_for_stylegan128-gpus2-batch32-gamma0.5/"
                    f"network-snapshot-006600.pkl",
    "shapenet_nmr": f"{STYLEGAN3_ROOT}/training-runs/"
                    f"00004-stylegan2-nmr128-gpus8-batch256-gamma10/"
                    f"network-snapshot-020889.pkl",
}

# --- Table 3: RE10K sensor / FOV shift ---------------------------------------
# The RE10K lifter is CATSplat (frozen). Per-sequence test clips are built once by
# tools/re10k/build_clips.py.
CATSPLAT_ROOT = os.getenv("CATSPLAT_ROOT", f"{EXTERN_ROOT}/iccv25_CATSplat")
CATSPLAT_CKPT = os.getenv("CATSPLAT_CKPT", f"{CATSPLAT_ROOT}/checkpoints/CATSplat.pth")

# The locked OOD-fisheye operating point. Severity was calibrated so that the baseline
# rows reproduce the published ones, because the paper's own fisheye synthesis code was
# unavailable -- so this is a reconstruction of the paper's OOD condition, not the
# condition itself. Protocol and sweeps: tools/re10k/README.md.
#   in_fov       field of view of the source perspective image, degrees
#   out_fov      diameter field of view of the synthesized fisheye, degrees
#   k_scale      scale on the theta-polynomial radial distortion profile
#   circle_scale image-circle scale
#   fxf_nominal  nominal fisheye focal in pixels, (S/2) / (radians(out_fov)/2) * circle_scale
RE10K_FISHEYE = dict(in_fov=75.0, out_fov=94.0, k_scale=0.5,
                     circle_scale=1.22, fxf_nominal=190.4)

# --- Table 5: RealCars (in-the-wild ARKit captures) --------------------------
# The lifter is the official Splatter-Image SRN-Cars model; pseudo-ground-truth is a
# FastGS fit to each scene's full multi-view capture, meshed through gs2mesh.
GS2MESH_ROOT   = os.getenv("GS2MESH_ROOT",  f"{EXTERN_ROOT}/gs2mesh")
GS2MESH_INPUT  = os.getenv("GS2MESH_INPUT", f"{GS2MESH_ROOT}/input")
DINO_PCA_BASIS = os.getenv("DINO_PCA_BASIS",
                           f"{EXTERN_ROOT}/datasets/srn_cars/cars_extras/dino_pca_basis.pt")

# --- rebuttal: optimization-baseline comparisons ------------------------------
# Upstream clones and released checkpoints for the four inversion baselines (pi-GAN,
# 3D-GAN-Inversion/EG3D+PTI, nerf-from-image/BRFI, FINV) plus EqM, and the large
# per-object intermediates their harnesses read. Needed only to RE-RUN
# tools/rebuttal/baselines; the recorded per-object scores ship in the repository.
BASELINES_ROOT  = os.getenv("BASELINES_ROOT",  f"{EXTERN_ROOT}/rebuttal_infer3d_baselines")
REBUTTAL_ASSETS = os.getenv("REBUTTAL_ASSETS", f"{EXTERN_ROOT}/splatter-image-rebuttal/rebuttal")

# --- bundled assets (ship with the repository) -------------------------------
NMR_SPLITS_JSON     = os.path.join(ASSETS_DIR, "nmr_splits.json")
NMR_TEST_SPLIT_JSON = os.path.join(ASSETS_DIR, "nmr_test_split.json")
NMR_CLASS_INFO_JSON = os.path.join(ASSETS_DIR, "object_ids_per_category_nmr.json")
CO3D_TEST_PATHS_CSV = os.path.join(ASSETS_DIR, "co3d_test_paths_1080.csv")        # hydrants
CO3D_VASES_TEST_CSV = os.path.join(ASSETS_DIR, "co3d_vases_test_paths_1080.csv")
EVAL_CSVS_DIR       = os.path.join(ASSETS_DIR, "eval_csvs")   # per-object results for make_tables


def nmr_intrins_json(view):
    """Per-view ShapeNet-NMR intrinsics, e.g. view='side-top' / 'so3' / 'se3' / 'test'."""
    return os.path.join(ASSETS_DIR, f"shapenet_nmr_intrins_{view}.json")
