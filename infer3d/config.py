"""Every path the code needs, in one place. Each is overridable by an environment
variable of the same name. Copy `.env.example` to `.env`, edit, and `source .env`.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(REPO_ROOT, "assets")
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")

# Everything this repo writes. Must be writable; meshing intermediates are large.
RUNS_ROOT = os.getenv("INFER3D_RUNS_ROOT", os.path.join(REPO_ROOT, "runs"))

# Parent of the downloaded datasets and models; the defaults below derive from it.
EXTERN_ROOT = os.getenv("INFER3D_EXTERN_ROOT", os.path.expanduser("~/infer3d-data"))

# --- datasets ---------------------------------------------------------------
CO3D_DATASET_ROOT = os.getenv("CO3D_DATASET_ROOT", f"{EXTERN_ROOT}/co3d_processed")
CO3D_DATASET_ROOT_HQ = os.getenv("CO3D_DATASET_ROOT_HQ", f"{EXTERN_ROOT}/co3d_processed_1080")
RE10K_CLIPS = os.getenv("RE10K_CLIPS", f"{EXTERN_ROOT}/re10k_clips_test_256")
REALCARS_ROOT = os.getenv("REALCARS_ROOT", f"{EXTERN_ROOT}/realcars")
REALCARS_PSEUDO_GT = os.getenv("REALCARS_PSEUDO_GT", f"{EXTERN_ROOT}/realcars_pseudo_gt")

# --- frozen 2D prior (Diffusion Autoencoder) --------------------------------
# A clone of the diffae repo with checkpoints/ inside it: we import its model
# definitions, so a bare weights file is not enough.
DIFFAE_ROOT = os.getenv("DIFFAE_ROOT", f"{EXTERN_ROOT}/diffae")

# --- frozen 3D lifters ------------------------------------------------------
# Splatter Image for CO3D and RealCars; CATSplat for RE10K.
LIFTER_DIR = os.getenv("LIFTER_DIR", f"{EXTERN_ROOT}/lifters")
LIFTER_CKPTS = {
    "hydrants": f"{LIFTER_DIR}/co3d_hydrants.pth",
    "vases": f"{LIFTER_DIR}/co3d_vases.pth",
    "cars": f"{LIFTER_DIR}/srn_cars.pth",
}
CATSPLAT_ROOT = os.getenv("CATSPLAT_ROOT", f"{EXTERN_ROOT}/CATSplat")
CATSPLAT_CKPT = os.getenv("CATSPLAT_CKPT", f"{CATSPLAT_ROOT}/checkpoints/CATSplat.pth")

# --- meshing (Chamfer distance for CO3D and RealCars) -----------------------
GS2MESH_ROOT = os.getenv("GS2MESH_ROOT", f"{EXTERN_ROOT}/gs2mesh")
GS2MESH_INPUT = os.getenv("GS2MESH_INPUT", f"{GS2MESH_ROOT}/input")

# --- RE10K fisheye operating point ------------------------------------------
# Degrees for the two FOVs; k_scale scales the theta-polynomial radial profile;
# circle_scale is the image-circle scale. Fixed for every RE10K result.
RE10K_FISHEYE = dict(in_fov=75.0, out_fov=94.0, k_scale=0.5,
                     circle_scale=1.22, fxf_nominal=190.4)

# --- bundled splits ---------------------------------------------------------
CO3D_TEST_PATHS_CSV = os.path.join(ASSETS_DIR, "co3d_test_paths_1080.csv")
CO3D_VASES_TEST_CSV = os.path.join(ASSETS_DIR, "co3d_vases_test_paths_1080.csv")
REALCARS_TEST_CSV = os.path.join(ASSETS_DIR, "realcars_test_paths.csv")
