"""Free azimuth orbit (36 frames) of OURS on RealCars (DiffAE variant), for the gallery.
Regenerates the exact ours splats from real_ood_realcars/<idx>/final.pth (same as
splats_to_ply_realcars.py) and renders an orbit with render_predicted."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, math, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
SI = f"{_EXT}/splatter-image"
sys.path.insert(0, SI)
os.environ.setdefault("SPLATTER_IMAGE_ROOT", SI)
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from scene.gaussian_predictor import GaussianSplatPredictor
from splatter_image_datasets.srn import SRNDataset
from utils.abs_utils import render_with_custom_camera
dev = "cuda"; NF = 36
OPT_ROOT = f"{SI}/experiments/neurips_submission/real_ood_realcars"
CKPT = f"{SI}/experiments/neurips_submission/real_ood_realcars_eval/checkpoints/model_cars.pth"
OUT = f"{WT}/rebuttal/results/orbits/ours_rc"


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1., 0], [-s, 0, c]], device=dev)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1., 0, 0], [0, c, -s], [0, s, c]], device=dev)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(CKPT):
        alt = glob.glob(f"{SI}/**/model_cars.pth", recursive=True)
        assert alt, "model_cars.pth not found"; ckpt = alt[0]
    else:
        ckpt = CKPT
    with initialize_config_dir(version_base=None, config_dir=f"{SI}/configs"):
        cfg = compose(config_name="abs_config", overrides=["abs=diffae_abs", "+dataset=cars"])
    gp = GaussianSplatPredictor(cfg).to(dev).eval()
    gp.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model_state_dict"])
    train_ds = SRNDataset(cfg, "train", data_category="cars")
    src = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))
    v2w = src["view_to_world_transforms"][:, :cfg.data.input_images, ...].to(dev)
    cv2wT = src["source_cv2wT_quat"][:, :cfg.data.input_images].to(dev)
    bg = torch.tensor([1., 1, 1] if cfg.data.white_background else [0., 0, 0], device=dev)
    zgt = 1.3
    idxs = sorted(p for p in os.listdir(OPT_ROOT) if p.isdigit())
    for idx in idxs:
        if os.path.exists(f"{OUT}/{idx}.npy"): continue
        final = f"{OPT_ROOT}/{idx}/final.pth"
        if not os.path.exists(final): print(f"skip {idx} no final"); continue
        fin = torch.load(final, map_location=dev, weights_only=False)
        best_input = fin["best"]["input"].to(dev).unsqueeze(0)
        with torch.no_grad():
            sp = gp(best_input.unsqueeze(1), v2w, cv2wT, None); sp = {k: v[0] for k, v in sp.items()}
        tilt = rot_x(-0.35)
        frames = []
        for az in np.linspace(0, 2 * np.pi, NF, endpoint=False):
            R = tilt @ rot_y(float(az))
            with torch.no_grad():
                img_r = render_with_custom_camera(sp, bg, cfg, None, R, zgt, device=dev)
            im = img_r[0] if img_r.dim() == 4 else img_r
            frames.append(np.clip(im.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8))
        np.save(f"{OUT}/{idx}.npy", np.stack(frames)); print(f"done {idx}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
