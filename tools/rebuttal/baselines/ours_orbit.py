"""Free azimuth orbit (36 frames) of OURS reconstruction (SO3 task) for the gallery.
Spins the canonical reconstruction with render_with_custom_camera (fixed camera,
object rotated) — the exact eval render path, guaranteed-correct convention."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, glob, math, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, WT)
os.environ.setdefault("SPLATTER_IMAGE_ROOT", WT)
os.environ.setdefault("SHAPENET_NMR_ROOT", f"{_EXT}/datasets/srn_nmr_classes")
from hydra import compose, initialize_config_dir
from scene.gaussian_predictor import GaussianSplatPredictor
from utils.abs_utils import render_with_custom_camera
dev = "cuda"; NF = 36
LIFTER = f"{_EXT}/splatter-image/experiments_out/2025-08-06/12-11-04/model_latest.pth"


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1., 0], [-s, 0, c]], device=dev)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1., 0, 0], [0, c, -s], [0, s, c]], device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{WT}/rebuttal/results/task_a_runs/ours")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/orbits/ours_so3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=f"{WT}/configs"):
        cfg = compose(config_name="abs_config", overrides=["+dataset=shapenet-nmr"])
    gp = GaussianSplatPredictor(cfg).to(dev).eval()
    gp.load_state_dict(torch.load(LIFTER, map_location=dev, weights_only=False)["model_state_dict"])
    v2w = torch.eye(4, device=dev).reshape(1, 1, 4, 4); quat = torch.tensor([1., 0, 0, 0], device=dev).reshape(1, 1, 4)
    bg = torch.tensor([1., 1, 1], device=dev)
    tilt = rot_x(-0.35)
    for d in sorted(glob.glob(f"{args.runs}/*/topk_best_everything_latest.pth")):
        oid = os.path.basename(os.path.dirname(d)).split("-")[0]
        if os.path.exists(f"{args.out}/{oid}.npy"): continue
        c = torch.load(d, map_location=dev, weights_only=False)
        img = c["best_input_image"].to(dev)
        if img.dim() == 4: img = img.unsqueeze(1)
        zgt = float(c.get("zgt", 1.3))
        with torch.no_grad():
            sp = gp(img, v2w, quat, None); sp = {k: v[0] for k, v in sp.items()}
        frames = []
        for az in np.linspace(0, 2 * np.pi, NF, endpoint=False):
            R = (tilt @ rot_y(float(az)))
            with torch.no_grad():
                img_r = render_with_custom_camera(sp, bg, cfg, None, R, zgt, device=dev)
            im = img_r[0] if img_r.dim() == 4 else img_r
            frames.append(np.clip(im.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8))
        np.save(f"{args.out}/{oid}.npy", np.stack(frames)); print(f"done {oid}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
