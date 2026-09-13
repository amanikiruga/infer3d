"""Ours (Infer3D) geometry for Task A: lift each run's best decoded image to
Gaussians, take opacity-filtered centers as the predicted point cloud. ICP-Chamfer
vs GT scored by icp_chamfer.py (same as all baselines)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, glob, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, WT)
os.environ.setdefault("SPLATTER_IMAGE_ROOT", WT)
os.environ.setdefault("SHAPENET_NMR_ROOT", f"{_EXT}/datasets/srn_nmr_classes")
from hydra import compose, initialize_config_dir
from scene.gaussian_predictor import GaussianSplatPredictor
dev = "cuda"
LIFTER = f"{_EXT}/splatter-image/experiments_out/2025-08-06/12-11-04/model_latest.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{WT}/rebuttal/results/task_a_runs/ours")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_geom/ours")
    ap.add_argument("--n_surface", type=int, default=50000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with initialize_config_dir(version_base=None, config_dir=f"{WT}/configs"):
        cfg = compose(config_name="abs_config", overrides=["+dataset=shapenet-nmr"])
    gp = GaussianSplatPredictor(cfg).to(dev).eval()
    ck = torch.load(LIFTER, map_location=dev, weights_only=False)
    gp.load_state_dict(ck["model_state_dict"])
    v2w = torch.eye(4, device=dev).reshape(1, 1, 4, 4)
    quat = torch.tensor([1., 0, 0, 0], device=dev).reshape(1, 1, 4)

    for d in sorted(glob.glob(f"{args.runs}/*/topk_best_everything_latest.pth")):
        oid = os.path.basename(os.path.dirname(d)).split("-")[0]
        if os.path.exists(f"{args.out}/{oid}.npy"):
            continue
        c = torch.load(d, map_location=dev, weights_only=False)
        img = c["best_input_image"].to(dev)          # (1,3,128,128) generator-decoded ID image
        if img.dim() == 4:
            img = img.unsqueeze(1)                    # (1,1,3,H,W)
        with torch.no_grad():
            sp = gp(img, v2w, quat, None)
        sp = {k: v[0] for k, v in sp.items()}
        xyz = sp["xyz"].detach().cpu().numpy()
        op = torch.sigmoid(sp["opacity"][:, 0]).detach().cpu().numpy() if sp["opacity"].dim() > 1 else torch.sigmoid(sp["opacity"]).detach().cpu().numpy()
        keep = op > 0.05
        xyz = xyz[keep] if keep.sum() > 100 else xyz
        if len(xyz) > args.n_surface:
            idx = np.random.RandomState(0).choice(len(xyz), args.n_surface, replace=False)
            xyz = xyz[idx]
        np.save(f"{args.out}/{oid}.npy", xyz.astype(np.float32))
        print(f"done {oid} ({len(xyz)} pts)")
    print("ALL DONE")


if __name__ == "__main__":
    main()
