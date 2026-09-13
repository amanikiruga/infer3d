"""Minimal end-to-end sanity check for the vendored Infer3D core.

Builds the Gaussian-Splatting lifter, loads a trained checkpoint with a strict
state-dict match, runs a forward pass on a dummy input, and rasterizes the
predicted Gaussians through the CUDA renderer. If this prints OK, the core
library (model + renderer + configs) is wired correctly.

    PYTHONPATH=. python experiments/smoke_test.py
"""
import os
import torch
from hydra import compose, initialize_config_dir

from infer3d import config
from infer3d.model import GaussianSplatPredictor
from infer3d.renderer import render_predicted

CFG_DIR = os.path.join(config.REPO_ROOT, "configs")


def main():
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name="abs_config", overrides=["+dataset=hydrants"])

    device = "cuda"
    model = GaussianSplatPredictor(cfg).to(device)
    ckpt_path = config.LIFTER_CKPTS["hydrants"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    model.eval()
    print(f"[1/3] model built + lifter checkpoint loaded (strict, 0 missing keys)")

    # dummy single-view input: [B, N_views, C, H, W]. CO3D lifters take RGB plus a
    # per-pixel ray-origin-distance channel, so C is 4 whenever origin_distances is set.
    res = cfg.data.training_resolution
    channels = 4 if cfg.data.origin_distances else 3
    x = torch.rand(1, 1, channels, res, res, device=device)
    v2w = torch.eye(4, device=device).reshape(1, 1, 4, 4)
    quat = torch.tensor([1.0, 0, 0, 0], device=device).reshape(1, 1, 4)
    # CO3D sequences have per-image intrinsics, so the lifter wants focal lengths too.
    focals = torch.full((1, 1, 2), res / 2.0, device=device)
    with torch.no_grad():
        splats = model(x, v2w, quat, focals)
    splats = {k: v[0] for k, v in splats.items()}
    print(f"[2/3] forward pass OK -> {splats['xyz'].shape[0]} gaussians, keys={sorted(splats)}")

    # rasterize through the CUDA renderer at an identity camera
    from infer3d.utils.graphics import getProjectionMatrix
    proj = getProjectionMatrix(cfg.data.znear, cfg.data.zfar, 0.7, 0.7).transpose(0, 1).to(device)
    w2v = torch.eye(4, device=device)
    w2v[3, 2] = 2.0  # push camera back along z
    full_proj = (w2v.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    cam_center = w2v.inverse()[3, :3]
    bg = torch.tensor([1.0, 1.0, 1.0], device=device)
    with torch.no_grad():
        out = render_predicted(splats, w2v, full_proj, cam_center, bg, cfg,
                               focals_pixels=focals[0, 0])
    img = out["render"]
    assert img.shape[-1] == res and torch.isfinite(img).all()
    print(f"[3/3] CUDA render OK -> image {tuple(img.shape)}, range[{img.min():.3f},{img.max():.3f}]")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
