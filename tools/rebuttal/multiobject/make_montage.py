"""Three-way qualitative montage for the compositionality response:
GT | feed-forward (depth-anchored, privileged) | Infer3D-composed (ours)
at novel views 1/10/19 for a few scenes. Ours frames come from the existing
stage3 gt_compare.mp4 (frame v = view v, right half, below the 32px title bar)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import sys, numpy as np
import imageio.v2 as imageio
import torch
from PIL import Image

sys.path.insert(0, f"{_EXT}/splatter-image-rebuttal/rebuttal/multiobject")
from baseline_multichair import (baseline_splats, depth_anchor_splats, render_at_views,
                                 ROOT, SRC, OUT, device)
from hydra import initialize_config_dir, compose
import hydra as _h
from scene.gaussian_predictor import GaussianSplatPredictor

SCENES = ["0000_b8f2712e8330ba6b3c9fe3a963c6d73b", "0004_be745a383ceccfe453fa79783efbc3bf",
          "0011_cbc76d55a04d5b2e1d9a8cea064f5297"]
VIEWS = [1, 10, 19]

_h.core.global_hydra.GlobalHydra.instance().clear()
initialize_config_dir(version_base=None, config_dir=f"{ROOT}/configs")
cfg = compose(config_name="abs_config", overrides=[
    "abs=diffae_abs", "+dataset=chairs",
    f"opt.pretrained_ckpt={ROOT}/checkpoints/model_chairs.pth",
    "general.data_example_ids_path=not_needed.json", "general.prefix=rebuttal-mc"])
gp = GaussianSplatPredictor(cfg).to(memory_format=torch.channels_last).to(device)
gp.load_state_dict(torch.load(cfg.opt.pretrained_ckpt, map_location=device, weights_only=False)["model_state_dict"])
gp.eval()
bg = torch.tensor([1., 1, 1], device=device)

rows = []
for scene in SCENES:
    bundle = torch.load(f"{SRC}/stage1/{scene}/bundle.pth", map_location="cpu", weights_only=False)
    bs = baseline_splats(bundle["rgb_t"].clone(), bundle, gp, cfg)
    asplats, _ = depth_anchor_splats(bs, bundle, cfg)
    renders_a = render_at_views(asplats, bundle, cfg, bg)
    mp4 = imageio.get_reader(f"{SRC}/stage3/{scene}/gt_compare.mp4")
    for v in VIEWS:
        gt = (bundle["gt_images"][v].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        ff = (renders_a[v].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        fr = mp4.get_data(v)
        ours = fr[32:, 128:]  # right panel below title bar
        if ours.shape[:2] != gt.shape[:2]:
            ours = np.array(Image.fromarray(ours).resize((gt.shape[1], gt.shape[0])))
        rows.append(np.concatenate([gt, ff, ours], axis=1))

Image.fromarray(np.concatenate(rows, axis=0)).save(f"{OUT}/montage_gt_ff_ours.png")
print("saved", f"{OUT}/montage_gt_ff_ours.png  [GT | anchored feed-forward | ours], views", VIEWS)
