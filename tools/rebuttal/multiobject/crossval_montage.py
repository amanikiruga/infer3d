"""Visual confirmation of the cross-validated registration: fit per-object 7-DoF on EVEN
views, render HELD-OUT (odd) views, save GT|ours strips. If these are real two-chair
scenes (not blobs), the placement result is trustworthy."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import sys, numpy as np
import torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, f"{_EXT}/splatter-image-rebuttal/rebuttal/multiobject")
import placement_crossval as C
from hydra import initialize_config_dir, compose
import hydra
from copy import deepcopy
import multichair_stage3_gt_compare as G
from gaussian_renderer import render_predicted

device = "cuda"
SCENES = ["0008_c92721a95fe44b018039b09dacd0f1a7", "0011_cbc76d55a04d5b2e1d9a8cea064f5297",
          "0009_c98b7e1952c2c7bb85f6153ed2033a1c", "0006_bf9ea87a9765eed7d53b73fe621a84b4"]
SHOW = [3, 9, 15]   # held-out (odd) views

hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize_config_dir(version_base=None, config_dir=f"{C.ROOT}/configs")
cfg = compose(config_name="abs_config", overrides=[
    "abs=diffae_abs", "+dataset=chairs",
    f"opt.pretrained_ckpt={C.ROOT}/checkpoints/model_chairs.pth",
    "general.data_example_ids_path=not_needed.json", "general.prefix=cvmont"])
cfgr = deepcopy(cfg); cfgr.data.fov = G.CHAIRS_FOV_DEG; cfgr.data.znear = G.CHAIRS_ZNEAR; cfgr.data.zfar = G.CHAIRS_ZFAR
gp = G.load_splatter_image(cfg)
bg = torch.tensor([1., 1, 1], device=device)

rows = []
for scene in SCENES:
    b = torch.load(f"{C.SRC}/stage1/{scene}/bundle.pth", map_location="cpu", weights_only=False)
    pa = torch.load(f"{C.SRC}/stage2/{scene}/chair0/final.pth", map_location="cpu", weights_only=False)
    pb = torch.load(f"{C.SRC}/stage2/{scene}/chair1/final.pth", map_location="cpu", weights_only=False)
    sa = {k: v.detach() for k, v in G.reconstruct_chair_splats(pa, gp, cfg).items()}
    sb = {k: v.detach() for k, v in G.reconstruct_chair_splats(pb, gp, cfg).items()}
    gt = b["gt_images"].to(device)
    if gt.shape[-2:] != (128, 128):
        gt = F.interpolate(gt, size=(128, 128), mode="bilinear", align_corners=False)
    cz = 0.5 * (float(sa["xyz"][:, 2].mean()) + float(sb["xyz"][:, 2].mean()))
    wvts, fpts, ccs = C.build_cams(b, cz / 7.0)
    fit_idx = list(range(1, gt.shape[0]))[0::2]
    params = {n: {"log_s": torch.zeros((), device=device, requires_grad=True),
                  "six": torch.tensor([1., 0, 0, 0, 1, 0], device=device, requires_grad=True),
                  "t": torch.zeros(3, device=device, requires_grad=True)} for n in "ab"}
    opt = torch.optim.Adam([{"params": [params["a"]["six"], params["b"]["six"]], "lr": 0.01},
                            {"params": [params["a"]["log_s"], params["a"]["t"], params["b"]["log_s"], params["b"]["t"]], "lr": 0.005}])
    def comp():
        ta = C.transform_splats(sa, torch.exp(params["a"]["log_s"]), C.sixd_to_R(params["a"]["six"]), params["a"]["t"])
        tb = C.transform_splats(sb, torch.exp(params["b"]["log_s"]), C.sixd_to_R(params["b"]["six"]), params["b"]["t"])
        return G.merge(ta, tb)
    def rnd(sp, i):
        return render_predicted(sp, wvts[i:i+1], fpts[i:i+1], ccs[i:i+1], bg, cfgr, focals_pixels=None)["render"].clamp(0, 1)
    for it in range(400):
        opt.zero_grad()
        pred = torch.stack([rnd(comp(), i) for i in fit_idx])
        (((pred - gt[fit_idx]) ** 2).mean()).backward()
        opt.step()
    with torch.no_grad():
        sp = comp()
        for i in SHOW:  # held-out
            g = (gt[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            o = (rnd(sp, i).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            rows.append(np.concatenate([g, o], axis=1))
out = f"{C.OUT}/crossval_heldout_montage.png"
Image.fromarray(np.concatenate(rows, axis=0)).save(out)
print("saved", out, "(rows = scenes x held-out views; each = GT | ours-registered)")
