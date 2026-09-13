"""Diagnostic: is the turntable/scale path correct? Feed a REAL in-distribution objaverse
test object (view-0) through lift + my build_turntable, compare against rendering at the
dataset's OWN cameras. If the dataset-cam render is coherent but my turntable collapses ->
my camera/scale is wrong. If both coherent -> the m1 car collapse is EqM-input style-OOD."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
from splatter_image_datasets.objaverse import ObjaverseDataset

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/m1"
cfg = P.objaverse_cfg()
gp = P.load_lifter(cfg)
ds = ObjaverseDataset(cfg, "test")
cams = P.build_turntable(cfg, num=60)

for idx in [0, 5]:
    item = ds[idx]
    oid = item.get("object_id", str(idx))
    x = item["gt_images"][0].unsqueeze(0).unsqueeze(0).to(P.DEV)
    v2w = item["view_to_world_transforms"][0].unsqueeze(0).unsqueeze(0).to(P.DEV)
    quat = item["source_cv2wT_quat"][0].unsqueeze(0).unsqueeze(0).to(P.DEV)
    with torch.no_grad():
        sp = gp(x, v2w, quat, None); sp = {k: v[0] for k, v in sp.items()}
    # (a) render at dataset's own novel cameras
    ds_frames = []
    for v in range(item["world_view_transforms"].shape[0]):
        with torch.no_grad():
            r = P.render_view(cfg, sp, item["world_view_transforms"][v].to(P.DEV),
                              item["full_proj_transforms"][v].to(P.DEV), item["camera_centers"][v].to(P.DEV))
        ds_frames.append((r.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
    # (b) render at MY turntable cameras (frame0 should == dataset source view)
    tt = P.turntable_frames(cfg, sp, cams, stride=6)
    src_in = (item["gt_images"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ds_strip = np.concatenate([src_in] + ds_frames[:5], axis=1)
    tt_strip = np.concatenate(tt[:6], axis=1)
    Image.fromarray(np.concatenate([ds_strip, tt_strip], axis=0)).save(f"{OUT}/diag_{idx}_{oid}.png")
    print(f"idx{idx} {oid}: saved diag (top=input+dataset-cams, bottom=my-turntable)")
print("DONE")
