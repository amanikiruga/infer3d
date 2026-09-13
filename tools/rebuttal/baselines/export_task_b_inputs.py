"""Export Task-B (RealCars) inputs for baselines from ours' cached inputs.pth.

For each of the 20 test frames: the SAM2-masked white-bg 128x128 crop (rgb_128,
what ours' pixel losses saw), the original-background crop (rgb_orig_128, for
methods with their own real-image machinery), and the mask. Also copies the raw
frame path list. Output: rebuttal/results/task_b_inputs/{idx}/
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os
import sys

import numpy as np
import torch
from PIL import Image

MAIN = f"{_EXT}/splatter-image"
WT = f"{_EXT}/splatter-image-rebuttal"
SRC = f"{MAIN}/experiments/neurips_submission/real_ood_realcars"
OUT = f"{WT}/rebuttal/results/task_b_inputs"


def to_png(t):  # [1,3,H,W] or [1,1,H,W] float
    a = t[0].permute(1, 2, 0).numpy()
    if a.shape[2] == 1:
        a = a[:, :, 0]
    return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))


frame_list = open(f"{MAIN}/experiments/neurips_submission/real_ood/realcars_test_paths.csv").read().split()
for idx in sorted(os.listdir(SRC)):
    p = f"{SRC}/{idx}/inputs_cache/inputs.pth"
    if not (idx.isdigit() and os.path.exists(p)):
        continue
    b = torch.load(p, map_location="cpu", weights_only=False)
    d = f"{OUT}/{idx}"
    os.makedirs(d, exist_ok=True)
    to_png(b["rgb_128"]).save(f"{d}/rgb_white_128.png")
    to_png(b["rgb_orig_128"]).save(f"{d}/rgb_orig_128.png")
    to_png(b["mask_128"]).save(f"{d}/mask_128.png")
    with open(f"{d}/frame_path.txt", "w") as f:
        f.write(frame_list[int(idx)])
    print(idx, "ok")
print("DONE")
