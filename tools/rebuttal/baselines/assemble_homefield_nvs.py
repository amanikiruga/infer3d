"""Homefield NVS videos (in-distribution / control) from ACTUAL SCORED renders.
Frame i = control GT target camera i. Panels: input | ours | <baseline> | GT.
ours renders: task_a_runs/ours_control_nvs/<oid>/{i}.png
eg3d renders: task_a_runs/eg3d_control_zup/<oid>/renders/{i}.png
finv renders: task_a_runs/finv_control_render/<oid>/renders/{i}.png  (if present)
GT: task_a_inputs_control/<oid>/targets/<sorted>[i]
Writes report/videos/homefield_nvs/<baseline>_<oid>.mp4
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
WT = f"{_EXT}/splatter-image-rebuttal"
R = f"{WT}/rebuttal/results"
OUT = f"{WT}/rebuttal/report/videos/homefield_nvs"; os.makedirs(OUT, exist_ok=True)
INP = f"{R}/task_a_inputs_control"

BASE = {
 "eg3d": ("EG3D+PTI", "task_a_runs/eg3d_control_zup/{oid}/renders", (20, 60, 200)),
 "finv": ("FINV-SV", "task_a_runs/finv_control_render/{oid}/renders", (20, 60, 200)),
}


def lab(arr, txt, col):
    im = Image.fromarray(np.asarray(Image.fromarray(arr.astype("uint8")).convert("RGB").resize((128, 128))))
    ImageDraw.Draw(im).text((3, 2), txt, fill=col); return np.asarray(im)


def load(p):
    return np.asarray(Image.open(p).convert("RGB")) if os.path.exists(p) else np.full((128, 128, 3), 245, np.uint8)


def main():
    made = {}
    for oid in sorted(os.listdir(INP)):
        gtdir = f"{INP}/{oid}/targets"
        if not os.path.isdir(gtdir):
            continue
        gtfs = sorted(os.listdir(gtdir)); N = len(gtfs)
        inp = load(f"{INP}/{oid}/input.png")
        ours_dir = f"{R}/task_a_runs/ours_control_nvs/{oid}"
        for key, (name, tmpl, col) in BASE.items():
            bdir = tmpl.format(oid=oid)
            if not os.path.isdir(f"{R}/{bdir}"):
                continue
            gap = np.full((128, 4, 3), 255, np.uint8)
            frames = []
            for i in range(N):
                cells = [lab(inp, "input", (20, 20, 20)),
                         lab(load(f"{ours_dir}/{i:02d}.png"), "OURS", (200, 20, 20)),
                         lab(load(f"{R}/{bdir}/{i:02d}.png"), name, col),
                         lab(load(f"{gtdir}/{gtfs[i]}"), "GT", (20, 120, 20))]
                row = []
                for c in cells:
                    row += [c, gap]
                frames.append(np.concatenate(row[:-1], 1))
            imageio.mimsave(f"{OUT}/{key}_{oid}.mp4", frames, fps=6, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p"])
            made[key] = made.get(key, 0) + 1
    print("homefield NVS videos:", made)


if __name__ == "__main__":
    main()
