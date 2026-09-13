"""Homefield comparison videos from the ACTUAL SCORED RENDERS (frame i = target camera i,
same protocol as assemble_nvs.py). One group per method, in its own generator domain:
  eg3d : input | EG3D+PTI own-pose | EG3D+PTI oracle-pose | GT (generator render)
  finv : input | FINV-SV | GT
  pigan: input | pi-GAN | GT
Writes report/videos/homefield/<method>_<oid>.mp4
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
WT = f"{_EXT}/splatter-image-rebuttal"
HF = f"{WT}/rebuttal/results/homefield"
OUT = f"{WT}/rebuttal/report/videos/homefield"; os.makedirs(OUT, exist_ok=True)

GROUPS = {
 "eg3d": ("inputs_eg3d", [("EG3D+PTI own", "eg3d", (20, 60, 200)),
                          ("EG3D+PTI oracle", "eg3d_oracle", (150, 90, 20))]),
 "finv": ("inputs_eg3d", [("FINV-SV", "finv", (20, 60, 200))]),
 "pigan": ("inputs_pigan", [("pi-GAN", "pigan", (20, 60, 200))]),
}


def lab(arr, txt, col):
    im = Image.fromarray(arr.astype("uint8")).convert("RGB").resize((128, 128))
    ImageDraw.Draw(im).text((3, 2), txt, fill=col); return np.asarray(im)


def main():
    made = {}
    for key, (inp_dir, methods) in GROUPS.items():
        base = f"{HF}/{inp_dir}"
        if not os.path.isdir(base):
            print(f"skip group {key} (no {inp_dir})"); continue
        for oid in sorted(os.listdir(base)):
            if not all(os.path.exists(f"{HF}/{m[1]}/{oid}/meta.json") for m in methods):
                continue
            inp = np.asarray(Image.open(f"{base}/{oid}/input.png").convert("RGB"))
            n = len(os.listdir(f"{base}/{oid}/targets"))
            gap = np.full((128, 4, 3), 255, np.uint8)
            frames = []
            for i in range(n):
                cells = [lab(inp, "input", (20, 20, 20))]
                for nm, d, col in methods:
                    cells.append(lab(np.asarray(Image.open(
                        f"{HF}/{d}/{oid}/renders/{i:02d}.png").convert("RGB")), nm, col))
                cells.append(lab(np.asarray(Image.open(
                    f"{base}/{oid}/targets/{i:02d}.png").convert("RGB")), "GT (gen)", (20, 120, 20)))
                row = []
                for c in cells:
                    row += [c, gap]
                frames.append(np.concatenate(row[:-1], 1))
            imageio.mimsave(f"{OUT}/{key}_{oid}.mp4", frames, fps=6, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p"])
            made[key] = made.get(key, 0) + 1
    print("homefield videos:", made)


if __name__ == "__main__":
    main()
