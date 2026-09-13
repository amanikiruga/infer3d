"""Per-baseline comparison videos (user layout): each baseline in its OWN group, panels
= input | OURS | <baseline> | GT, per object. Per-object SO(3) NVS PSNR burned into each
method panel. ours+GT repeated in every group. Writes report/videos/by_method/<B>_<task>_<oid>.mp4
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
WT = f"{_EXT}/splatter-image-rebuttal"
R = f"{WT}/rebuttal/results"; NF = 24
SC = json.load(open(f"{R}/per_object_scores.json")) if os.path.exists(f"{R}/per_object_scores.json") else {}
BASELINES = [("nfi", "NFI (nerf-from-image)", "nfi_own"),
             ("eg3d", "3D-GAN-Inv (EG3D+PTI)", "eg3d_own"),
             ("pigan", "pi-GAN", None),
             ("finv", "FINV-SV", "finv_own")]


def orbit(path):
    if not os.path.exists(path):
        return None
    a = np.load(path); idx = np.linspace(0, len(a) - 1, NF).round().astype(int); return a[idx]


def gt_orbit(oid):
    d = f"{R}/task_a_inputs/{oid}/targets"
    if not os.path.isdir(d):
        return None
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    a = np.stack([np.asarray(Image.open(f"{d}/{x}").convert("RGB")) for x in fs])
    return a[np.linspace(0, len(a) - 1, NF).round().astype(int)]


def lab(img, txt, col):
    im = Image.fromarray(np.asarray(Image.fromarray(img.astype("uint8")).convert("RGB").resize((128, 128))))
    ImageDraw.Draw(im).text((3, 2), txt, fill=col); return np.asarray(im)


def build(task):
    src = f"{R}/task_a_inputs" if task == "so3" else f"{R}/task_b_inputs"
    inpname = "input.png" if task == "so3" else "rgb_white_128.png"
    outdir = f"{WT}/rebuttal/report/videos/by_method"; os.makedirs(outdir, exist_ok=True)
    made = {b[0]: 0 for b in BASELINES}
    for oid in sorted(os.listdir(src)):
        inp = np.asarray(Image.open(f"{src}/{oid}/{inpname}").convert("RGB").resize((128, 128)))
        ours = orbit(f"{R}/orbits/ours_{task}/{oid}.npy")
        gt = gt_orbit(oid) if task == "so3" else None
        sc = SC.get(oid, {})
        for key, name, scorekey in BASELINES:
            b = orbit(f"{R}/orbits/{key}_{task}/{oid}.npy")
            if b is None or ours is None:
                continue
            gap = np.full((128, 4, 3), 255, np.uint8)
            frames = []
            for i in range(NF):
                cells = [lab(inp, "input", (20, 20, 20)),
                         lab(ours[i], "OURS", (200, 20, 20)),
                         lab(b[i], name.split('(')[0].strip().split('—')[0].strip(), (20, 60, 200))]
                if gt is not None:
                    cells.append(lab(gt[i], "GT", (20, 120, 20)))
                row = []
                for c in cells:
                    row += [c, gap]
                frames.append(np.concatenate(row[:-1], 1))
            out = f"{outdir}/{key}_{task}_{oid}.mp4"
            imageio.mimsave(out, frames, fps=8, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            made[key] += 1
    print(f"[{task}]", made)


for t in ["so3", "rc"]:
    build(t)
print("BY_METHOD DONE")
