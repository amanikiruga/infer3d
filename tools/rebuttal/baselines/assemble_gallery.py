"""Assemble combined per-object comparison videos for the report gallery.
Each video: input | ours | NFI | EG3D | pi-GAN | GT  (orbit, frame-synced to 24).
Missing method panels are skipped gracefully (labeled). Writes report/videos/gallery/.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw

WT = f"{_EXT}/splatter-image-rebuttal"
R = f"{WT}/rebuttal/results"; NF = 24


def load_npy_orbit(p):
    if not os.path.exists(p):
        return None
    a = np.load(p)  # (F,128,128,3)
    idx = np.linspace(0, len(a) - 1, NF).round().astype(int)
    return a[idx]


def gt_orbit_so3(oid):
    d = f"{R}/task_a_inputs/{oid}/targets"
    if not os.path.isdir(d):
        return None
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    a = np.stack([np.asarray(Image.open(f"{d}/{x}").convert("RGB")) for x in fs])
    idx = np.linspace(0, len(a) - 1, NF).round().astype(int)
    return a[idx]


def lab(img, txt, col=(220, 20, 20)):
    im = Image.fromarray(img.copy().astype(np.uint8)); ImageDraw.Draw(im).text((3, 2), txt, fill=col)
    return np.asarray(im)


def build(oid, task, out_dir):
    if task == "so3":
        inp = f"{R}/task_a_inputs/{oid}/input.png"
        panels = [("input", None), ("OURS", f"{R}/orbits/ours_so3/{oid}.npy"),
                  ("NFI", f"{R}/orbits/nfi_so3/{oid}.npy"), ("EG3D", f"{R}/orbits/eg3d_so3/{oid}.npy"),
                  ("pi-GAN", f"{R}/orbits/pigan_so3/{oid}.npy")]
        gt = gt_orbit_so3(oid)
    else:
        inp = f"{R}/task_b_inputs/{oid}/rgb_white_128.png"
        panels = [("input", None), ("OURS", f"{R}/orbits/ours_rc/{oid}.npy"),
                  ("NFI", f"{R}/orbits/nfi_rc/{oid}.npy"), ("EG3D", f"{R}/orbits/eg3d_rc/{oid}.npy"),
                  ("pi-GAN", f"{R}/orbits/pigan_rc/{oid}.npy")]
        gt = None
    inp_img = np.asarray(Image.open(inp).convert("RGB").resize((128, 128)))
    orbits = []
    for name, p in panels:
        if p is None:
            orbits.append((name, None))
        else:
            orbits.append((name, load_npy_orbit(p)))
    if gt is not None:
        orbits.append(("GT", gt))
    frames = []
    gap = np.full((128, 4, 3), 255, np.uint8)
    for i in range(NF):
        row = []
        for name, orb in orbits:
            if name == "input":
                cell = lab(inp_img, "input", (20, 20, 20))
            elif orb is None:
                cell = lab(np.full((128, 128, 3), 245, np.uint8), name + " n/a", (150, 150, 150))
            else:
                cell = lab(np.asarray(Image.fromarray(orb[i]).convert("RGB").resize((128, 128))), name,
                           (20, 120, 20) if name == "GT" else (220, 20, 20))
            row.append(cell); row.append(gap)
        frames.append(np.concatenate(row[:-1], axis=1))
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{task}_{oid}.mp4"
    imageio.mimsave(out, frames, fps=8, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["so3", "rc", "both"], default="both")
    args = ap.parse_args()
    out_dir = f"{WT}/rebuttal/report/videos/gallery"
    tasks = ["so3", "rc"] if args.task == "both" else [args.task]
    for task in tasks:
        src = f"{R}/task_a_inputs" if task == "so3" else f"{R}/task_b_inputs"
        ids = sorted(os.listdir(src))
        n = 0
        for oid in ids:
            try:
                build(oid, task, out_dir); n += 1
            except Exception as e:
                print(f"skip {task} {oid}: {e}")
        print(f"[{task}] built {n} gallery videos")
    print("GALLERY DONE")


if __name__ == "__main__":
    main()
