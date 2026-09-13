"""Build NVS comparison videos from the ACTUAL SCORED RENDERS (each method rendered at the
same 24 GT target cameras, then PSNR'd vs the GT target images). Frame i = GT camera i, so
every panel is the same viewpoint and the video literally shows what the metric measured.

SO(3) only (RealCars has no GT novel views -> no NVS; that stays an orbit, labeled so).
NFI panel set includes own-pose AND oracle-pose so the pose correction is visible.
pi-GAN has no calibrated NVS (assumes frontal) -> not included here (orbit only).
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, json, os
import numpy as np, imageio.v2 as imageio
from PIL import Image, ImageDraw
WT = f"{_EXT}/splatter-image-rebuttal"
R = f"{WT}/rebuttal/results"
SC = json.load(open(f"{R}/per_object_scores.json"))
OUT = f"{WT}/rebuttal/report/videos/nvs"; os.makedirs(OUT, exist_ok=True)

# panel spec per baseline: list of (label, dir-template or 'input'/'gt', color)
OURS = ("OURS", "task_a_runs/ours_nvs/{oid}", (200, 20, 20))
GROUPS = {
 "nfi": [("input", "input", (20, 20, 20)), OURS,
         ("NFI own-pose", "task_a_runs/nfi/{oid}/renders_ext300", (20, 60, 200)),
         ("NFI oracle-pose", "task_a_oracle/nfi_so3/{oid}/renders", (150, 90, 20)),
         ("GT", "gt", (20, 120, 20))],
 "eg3d": [("input", "input", (20, 20, 20)), OURS,
          ("EG3D+PTI", "task_a_runs/eg3d/{oid}/renders", (20, 60, 200)), ("GT", "gt", (20, 120, 20))],
 "finv": [("input", "input", (20, 20, 20)), OURS,
          ("FINV-SV", "task_a_runs/finv/{oid}/renders", (20, 60, 200)), ("GT", "gt", (20, 120, 20))],
}


def lab(arr, txt, col):
    im = Image.fromarray(arr.astype("uint8")).convert("RGB").resize((128, 128))
    ImageDraw.Draw(im).text((3, 2), txt, fill=col); return np.asarray(im)


def load(spec, oid, i, gtfs, inp_img):
    if spec == "input":
        return inp_img
    if spec == "gt":
        return np.asarray(Image.open(f"{R}/task_a_inputs/{oid}/targets/{gtfs[i]}").convert("RGB"))
    p = f"{R}/{spec.format(oid=oid)}/{i:02d}.png"
    if not os.path.exists(p):
        return np.full((128, 128, 3), 245, np.uint8)
    return np.asarray(Image.open(p).convert("RGB"))


def main():
    made = {}
    for oid in sorted(os.listdir(f"{R}/task_a_inputs")):
        gtdir = f"{R}/task_a_inputs/{oid}/targets"
        if not os.path.isdir(gtdir):
            continue
        gtfs = sorted(os.listdir(gtdir)); N = len(gtfs)
        inp_img = np.asarray(Image.open(f"{R}/task_a_inputs/{oid}/input.png").convert("RGB"))
        for key, panels in GROUPS.items():
            # require the baseline's own render dir to exist
            bdir = [p[1] for p in panels if p[1] not in ("input", "gt") and "ours" not in p[1]][0].format(oid=oid)
            if not os.path.isdir(f"{R}/{bdir}"):
                continue
            gap = np.full((128, 4, 3), 255, np.uint8)
            frames = []
            for i in range(N):
                cells = [lab(load(sp, oid, i, gtfs, inp_img), nm, col) for nm, sp, col in panels]
                row = []
                for c in cells:
                    row += [c, gap]
                frames.append(np.concatenate(row[:-1], 1))
            imageio.mimsave(f"{OUT}/{key}_{oid}.mp4", frames, fps=6, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p"])
            made[key] = made.get(key, 0) + 1
    print("NVS videos built:", made)


if __name__ == "__main__":
    main()
