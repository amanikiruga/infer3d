"""Patch existing occlusion videos: replace the first column (partial input) so the occluded
side is FULLY occluded (white) instead of grey. Only rewrites the left 128px of each frame;
the inversion results (recon / feed-forward / inversion 3D) are untouched. CPU only."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, os, sys
import numpy as np, imageio.v2 as imageio
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = f"{_EXT}"
OBJROOT = f"{ROOT}/datasets/objaverse/views_release"
FRAC = 0.45


def white_occluded(I):
    """I: [128,128,3] float 0-1 (object on white). Occlude right FRAC of the object -> white."""
    M = (I < 0.985).any(2)
    ys, xs = np.where(M)
    disp = I.copy()
    if len(xs):
        x0, x1 = xs.min(), xs.max()
        O = np.zeros_like(M); O[:, int(x1 - FRAC * (x1 - x0)):] = True
        disp[O & M] = 1.0
    return (np.clip(disp, 0, 1) * 255).astype(np.uint8)


def real_disp(tag):
    p = f"{HERE}/real_images/proc/{tag.replace('real_', '')}.png"
    I = np.asarray(Image.open(p).convert("RGB").resize((128, 128))).astype(np.float32) / 255.
    return white_occluded(I)


def obj_disp(tag):
    # tag like obj_teapot_00 -> need uid; recover from ig_batch OBJ_UIDS by category+index
    sys.path.insert(0, HERE)
    from ig_batch import OBJ_UIDS
    parts = tag.split("_"); cat = parts[1]; idx = int(parts[2])
    uid = OBJ_UIDS[cat][idx]
    p = sorted(glob.glob(f"{OBJROOT}/{uid}/*.png"))[0]
    im = Image.open(p).convert("RGBA").resize((128, 128))
    a = np.asarray(im).astype(np.float32) / 255.
    rgb, al = a[..., :3], a[..., 3:]
    I = rgb * al + (1 - al)                      # composite on white (matches loader)
    return white_occluded(I)


def patch(sub, disp_fn):
    n = 0
    for mp4 in sorted(glob.glob(f"{HERE}/occ_out/{sub}/*.mp4")):
        tag = os.path.splitext(os.path.basename(mp4))[0]
        try:
            disp = disp_fn(tag)
        except Exception as e:
            print("skip", tag, e); continue
        frames = [f.copy() for f in imageio.get_reader(mp4)]
        for f in frames:
            f[:, :128] = disp
        imageio.mimsave(mp4, frames, fps=18, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
        picks = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4]
        Image.fromarray(np.concatenate([frames[i] for i in picks], 0)).save(mp4.replace(".mp4", "_strip.png"))
        n += 1
    print(f"patched {n} in {sub}")


if __name__ == "__main__":
    subs = sys.argv[1:] or ["real", "obj"]
    for sub in subs:
        if os.path.isdir(f"{HERE}/occ_out/{sub}"):
            patch(sub, real_disp if sub.startswith("real") else obj_disp)
