"""MILESTONE 1: EqM sample -> SAM3 mask -> center -> Objaverse lifter -> turntable.
Tests the forward chain (no inversion yet): can Phi lift EqM/natural-style masked images
into recognizable 3D? Saves per-object montage + turntable mp4 for VISUAL inspection.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
sys.path.insert(0, P.EQM)
from eqm_sanity import nag_gd_sample  # reuse sampler

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/m1"
# ImageNet id -> SAM3 noun prompt (clean single-object classes)
CLASSES = {817: "sports car", 555: "fire truck", 850: "teddy bear", 954: "banana",
           953: "pineapple", 968: "coffee cup", 574: "golf ball", 933: "hamburger"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--tt", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg)
    eqm = P.load_eqm()
    vae = P.load_vae()
    sam = P.SAM3()
    cams = P.build_turntable(cfg, num=args.tt)
    print("loaded all models")

    ids = list(CLASSES.keys())
    ys = torch.tensor(ids, device=P.DEV)
    with torch.no_grad():
        lat = nag_gd_sample(eqm, ys, args.steps, cfg=args.cfg)
        imgs = P.vae_decode(vae, lat)                    # [B,3,256,256]

    for bi, cid in enumerate(ids):
        img01 = imgs[bi]
        img_u8 = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        noun = CLASSES[cid]
        mask = sam.mask(img_u8, noun)
        if mask is None:
            print(f"[{cid} {noun}] SAM3 no mask; using full frame")
            mask = np.ones(img_u8.shape[:2], bool)
        u = P.center_on_white(img01, mask, out=128)      # [3,128,128]
        with torch.no_grad():
            splats = P.lift(gp, u)
            src = P.render_view(cfg, splats, cams[0][0], cams[1][0], cams[2][0])
        # montage: EqM image | masked+centered input | source render
        row = np.concatenate([
            img_u8,
            (u.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).repeat(2, 0).repeat(2, 1),
            (src.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).repeat(2, 0).repeat(2, 1),
        ], axis=1)
        Image.fromarray(row).save(f"{OUT}/obj_{cid}_{noun.replace(' ', '_')}.png")
        frames = P.turntable_frames(cfg, splats, cams)
        imageio.mimsave(f"{OUT}/tt_{cid}_{noun.replace(' ', '_')}.mp4", frames, fps=20,
                        codec="libx264", output_params=["-pix_fmt", "yuv420p"])
        ng = splats["xyz"].shape[0]
        print(f"[{cid} {noun}] mask fg={100*mask.mean():.0f}% nGauss={ng} -> saved montage+tt")

    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
