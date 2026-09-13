"""Broad-prior breadth gallery: sample diverse objects from the EqM ImageNet EBM, mask
(SAM3), lift with the single Objaverse lifter, render turntables. One (prior, lifter) pair
spanning many categories -> direct answer to R1 'no single model spanning datasets'.
Saves per-object: EqM sample, masked-centered input, source render, turntable mp4 + strip,
and source-view reconstruction PSNR (feed-forward + inverted)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys, json
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
from eqm_sanity import nag_gd_sample
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/gallery_eqm"
# diverse compact single-object ImageNet classes (id -> SAM3 noun)
CLASSES = {850: "teddy bear", 968: "cup", 849: "teapot", 883: "vase", 954: "banana",
           953: "pineapple", 950: "orange", 951: "lemon", 957: "pomegranate", 947: "mushroom",
           931: "bagel", 933: "hamburger", 963: "pizza", 574: "golf ball", 805: "soccer ball",
           430: "basketball", 898: "water bottle", 907: "wine bottle", 440: "beer bottle",
           448: "birdhouse", 607: "jack-o-lantern", 859: "toaster", 892: "wall clock",
           673: "computer mouse", 504: "coffee mug", 999: "toilet paper roll", 837: "sunglasses",
           515: "cowboy hat", 852: "tennis ball", 723: "ping-pong ball"}


def psnr(a, b): return float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--tt", type=int, default=48)
    ap.add_argument("--bs", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/tt", exist_ok=True)

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg)
    eqm = P.load_eqm()
    vae = P.load_vae()
    sam = P.SAM3()
    cams = P.build_turntable(cfg, num=args.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()

    ids = list(CLASSES.keys())
    rows = []
    for b0 in range(0, len(ids), args.bs):
        batch = ids[b0:b0 + args.bs]
        with torch.no_grad():
            lat = nag_gd_sample(eqm, torch.tensor(batch, device=P.DEV), args.steps, cfg=args.cfg)
            imgs = P.vae_decode(vae, lat)
        for bi, cid in enumerate(batch):
            noun = CLASSES[cid]
            img01 = imgs[bi]
            u8 = (img01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            mk = sam.mask(u8, noun)
            if mk is None or mk.mean() < 0.01:
                print(f"[{cid} {noun}] SAM3 weak/no mask ({0 if mk is None else 100*mk.mean():.0f}%), skip")
                continue
            u = P.center_on_white(img01, mk, out=128)
            with torch.no_grad():
                sp = P.lift(gp, u)
                src = P.render_view(cfg, sp, cams[0][0], cams[1][0], cams[2][0])
            src_psnr = psnr(src, u)
            tt = P.turntable_frames(cfg, sp, cams)
            uu8 = (u.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames = [np.concatenate([uu8, f], axis=1) for f in tt]
            imageio.mimsave(f"{OUT}/tt/{cid}_{noun.replace(' ', '_')}.mp4", frames, fps=16,
                            codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            Image.fromarray(u8).save(f"{OUT}/tt/{cid}_{noun.replace(' ', '_')}_eqm.png")
            Image.fromarray(uu8).save(f"{OUT}/tt/{cid}_{noun.replace(' ', '_')}_input.png")
            picks = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4]
            Image.fromarray(np.concatenate([frames[p] for p in picks], axis=0)).save(
                f"{OUT}/tt/{cid}_{noun.replace(' ', '_')}_strip.png")
            rows.append({"cid": cid, "noun": noun, "src_psnr": src_psnr, "fg": float(mk.mean())})
            print(f"[{cid} {noun}] src-recon PSNR {src_psnr:.2f} fg={100*mk.mean():.0f}%")
    json.dump({"n": len(rows), "rows": sorted(rows, key=lambda r: -r["src_psnr"])},
              open(f"{OUT}/summary.json", "w"), indent=1)
    print(f"\nDONE n={len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
