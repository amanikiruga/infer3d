"""Batch Infer3D inverse-graphics inversion (EqM prior + Objaverse lifter, latent+pose,
multi-start, DINO+LPIPS+MSE) over real photos and Objaverse objects. Loads models once.
Saves per-object [input | inverted recon | recovered 3D turntable] mp4 + strip + PSNR."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, glob, json, os, sys
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/ig_batch_out"

OBJ_UIDS = {  # curated categories with on-disk objaverse renders (from LVIS)
 "teapot": ["45db143ee7cf48f4803a5d7be9d9b865","7fdad1eef0c2410480fd30f159cf7e92","ba49973ae14744a999a53749667c9092",
            "97c6a7860ce549bd933f55b4255f1162","d9951aeea7ef481bafb777ea54198f02","ac58cb3a28cb4a0d92ce1016f74fa5ff",
            "a5def4decf18404cb999ca6b9e22fee2","ccbeb0fe3e1f405a82ec9c4538ea8c9a","b0fcc0d3b15d4d3abf2e130d8ab79b96",
            "27cff3716bf84fedadd282b23c64a092","0bb17661ab004906be1ab452c6b2ade6","4c30437c274647e99a5638675516b653"],
 "mug": ["04d0553202d34b299bc0bf43025b6ef8","e5e87ddbf3f6470384bef58431351e2a","b4e5f0e6367e442c995eb1e241e61f74","ad2546804e8845b09e5db7c6db16b704"],
 "vase": ["c924ee98c4324ab0a5f4a1e3a11c5eae","d8ede9f25f4f4b36bdbe95957b5a8ee1","c1f0c96d628e47deafaf75848eb41048","a2a623cd107047caa11ac79451571918"],
 "helmet": ["7e5de66b10694898ae9b60cb1927a361","a2448a561901428ba571752068ec89ef"],
 "pineapple": ["ea05e523bb2946f992ed02b2a1bdb51c","e18c3147f7084f9096c6b54b094f08cd"],
 "pumpkin": ["2189104c64c149d6a6db6ebf1a2903c1","b51ee2d17c04491db2e5c6781ddb2f21"],
 "winebottle": ["72f5d79f44a7452fbc181845c137ec71","b60b4974c3724f478cfe7899a4896120"],
}


def obj_list():
    out = []
    for cat, uids in OBJ_UIDS.items():
        for i, u in enumerate(uids):
            out.append((f"obj_{cat}_{i:02d}_{u[:8]}", f"objuid:{u}"))
    return out


def real_list():
    R = os.path.dirname(os.path.abspath(__file__)) + "/real_images/proc"
    return [(f"real_{os.path.splitext(os.path.basename(p))[0]}", f"png:{p}")
            for p in sorted(glob.glob(f"{R}/*.png"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["real", "obj"], required=True)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--nrot", type=int, default=2)
    ap.add_argument("--steps1", type=int, default=45)
    ap.add_argument("--steps2", type=int, default=65)
    ap.add_argument("--tt", type=int, default=60)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=999)
    args = ap.parse_args()
    os.makedirs(f"{OUT}/{args.which}", exist_ok=True)

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae()
    cams = P.build_turntable(cfg, num=args.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()

    items = (real_list() if args.which == "real" else obj_list())[args.start:args.start + args.count]
    rows = []
    for tag, spec in items:
        try:
            I, M, _, gt = ig.load_obs(spec, cfg)
            spr, sp_canon, recon, R = ig.invert(I, M, gp, eqm, vae, cams, lp, cfg, use_dino=True, pose=True,
                                                K=args.K, nrot=args.nrot, steps1=args.steps1, steps2=args.steps2)
            psnr = None
            if gt is not None:
                ps = []
                for v in range(1, gt["wv"].shape[0]):
                    with torch.no_grad():
                        rr = P.render_view(cfg, spr, gt["wv"][v], gt["fp"][v], gt["cc"][v])
                    ps.append(float(-10 * torch.log10(((rr - gt["imgs"][v]) ** 2).mean() + 1e-12)))
                psnr = float(np.mean(ps))
            tt = P.turntable_frames(cfg, spr, cams)
            inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rec = (recon.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames = [np.concatenate([inp, rec, f], 1) for f in tt]
            imageio.mimsave(f"{OUT}/{args.which}/{tag}.mp4", frames, fps=20, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p"])
            Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/{args.which}/{tag}_strip.png")
            rows.append({"tag": tag, "psnr": psnr})
            print(f"[{args.which}] {tag}: novel PSNR {psnr}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"skip {tag}: {e}")
    json.dump(rows, open(f"{OUT}/{args.which}/summary_{args.start}.json", "w"), indent=1)
    ps = [r["psnr"] for r in rows if r["psnr"]]
    print(f"\n{args.which} done n={len(rows)} mean novel PSNR {np.mean(ps) if ps else 'n/a'}")


if __name__ == "__main__":
    main()
