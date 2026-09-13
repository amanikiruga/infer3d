"""Batch: held-out Objaverse objects (diverse LVIS categories) -> single-view 3D via the
analysis-by-synthesis loop (latent of the broad generative prior, optimized through the
FROZEN Objaverse lifter). GT-backed novel-view PSNR (feed-forward vs inverted) + turntables.

This is the genuinely-working regime (lifter in-domain). Produces the material for the
20+ object report answering R1 'single model spanning categories'.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys, json
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/gallery_obj"


def psnr(a, b): return float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tt", type=int, default=48)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/tt", exist_ok=True)

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg)
    vae = P.load_vae()
    cams = P.build_turntable(cfg, num=args.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()
    from splatter_image_datasets.objaverse import ObjaverseDataset
    ds = ObjaverseDataset(cfg, "test")

    rows = []
    for idx in range(args.start, min(args.start + args.count, len(ds))):
        try:
            item = ds[idx]
            oid = item.get("object_id", str(idx))
            img0 = item["gt_images"][0].to(P.DEV)
            I_128 = img0.unsqueeze(0)
            m = (img0 < 0.99).any(0, keepdim=True).float().unsqueeze(0)
            mask256 = F.interpolate(m, (256, 256), mode="nearest")
            gtw, gtf, gtc = item["world_view_transforms"].to(P.DEV), item["full_proj_transforms"].to(P.DEV), item["camera_centers"].to(P.DEV)
            gti = item["gt_images"].to(P.DEV)
            nV = gtw.shape[0]

            def novel(splats):
                ps = []
                for v in range(1, nV):
                    with torch.no_grad():
                        r = P.render_view(cfg, splats, gtw[v], gtf[v], gtc[v])
                    ps.append(psnr(r, gti[v]))
                return float(np.mean(ps))

            with torch.no_grad():
                ff = P.lift(gp, I_128)
            ff_psnr = novel(ff)

            I256 = F.interpolate(I_128, (256, 256), mode="bilinear", align_corners=False)
            with torch.no_grad():
                z0 = P.vae_encode(vae, I256)
            z = z0.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=args.lr)
            for s in range(args.steps):
                opt.zero_grad()
                img = P.vae_decode(vae, z)
                u = F.interpolate(img * mask256 + (1 - mask256), (128, 128), mode="bilinear", align_corners=False)
                sp = P.lift(gp, u)
                r = P.render_view(cfg, sp, cams[0][0], cams[1][0], cams[2][0]).unsqueeze(0)
                loss = F.mse_loss(r, I_128) + 0.5 * lp(r * 2 - 1, I_128 * 2 - 1).mean() + 0.02 * ((z - z0) ** 2).mean()
                loss.backward(); opt.step()
            with torch.no_grad():
                img = P.vae_decode(vae, z)
                u = F.interpolate(img * mask256 + (1 - mask256), (128, 128), mode="bilinear", align_corners=False)
                inv = P.lift(gp, u)
            inv_psnr = novel(inv)

            # turntable of the recovered geometry (input | GT-turntable-proxy via inv)
            tt = P.turntable_frames(cfg, inv, cams)
            inp = (I_128[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames = [np.concatenate([inp, f], axis=1) for f in tt]
            imageio.mimsave(f"{OUT}/tt/{idx:03d}_{oid}.mp4", frames, fps=16, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p"])
            Image.fromarray(inp).save(f"{OUT}/tt/{idx:03d}_{oid}_input.png")
            rows.append({"idx": idx, "oid": oid, "ff_psnr": ff_psnr, "inv_psnr": inv_psnr})
            print(f"[{idx}] {oid}: FF {ff_psnr:.2f} | INV {inv_psnr:.2f}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"skip {idx}: {e}")

    rows.sort(key=lambda r: -r["inv_psnr"])
    summ = {"n": len(rows), "mean_ff": float(np.mean([r["ff_psnr"] for r in rows])),
            "mean_inv": float(np.mean([r["inv_psnr"] for r in rows])), "rows": rows}
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print(f"\nn={summ['n']} mean FF {summ['mean_ff']:.2f} | mean INV {summ['mean_inv']:.2f}")


if __name__ == "__main__":
    main()
