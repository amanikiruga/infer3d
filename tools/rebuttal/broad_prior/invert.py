"""Analysis-by-synthesis inversion: EqM (ImageNet EBM) prior + Objaverse lifter.
Optimize the EqM latent z so that Phi(decode(z)) rendered at the source view matches the
input; an EqM equilibrium-gradient step keeps z on the ImageNet manifold (frozen-lifter +
frozen-prior, exactly the paper's Eq.2 with an EBM prior). Compare feed-forward vs inverted
on NOVEL views (GT-backed for objaverse inputs) and render turntables.

  --source objaverse:IDX   (has GT novel views -> reports PSNR)
  --source eqm:CLASSID      (EqM-generated input, qualitative)
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys, json
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/inv"
CLASSES = {817: "sports car", 555: "fire truck", 850: "teddy bear", 954: "banana",
           953: "pineapple", 968: "coffee cup", 574: "golf ball", 933: "hamburger"}


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12))


def load_source(kind, cfg, gp, eqm, vae, sam, arg):
    """Returns (I_128 target [1,3,128,128], mask256 [1,1,256,256], gt_novel or None, tag)."""
    if kind == "objaverse":
        from splatter_image_datasets.objaverse import ObjaverseDataset
        ds = ObjaverseDataset(cfg, "test")
        item = ds[int(arg)]
        oid = item.get("object_id", arg)
        img0 = item["gt_images"][0].to(P.DEV)                         # [3,128,128] white bg
        I_128 = img0.unsqueeze(0)
        m = (img0 < 0.99).any(0, keepdim=True).float().unsqueeze(0)   # fg mask 128
        mask256 = F.interpolate(m, (256, 256), mode="nearest")
        # GT novel views + cameras for PSNR
        gt = {"imgs": item["gt_images"].to(P.DEV),
              "wv": item["world_view_transforms"].to(P.DEV),
              "fp": item["full_proj_transforms"].to(P.DEV),
              "cc": item["camera_centers"].to(P.DEV)}
        return I_128, mask256, gt, f"obj_{oid}"
    elif kind == "eqm":
        from eqm_sanity import nag_gd_sample
        cid = int(arg)
        with torch.no_grad():
            lat = nag_gd_sample(eqm, torch.tensor([cid], device=P.DEV), 250, cfg=1.5)
            img = P.vae_decode(vae, lat)[0]                            # [3,256,256]
        u8 = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mk = sam.mask(u8, CLASSES.get(cid, "object"))
        if mk is None:
            mk = np.ones(u8.shape[:2], bool)
        I_128 = P.center_on_white(img, mk, out=128).unsqueeze(0)
        mask256 = F.interpolate(torch.from_numpy(mk).float().view(1, 1, 256, 256).to(P.DEV), (256, 256))
        return I_128, mask256, None, f"eqm_{cid}_{CLASSES.get(cid,'obj').replace(' ','_')}"
    else:
        raise ValueError(kind)


def eval_novel(cfg, gp, splats, gt):
    if gt is None:
        return None
    ps = []
    for v in range(1, gt["wv"].shape[0]):
        with torch.no_grad():
            r = P.render_view(cfg, splats, gt["wv"][v], gt["fp"][v], gt["cc"][v])
        ps.append(psnr(r, gt["imgs"][v]))
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="objaverse:IDX or eqm:CLASSID")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.02, help="anchor ||z-z0||")
    ap.add_argument("--eqm_eta", type=float, default=0.0, help="EqM manifold step (0=off)")
    ap.add_argument("--tt", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    kind, arg = args.source.split(":")

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg)
    eqm = P.load_eqm() if kind == "eqm" or args.eqm_eta > 0 else None
    vae = P.load_vae()
    sam = P.SAM3() if kind == "eqm" else None
    cams = P.build_turntable(cfg, num=args.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()

    I_128, mask256, gt, tag = load_source(kind, cfg, gp, eqm, vae, sam, arg)
    print(f"source {tag}")

    # feed-forward baseline
    with torch.no_grad():
        ff_splats = P.lift(gp, I_128)
    ff_novel = eval_novel(cfg, gp, ff_splats, gt)

    # init latent from VAE-encode of the (white-bg) input at 256
    I_256 = F.interpolate(I_128, (256, 256), mode="bilinear", align_corners=False)
    with torch.no_grad():
        z0 = P.vae_encode(vae, I_256)
    z = z0.clone().requires_grad_(True)
    y = torch.tensor([1000], device=P.DEV)  # null class (unconditional prior)
    opt = torch.optim.Adam([z], lr=args.lr)

    for step in range(args.steps):
        opt.zero_grad()
        img = P.vae_decode(vae, z)                                  # [1,3,256,256]
        u = F.interpolate(img * mask256 + (1 - mask256), (128, 128), mode="bilinear", align_corners=False)
        splats = P.lift(gp, u)
        r = P.render_view(cfg, splats, cams[0][0], cams[1][0], cams[2][0]).unsqueeze(0)
        loss = F.mse_loss(r, I_128) + 0.5 * lp(r * 2 - 1, I_128 * 2 - 1).mean() + args.lam * ((z - z0) ** 2).mean()
        loss.backward()
        opt.step()
        if args.eqm_eta > 0:
            with torch.no_grad():
                g = P.eqm_grad(eqm, z, y)
                z.data += args.eqm_eta * g
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step}: loss {loss.item():.4f}")

    with torch.no_grad():
        img = P.vae_decode(vae, z)
        u = F.interpolate(img * mask256 + (1 - mask256), (128, 128), mode="bilinear", align_corners=False)
        inv_splats = P.lift(gp, u)
    inv_novel = eval_novel(cfg, gp, inv_splats, gt)
    print(f"[{tag}] feed-forward novel PSNR {ff_novel} | inverted novel PSNR {inv_novel}")

    # turntables: feed-forward vs inverted, side by side
    ff_tt = P.turntable_frames(cfg, ff_splats, cams)
    inv_tt = P.turntable_frames(cfg, inv_splats, cams)
    inp = (I_128[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, ff, iv], axis=1) for ff, iv in zip(ff_tt, inv_tt)]
    imageio.mimsave(f"{OUT}/tt_{tag}.mp4", frames, fps=20, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    # strip
    picks = [0, len(frames) // 6, len(frames) // 3, len(frames) // 2, 2 * len(frames) // 3]
    Image.fromarray(np.concatenate([frames[p] for p in picks], axis=0)).save(f"{OUT}/strip_{tag}.png")
    json.dump({"tag": tag, "ff_novel_psnr": ff_novel, "inv_novel_psnr": inv_novel,
               "steps": args.steps, "eqm_eta": args.eqm_eta},
              open(f"{OUT}/res_{tag}.json", "w"), indent=1)
    print(f"saved tt_{tag}.mp4 (input | feed-forward | inverted), strip_{tag}.png")


if __name__ == "__main__":
    main()
