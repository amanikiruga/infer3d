"""Batch OCCLUSION posterior-sampling (option 2): partial observation -> EqM prior completes
the hidden content -> coherent 3D. Real photos (held-out-region completion metric) and
Objaverse (GT novel-view IoU/PSNR, inversion vs feed-forward). Saves per-object
[partial input | inverted recon | feed-forward 3D | inversion 3D] + metrics."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, glob, json, os, sys
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
from ig2 import make_visibility, invert_partial
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/occ_out"
# real-object noun -> ImageNet class (for the class-conditional prior)
NOUN_CLASS = {"teapot": 849, "mug": 504, "vase": 883, "kettle": 899, "teddybear": 850,
              "backpack": 414, "banana": 954, "pineapple": 953, "guitar": 402, "helmet": 518,
              "birdhouse": 448, "toaster": 859, "winebottle": 907, "cowboyhat": 515,
              "pumpkin": 954, "boot": 514, "handbag": 748, "firehydrant": 415, "wateringcan": 899}


def tag_class(tag, which):
    if which != "real":
        for c, cid in {"teapot": 849, "mug": 504, "vase": 883, "helmet": 518, "pineapple": 953,
                       "pumpkin": 954, "winebottle": 907}.items():
            if c in tag:
                return cid
        return None
    noun = tag.replace("real_", "").rsplit("_", 1)[0]
    return NOUN_CLASS.get(noun)


def novel_iou_psnr(cfg, splats, gt):
    ious, ps = [], []
    for v in range(1, gt["wv"].shape[0]):
        with torch.no_grad():
            r = P.render_view(cfg, splats, gt["wv"][v], gt["fp"][v], gt["cc"][v])
        gm = (gt["imgs"][v] < 0.985).any(0).float(); rm = (r < 0.985).any(0).float()
        ious.append(float((gm * rm).sum() / ((gm + rm - gm * rm).sum() + 1e-6)))
        ps.append(float(-10 * torch.log10(((r - gt["imgs"][v]) ** 2).mean() + 1e-12)))
    return float(np.mean(ious)), float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["real", "obj"], required=True)
    ap.add_argument("--occ", default="right"); ap.add_argument("--frac", type=float, default=0.45)
    ap.add_argument("--tt", type=int, default=54)
    ap.add_argument("--classcond", action="store_true", help="use the true ImageNet class in the prior")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=999)
    a = ap.parse_args()
    sub = f"{a.which}_cls" if a.classcond else a.which
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae()
    cams = P.build_turntable(cfg, num=a.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()

    if a.which == "real":
        items = [(f"real_{os.path.splitext(os.path.basename(p))[0]}", f"png:{p}")
                 for p in sorted(glob.glob(os.path.dirname(os.path.abspath(__file__)) + "/real_images/proc/*.png"))]
    else:
        from ig_batch import OBJ_UIDS as U
        items = [(f"obj_{c}_{i:02d}", f"objuid:{u}") for c, us in U.items() for i, u in enumerate(us)]
    items = items[a.start:a.start + a.count]

    rows = []
    for tag, spec in items:
        try:
            I, M, _, gt = ig.load_obs(spec, cfg)
            V, O = make_visibility(M, a.occ, a.frac)
            occ_img = I * V + (1 - V)
            with torch.no_grad():
                ff = P.lift(gp, occ_img)
            cid = tag_class(tag, a.which) if a.classcond else None
            spr, recon, R = invert_partial(I, M, V, gp, eqm, vae, cams, lp, cfg, use_dino=True,
                                           pose=True, warm=True, n_prior=2, K=6, nrot=2,
                                           steps1=45, steps2=70, classid=cid)
            rec = {"tag": tag}
            Occ = (M - V).clamp(0, 1)
            rec["completion_psnr"] = float(-10 * torch.log10(((Occ[0] * (recon - I[0])) ** 2).sum() / (Occ.sum() * 3 + 1) + 1e-12))
            if gt is not None:
                ff_iou, ff_p = novel_iou_psnr(cfg, ff, gt)
                iv_iou, iv_p = novel_iou_psnr(cfg, spr, gt)
                rec.update(ff_iou=ff_iou, inv_iou=iv_iou, ff_psnr=ff_p, inv_psnr=iv_p)
            ff_tt = P.turntable_frames(cfg, ff, cams); iv_tt = P.turntable_frames(cfg, spr, cams)
            disp = ((I[0] * V[0] + Occ[0] + (1 - M[0])).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # occluded side -> white
            rc = (recon.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames = [np.concatenate([disp, rc, f, iv], 1) for f, iv in zip(ff_tt, iv_tt)]
            imageio.mimsave(f"{OUT}/{sub}/{tag}.mp4", frames, fps=18, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/{sub}/{tag}_strip.png")
            rows.append(rec); print(f"[{a.which}] {tag}: {rec}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"skip {tag}: {e}")
    json.dump(rows, open(f"{OUT}/{sub}/summary_{a.start}.json", "w"), indent=1)
    print(f"\n{a.which} done n={len(rows)}")


if __name__ == "__main__":
    main()
