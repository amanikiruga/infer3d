"""Occlusion CD eval: for Objaverse objects, occlude the observation, compare
(a) Splatter-Image direct = Phi(occluded)  vs  (b) Infer3D-EqM occlusion inversion,
by FUSED Chamfer Distance to the true GLB-mesh GT (fuse_cd.py). Class-conditional prior.
Saves per-object overlay [occluded input | Splatter fused vs GT | Infer3D fused vs GT] to
eyeball, and a summary with CD_ff, CD_inv per object. Select winners (CD_inv < CD_ff) after."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P, fuse_cd as FC
from ig2 import make_visibility, invert_partial
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/cd_out"


def scat(A, B, res=200):
    img = np.full((res, res, 3), 255, np.uint8)
    allp = torch.cat([A, B], 0); c = allp.mean(0); s = (allp - c).abs().max() * 1.1 + 1e-9
    def draw(pts, col):
        p = ((pts - c) / s * 0.5 + 0.5)[:, [0, 1]].clamp(0, 0.995)
        xy = (p * res).long().cpu().numpy()
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(res - 1 - xy[:, 1] - dy, 0, res - 1), np.clip(xy[:, 0] + dx, 0, res - 1)] = col
    draw(B, np.array([150, 150, 150], np.uint8)); draw(A, np.array([220, 30, 30], np.uint8))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=999)
    ap.add_argument("--occ", default="right"); ap.add_argument("--frac", type=float, default=0.45)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pool = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/cd_pool.json"))
    cats = pool["cats"]
    items = [(uid, c, cats[c]) for c, us in pool["pool"].items() for uid in us][a.start:a.start + a.count]

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae()
    cams = P.build_turntable(cfg, num=8)  # unused for CD; kept for invert_partial signature
    lp = lpips_lib.LPIPS(net="vgg").to(FC.DEV).eval()

    rows = []
    for uid, cat, cid in items:
        try:
            gt = FC.gt_cloud(uid, 50000)
            it = ig.objaverse_item_uid(cfg, uid, src_idx=0)
            I = it["gt_images"][0].unsqueeze(0).to(FC.DEV); M = (I < 0.985).any(1, keepdim=True).float()
            V, O = make_visibility(M, a.occ, a.frac)
            occ_img = I * V + (1 - V)
            with torch.no_grad():
                ff = P.lift(gp, occ_img)
            spr, recon, R = invert_partial(I, M, V, gp, eqm, vae, cams, lp, cfg, use_dino=True,
                                           pose=False, warm=True, n_prior=2, K=6, nrot=2,
                                           steps1=45, steps2=70, classid=cid)
            Pff = FC.fuse(cfg, ff, n_views=120)
            Pinv = FC.fuse(cfg, spr, n_views=120)
            cd_ff, Xff, Y, _ = FC.align_cd(Pff, gt)
            cd_inv, Xinv, _, _ = FC.align_cd(Pinv, gt)
            tag = f"{cat.replace(' ', '')}_{uid[:8]}"
            disp = ((I[0] * V[0] + O[0] + (1 - M[0])).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            disp = np.array(Image.fromarray(disp).resize((200, 200)))
            grid = np.concatenate([disp, scat(Xff, Y), scat(Xinv, Y)], 1)
            Image.fromarray(grid).save(f"{OUT}/{tag}.png")
            rows.append(dict(tag=tag, cat=cat, uid=uid, cd_ff=cd_ff, cd_inv=cd_inv, win=cd_inv < cd_ff))
            print(f"[{tag}] CD  splatter {cd_ff:.3f} | infer3d {cd_inv:.3f}  {'WIN' if cd_inv<cd_ff else '-'}")
        except Exception as e:
            import traceback; traceback.print_exc(); print("skip", uid, e)
    json.dump(rows, open(f"{OUT}/summary_{a.start}.json", "w"), indent=1)
    w = sum(r["win"] for r in rows)
    print(f"\nshard done n={len(rows)} wins {w}")


if __name__ == "__main__":
    main()
