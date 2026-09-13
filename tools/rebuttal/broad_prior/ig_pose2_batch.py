"""Batch OPTION 1 (pose reasoning): observe top-down Objaverse view -> EqM natural hypotheses
-> pose search -> recovered natural 3D. Loads models once. Per object: recovered pose tilt vs
true observation elevation (pose-recovery metric) + [top-down | natural hypothesis |
feed-forward 3D | Infer3D 3D] video."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
from ig import rotate_splats, dino_feats
from ig_pose import Rx, Ry
from eqm_sanity import nag_gd_sample
from utils.abs_utils import symmetric_orthogonalization

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/pose2_batch"
CLASS = {"teapot": 849, "mug": 504, "vase": 883, "helmet": 518, "pineapple": 953, "pumpkin": 954, "winebottle": 907}


def sil(x):
    return (1 - x.min(1, keepdim=True).values).clamp(0, 1)


def novel_iou(cfg, splats, gt):
    ious = []
    for v in range(1, gt["wv"].shape[0]):
        with torch.no_grad():
            r = P.render_view(cfg, splats, gt["wv"][v], gt["fp"][v], gt["cc"][v])
        gm = (gt["imgs"][v] < 0.985).any(0).float(); rm = (r < 0.985).any(0).float()
        ious.append(float((gm * rm).sum() / ((gm + rm - gm * rm).sum() + 1e-6)))
    return float(np.mean(ious))


def run_one(uid, noun, cfg, gp, eqm, vae, sam, cams, K, steps, ROTS):
    td, elevs = ig.objaverse_topdown_idx(cfg, uid)
    it = ig.objaverse_item_uid(cfg, uid, src_idx=td)
    I = it["gt_images"][0].unsqueeze(0).to(P.DEV); M = (I < 0.985).any(1, keepdim=True).float()
    gt = {"imgs": it["gt_images"].to(P.DEV), "wv": it["world_view_transforms"].to(P.DEV),
          "fp": it["full_proj_transforms"].to(P.DEV), "cc": it["camera_centers"].to(P.DEV)}
    src = (gt["wv"][0], gt["fp"][0], gt["cc"][0])
    with torch.no_grad():
        ff = P.lift(gp, I)
    ff_iou = novel_iou(cfg, ff, gt)
    with torch.no_grad():
        imgs = P.vae_decode(vae, nag_gd_sample(eqm, torch.full((K,), CLASS[noun], device=P.DEV), 250, cfg=1.5))
    hyp = []
    for k in range(K):
        u8 = (imgs[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mk = sam.mask(u8, noun)
        if mk is None or mk.mean() < 0.02:
            continue
        u = P.center_on_white(imgs[k], mk, out=128)
        with torch.no_grad():
            hyp.append((u, P.lift(gp, u.unsqueeze(0))))
    if not hyp:
        return None
    Isil, Idino = sil(I), dino_feats(I)
    best = None
    for u, sp in hyp:
        for R0 in ROTS:
            R = R0.clone().requires_grad_(True); opt = torch.optim.Adam([R], lr=0.05)
            for t in range(steps):
                opt.zero_grad()
                spr = rotate_splats(sp, symmetric_orthogonalization(R.unsqueeze(0))[0])
                r = P.render_view(cfg, spr, src[0], src[1], src[2]).unsqueeze(0)
                (F.mse_loss(sil(r), Isil) + 0.5 * (1 - (dino_feats(r) * Idino).sum(-1).mean())).backward()
                opt.step()
            with torch.no_grad():
                Rf = symmetric_orthogonalization(R.detach().unsqueeze(0))[0]
                r = P.render_view(cfg, rotate_splats(sp, Rf), src[0], src[1], src[2]).unsqueeze(0)
                loss = float(F.mse_loss(sil(r), Isil) + 0.5 * (1 - (dino_feats(r) * Idino).sum(-1).mean()))
            if best is None or loss < best[0]:
                best = (loss, Rf, u, sp)
    loss, Rf, u, sp = best
    spr = rotate_splats(sp, Rf)
    tilt = float(torch.acos(((torch.diagonal(Rf).sum() - 1) / 2).clamp(-1, 1)) * 180 / np.pi)
    inv_iou = novel_iou(cfg, spr, gt)
    ff_tt = P.turntable_frames(cfg, ff, cams); iv_tt = P.turntable_frames(cfg, spr, cams)
    inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hy = (u.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, hy, f, iv], 1) for f, iv in zip(ff_tt, iv_tt)]
    return dict(uid=uid, noun=noun, elev=elevs[td], tilt=tilt, ff_iou=ff_iou, inv_iou=inv_iou), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=3); ap.add_argument("--steps", type=int, default=70)
    ap.add_argument("--tt", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae(); sam = P.SAM3()
    cams = P.build_turntable(cfg, num=a.tt)
    ROTS = [Rx(55), Rx(-55), Ry(55) @ Rx(60), Rx(90)]
    from ig_batch import OBJ_UIDS
    items = [(u, cat) for cat, us in OBJ_UIDS.items() for u in us if cat in CLASS]
    rows = []
    for uid, noun in items:
        try:
            res = run_one(uid, noun, cfg, gp, eqm, vae, sam, cams, a.K, a.steps, ROTS)
            if res is None:
                print(f"skip {uid} (no hyp)"); continue
            rec, frames = res
            tag = f"{noun}_{uid[:8]}"
            imageio.mimsave(f"{OUT}/{tag}.mp4", frames, fps=16, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/{tag}_strip.png")
            rec["tag"] = tag; rows.append(rec)
            print(f"[{tag}] elev {rec['elev']:.0f} -> recovered tilt {rec['tilt']:.0f}  IoU ff {rec['ff_iou']:.3f} inv {rec['inv_iou']:.3f}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"skip {uid}: {e}")
    json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=1)
    print(f"\ndone n={len(rows)}")


if __name__ == "__main__":
    main()
