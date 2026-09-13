"""OPTION 1 v4 -- brute-force pose, no rotation drift. For each of many grid rotations we FREEZE
the rotation and fit only scale+translation to the observation silhouette; the rotation whose
best-fit reprojection matches the observation wins. This cannot drift to a side view because
rotation is never gradient-updated. Goal: at least one object where the recovered 3D, rendered
back at the INPUT pose, reproduces the top-down observation (reprojection IoU high) AND is a
coherent natural object (turntable) -- a genuine positive for pose reasoning with EqM.
Runs over several uids and reports the best.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys, json
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
from ig import dino_feats
from ig_pose import Rx, Ry, Rz
from ig_pose3 import transform_splats, sil, shape_score
from utils.abs_utils import symmetric_orthogonalization

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/pose4_out"
def _candidates():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ig_batch import OBJ_UIDS
    CLS = {"teapot": (849, "teapot"), "mug": (504, "coffee mug"), "vase": (883, "vase"),
           "winebottle": (907, "wine bottle"), "pumpkin": (954, "pumpkin"),
           "pineapple": (953, "pineapple"), "helmet": (518, "helmet")}
    out = []
    for cat, uids in OBJ_UIDS.items():
        if cat not in CLS:
            continue
        cid, noun = CLS[cat]
        for u in uids:
            out.append((u, cid, noun))
    return out


CANDIDATES = _candidates()


def reproj_iou(cfg, splats, srcC, Isil_bool):
    with torch.no_grad():
        r = P.render_view(cfg, splats, srcC[0], srcC[1], srcC[2])
    rm = (r < 0.985).any(0).float()
    inter = (Isil_bool * rm).sum(); union = (Isil_bool + rm - Isil_bool * rm).sum()
    return float(inter / (union + 1e-6)), r


def novel_iou(cfg, splats, gt):
    ious = []
    for v in range(1, gt["wv"].shape[0]):
        with torch.no_grad():
            r = P.render_view(cfg, splats, gt["wv"][v], gt["fp"][v], gt["cc"][v])
        gm = (gt["imgs"][v] < 0.985).any(0).float(); rm = (r < 0.985).any(0).float()
        ious.append(float((gm * rm).sum() / ((gm + rm - gm * rm).sum() + 1e-6)))
    return float(np.mean(ious))


def run(uid, classid, noun, cfg, gp, eqm, vae, sam, cams, K=3):
    from eqm_sanity import nag_gd_sample
    td, elevs = ig.objaverse_topdown_idx(cfg, uid)
    it = ig.objaverse_item_uid(cfg, uid, src_idx=td)
    I = it["gt_images"][0].unsqueeze(0).to(P.DEV); M = (I < 0.985).any(1, keepdim=True).float()
    gt = {"imgs": it["gt_images"].to(P.DEV), "wv": it["world_view_transforms"].to(P.DEV),
          "fp": it["full_proj_transforms"].to(P.DEV), "cc": it["camera_centers"].to(P.DEV)}
    srcC = (gt["wv"][0], gt["fp"][0], gt["cc"][0])
    Isil = sil(I); Isb = (I < 0.985).any(1).float()[0]
    with torch.no_grad():
        ff = P.lift(gp, I)
    ff_iou = novel_iou(cfg, ff, gt)
    with torch.no_grad():
        imgs = P.vae_decode(vae, nag_gd_sample(eqm, torch.full((K,), classid, device=P.DEV), 250, cfg=1.5))
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
    GRID = [Rz(rz) @ Ry(az) @ Rx(el) for el in range(0, 180, 20) for az in range(0, 360, 30) for rz in (0, 90, 180, 270)]
    one = torch.tensor(1.0, device=P.DEV); zero = torch.zeros(3, device=P.DEV)
    best = None
    for u, sp in hyp:
        # coarse: rank rotations by scale/pos-invariant shape match
        scored = []
        for R0 in GRID:
            with torch.no_grad():
                r = P.render_view(cfg, transform_splats(sp, R0, one, zero), srcC[0], srcC[1], srcC[2]).unsqueeze(0)
                scored.append((shape_score(sil(r), Isil), R0))
        scored.sort(key=lambda x: -x[0])
        # for the top rotations, FREEZE R and fit only scale+translation, score reprojection IoU
        for _, R0 in scored[:12]:
            logs = torch.zeros((), device=P.DEV, requires_grad=True)
            t = torch.zeros(3, device=P.DEV, requires_grad=True)
            opt = torch.optim.Adam([logs, t], lr=0.03)
            for i in range(80):
                opt.zero_grad()
                spr = transform_splats(sp, R0, torch.exp(logs), t)
                r = P.render_view(cfg, spr, srcC[0], srcC[1], srcC[2]).unsqueeze(0)
                F.mse_loss(sil(r), Isil).backward(); opt.step()
            with torch.no_grad():
                spr = transform_splats(sp, R0, torch.exp(logs.detach()), t.detach())
            riou, _ = reproj_iou(cfg, spr, srcC, Isb)
            if best is None or riou > best[0]:
                best = (riou, R0, torch.exp(logs.detach()), t.detach(), u, sp)
    riou, R0, s, t, u, sp = best
    spr = transform_splats(sp, R0, s, t)
    niou = novel_iou(cfg, spr, gt)
    with torch.no_grad():
        recov = P.render_view(cfg, spr, srcC[0], srcC[1], srcC[2])
    # assemble video
    ff_tt = P.turntable_frames(cfg, ff, cams); iv_tt = P.turntable_frames(cfg, spr, cams)
    inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hy = (u.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rai = (recov.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, hy, f, iv, rai], 1) for f, iv in zip(ff_tt, iv_tt)]
    return dict(uid=uid, noun=noun, elev=elevs[td], reproj_iou=riou, novel_iou=niou, ff_iou=ff_iou), frames


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=999)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    OUT2 = a.out; os.makedirs(OUT2, exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae(); sam = P.SAM3()
    cams = P.build_turntable(cfg, num=48)
    rows = []
    for uid, cid, noun in CANDIDATES[a.start:a.start + a.count]:
        try:
            res = run(uid, cid, noun, cfg, gp, eqm, vae, sam, cams, a.K)
            if res is None:
                print("no hyp", uid); continue
            rec, frames = res
            tag = f"{noun.replace(' ', '')}_{uid[:8]}"
            imageio.mimsave(f"{OUT2}/{tag}.mp4", frames, fps=16, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT2}/{tag}_strip.png")
            rec["tag"] = tag; rows.append(rec)
            print(f"[{tag}] elev {rec['elev']:.0f}  REPROJ IoU {rec['reproj_iou']:.3f}  novel inv {rec['novel_iou']:.3f} ff {rec['ff_iou']:.3f}")
        except Exception as e:
            import traceback; traceback.print_exc(); print("skip", uid, e)
    rows.sort(key=lambda r: -r["reproj_iou"])
    json.dump(rows, open(f"{OUT2}/summary_{a.start}.json", "w"), indent=1)
    if rows:
        b = rows[0]
        print(f"\nBEST: {b['tag']}  reproj IoU {b['reproj_iou']:.3f}  (novel inv {b['novel_iou']:.3f} vs ff {b['ff_iou']:.3f})")


if __name__ == "__main__":
    main()
