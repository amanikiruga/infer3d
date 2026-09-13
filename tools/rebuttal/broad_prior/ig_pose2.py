"""OPTION 1 (clean): pose reasoning through search. EqM proposes a few CLEAN natural object
hypotheses (class-conditional, SAM3-masked); each is lifted to good 3D; we search only over
POSE so that the rotated hypothesis, rendered at the observation camera, explains an UNNATURAL
(top-down) observation. The lifter never sees the unnatural view -> no degeneracy. Verified on
held-out natural views (silhouette IoU) vs feed-forward, + qualitative turntable.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
from ig import rotate_splats, dino_feats
from ig_pose import Rx, Ry, Rz
from eqm_sanity import nag_gd_sample
from utils.abs_utils import symmetric_orthogonalization

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/pose2_out"


def sil(x):  # soft foreground (1 - whiteness), differentiable
    return (1 - x.min(1, keepdim=True).values).clamp(0, 1)


def novel_iou(cfg, splats, gt):
    ious = []
    for v in range(1, gt["wv"].shape[0]):
        with torch.no_grad():
            r = P.render_view(cfg, splats, gt["wv"][v], gt["fp"][v], gt["cc"][v])
        gm = (gt["imgs"][v] < 0.985).any(0).float(); rm = (r < 0.985).any(0).float()
        ious.append(float((gm * rm).sum() / ((gm + rm - gm * rm).sum() + 1e-6)))
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", required=True); ap.add_argument("--tag", default=None)
    ap.add_argument("--classid", type=int, default=849)
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--tt", type=int, default=54)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae(); sam = P.SAM3()
    cams = P.build_turntable(cfg, num=a.tt)
    tag = a.tag or f"pose2_{a.uid[:8]}"

    td, elevs = ig.objaverse_topdown_idx(cfg, a.uid)
    it = ig.objaverse_item_uid(cfg, a.uid, src_idx=td)
    I = it["gt_images"][0].unsqueeze(0).to(P.DEV); M = (I < 0.985).any(1, keepdim=True).float()
    gt = {"imgs": it["gt_images"].to(P.DEV), "wv": it["world_view_transforms"].to(P.DEV),
          "fp": it["full_proj_transforms"].to(P.DEV), "cc": it["camera_centers"].to(P.DEV)}
    src = (gt["wv"][0], gt["fp"][0], gt["cc"][0])
    print(f"{tag}: top-down idx={td} elev={elevs[td]:.0f}")

    with torch.no_grad():
        ff = P.lift(gp, I)                                       # feed-forward on top-down
    ff_iou = novel_iou(cfg, ff, gt)

    # EqM natural hypotheses -> SAM3 mask+center -> clean object-on-white -> lift (fixed 3D each)
    with torch.no_grad():
        lat = nag_gd_sample(eqm, torch.full((a.K,), a.classid, device=P.DEV), 250, cfg=1.5)
        imgs = P.vae_decode(vae, lat)
    hyp = []
    for k in range(a.K):
        u8 = (imgs[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mk = sam.mask(u8, "teapot")
        if mk is None or mk.mean() < 0.02:
            continue
        u = P.center_on_white(imgs[k], mk, out=128)
        with torch.no_grad():
            sp = P.lift(gp, u.unsqueeze(0))
        hyp.append((u, sp))
    print(f"{len(hyp)} clean hypotheses")

    Isil = sil(I); Idino = dino_feats(I)
    # DENSE global rotation grid (brute force) escapes the local minima the gradient search hits.
    GRID = [Rz(rz) @ Ry(az) @ Rx(el) for el in range(0, 100, 20) for az in range(0, 360, 45)
            for rz in (0, 90, 180, 270)]
    best = None
    for hi, (u, sp) in enumerate(hyp):
        # coarse: reprojection silhouette match at every grid rotation (cheap, no DINO)
        scored = []
        for R0 in GRID:
            with torch.no_grad():
                r = P.render_view(cfg, rotate_splats(sp, R0), src[0], src[1], src[2]).unsqueeze(0)
                scored.append((float(F.mse_loss(sil(r), Isil)), R0))
        scored.sort(key=lambda x: x[0])
        # refine the top few with gradient + DINO orientation cue
        for _, R0 in scored[:4]:
            R = R0.clone().requires_grad_(True); opt = torch.optim.Adam([R], lr=0.03)
            for t in range(a.steps):
                opt.zero_grad()
                spr = rotate_splats(sp, symmetric_orthogonalization(R.unsqueeze(0))[0])
                r = P.render_view(cfg, spr, src[0], src[1], src[2]).unsqueeze(0)
                (F.mse_loss(sil(r), Isil) + 0.3 * (1 - (dino_feats(r) * Idino).sum(-1).mean())).backward()
                opt.step()
            with torch.no_grad():
                Rf = symmetric_orthogonalization(R.detach().unsqueeze(0))[0]
                r = P.render_view(cfg, rotate_splats(sp, Rf), src[0], src[1], src[2]).unsqueeze(0)
                loss = float(F.mse_loss(sil(r), Isil) + 0.3 * (1 - (dino_feats(r) * Idino).sum(-1).mean()))
            if best is None or loss < best[0]:
                best = (loss, hi, Rf, u, sp)
    loss, hi, Rf, u, sp = best
    spr = rotate_splats(sp, Rf)
    inv_iou = novel_iou(cfg, spr, gt)
    # recovered tilt angle of the pose (how far from identity)
    ang = float(torch.acos(((torch.diagonal(Rf).sum() - 1) / 2).clamp(-1, 1)) * 180 / np.pi)
    print(f"[{tag}] novel-view silhouette IoU  feed-forward {ff_iou:.3f} | Infer3D {inv_iou:.3f}  "
          f"recovered pose tilt {ang:.0f}deg (obs elev {elevs[td]:.0f})")

    ff_tt = P.turntable_frames(cfg, ff, cams); iv_tt = P.turntable_frames(cfg, spr, cams)
    with torch.no_grad():                                  # recovered 3D rendered AT the input pose
        recov_at_input = P.render_view(cfg, spr, src[0], src[1], src[2])
    reproj_iou = float(((I < 0.985).any(1).float()[0] * (recov_at_input < 0.985).any(0).float()).sum() /
                       (((I < 0.985).any(1).float()[0] + (recov_at_input < 0.985).any(0).float()).clamp(0, 1).sum() + 1e-6))
    print(f"[{tag}] reprojection IoU (recovered 3D @ input pose vs observation) {reproj_iou:.3f}")
    inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hyp_img = (u.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rai = (recov_at_input.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, hyp_img, f, iv, rai], 1) for f, iv in zip(ff_tt, iv_tt)]
    imageio.mimsave(f"{OUT}/{tag}.mp4", frames, fps=18, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/{tag}_strip.png")
    print(f"saved {tag}.mp4 (top-down obs | natural hypothesis | feed-forward 3D | Infer3D 3D at recovered pose)")


if __name__ == "__main__":
    main()
