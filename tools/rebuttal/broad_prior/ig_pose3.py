"""OPTION 1 v3 (first-principles): explain an UNNATURAL (top-down) observation with a NATURAL
object by searching a full SIMILARITY transform (rotation + scale + translation), not rotation
alone. EqM proposes clean natural hypotheses; each is lifted; we align it (R,s,t) so the
rendered result at the observation camera reproduces the observation. Feed-forward lifts the
top-down view directly and collapses. Verifiable: reprojection IoU (recovered @ input pose vs
observation) + 5th column shows it. The recovered 3D is a coherent natural object.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P
from ig import dino_feats
from ig_pose import Rx, Ry, Rz
from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from utils.abs_utils import symmetric_orthogonalization

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/pose3_out"


def transform_splats(sp, R, s, t):
    """Similarity transform about origin: xyz' = s*(R xyz) + t; scale gaussian sizes by s."""
    out = dict(sp)
    out["xyz"] = s * (sp["xyz"] @ R.T) + t
    q = matrix_to_quaternion(R).unsqueeze(0).expand(sp["rotation"].shape[0], -1)
    out["rotation"] = quaternion_raw_multiply(q, sp["rotation"])
    if "scaling" in sp:
        out["scaling"] = sp["scaling"] * s
    return out


def sil(x):
    return (1 - x.min(1, keepdim=True).values).clamp(0, 1)


def shape_score(sr, so):
    """Scale/position-invariant silhouette SHAPE match: crop each to its bbox, resize to 64x64,
    return IoU. Lets the coarse rotation search rank ORIENTATION independent of scale/position."""
    def norm(m):
        m = (m[0, 0] > 0.3)
        ys, xs = torch.where(m)
        if len(ys) < 5:
            return torch.zeros(64, 64, device=m.device)
        c = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].float()[None, None]
        return (F.interpolate(c, (64, 64), mode="bilinear", align_corners=False)[0, 0] > 0.3).float()
    a, b = norm(sr), norm(so)
    return float((a * b).sum() / ((a + b - a * b).sum() + 1e-6))


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
    ap.add_argument("--noun", default="teapot", help="SAM3 prompt for masking the EqM hypotheses")
    ap.add_argument("--K", type=int, default=4); ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--tt", type=int, default=54)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae(); sam = P.SAM3()
    cams = P.build_turntable(cfg, num=a.tt)
    from eqm_sanity import nag_gd_sample
    tag = a.tag or f"pose3_{a.uid[:8]}"

    td, elevs = ig.objaverse_topdown_idx(cfg, a.uid)
    it = ig.objaverse_item_uid(cfg, a.uid, src_idx=td)
    I = it["gt_images"][0].unsqueeze(0).to(P.DEV); M = (I < 0.985).any(1, keepdim=True).float()
    gt = {"imgs": it["gt_images"].to(P.DEV), "wv": it["world_view_transforms"].to(P.DEV),
          "fp": it["full_proj_transforms"].to(P.DEV), "cc": it["camera_centers"].to(P.DEV)}
    srcC = (gt["wv"][0], gt["fp"][0], gt["cc"][0])
    print(f"{tag}: top-down idx={td} elev={elevs[td]:.0f}")

    with torch.no_grad():
        ff = P.lift(gp, I)
    ff_iou = novel_iou(cfg, ff, gt)

    with torch.no_grad():
        imgs = P.vae_decode(vae, nag_gd_sample(eqm, torch.full((a.K,), a.classid, device=P.DEV), 250, cfg=1.5))
    hyp = []
    for k in range(a.K):
        u8 = (imgs[k].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        mk = sam.mask(u8, a.noun)
        if mk is None or mk.mean() < 0.02:
            continue
        u = P.center_on_white(imgs[k], mk, out=128)
        with torch.no_grad():
            hyp.append((u, P.lift(gp, u.unsqueeze(0))))
    print(f"{len(hyp)} clean hypotheses")

    Isil, Idino = sil(I), dino_feats(I)
    GRID = [Rz(rz) @ Ry(az) @ Rx(el) for el in range(0, 100, 25) for az in range(0, 360, 60) for rz in (0, 180)]
    best = None
    for u, sp in hyp:
        # coarse rotation grid ranked by SCALE-INVARIANT shape match (orientation only)
        scored = []
        one = torch.tensor(1.0, device=P.DEV); zero = torch.zeros(3, device=P.DEV)
        for R0 in GRID:
            with torch.no_grad():
                r = P.render_view(cfg, transform_splats(sp, R0, one, zero), srcC[0], srcC[1], srcC[2]).unsqueeze(0)
                scored.append((-shape_score(sil(r), Isil), R0))   # higher shape IoU = better
        scored.sort(key=lambda x: x[0])
        for _, R0 in scored[:4]:
            R = R0.clone().requires_grad_(True)
            logs = torch.zeros((), device=P.DEV, requires_grad=True)
            t = torch.zeros(3, device=P.DEV, requires_grad=True)
            opt = torch.optim.Adam([{"params": [R], "lr": 0.03}, {"params": [logs, t], "lr": 0.02}])
            for i in range(a.steps):
                opt.zero_grad()
                Rn = symmetric_orthogonalization(R.unsqueeze(0))[0]
                spr = transform_splats(sp, Rn, torch.exp(logs), t)
                r = P.render_view(cfg, spr, srcC[0], srcC[1], srcC[2]).unsqueeze(0)
                (F.mse_loss(sil(r), Isil) + 0.3 * (1 - (dino_feats(r) * Idino).sum(-1).mean())).backward()
                opt.step()
            with torch.no_grad():
                Rn = symmetric_orthogonalization(R.detach().unsqueeze(0))[0]
                spr = transform_splats(sp, Rn, torch.exp(logs.detach()), t.detach())
                r = P.render_view(cfg, spr, srcC[0], srcC[1], srcC[2]).unsqueeze(0)
                loss = float(F.mse_loss(sil(r), Isil) + 0.3 * (1 - (dino_feats(r) * Idino).sum(-1).mean()))
            if best is None or loss < best[0]:
                best = (loss, Rn, torch.exp(logs.detach()), t.detach(), u, sp)
    loss, Rn, s, t, u, sp = best
    spr = transform_splats(sp, Rn, s, t)
    with torch.no_grad():
        recov = P.render_view(cfg, spr, srcC[0], srcC[1], srcC[2])
    reproj_iou = float(((I < 0.985).any(1).float()[0] * (recov < 0.985).any(0).float()).sum() /
                       (((I < 0.985).any(1).float()[0] + (recov < 0.985).any(0).float()).clamp(0, 1).sum() + 1e-6))
    inv_iou = novel_iou(cfg, spr, gt)
    print(f"[{tag}] reprojection IoU (recovered @ input pose vs obs) {reproj_iou:.3f}  "
          f"| novel IoU ff {ff_iou:.3f} inv {inv_iou:.3f}  scale {float(s):.2f}")

    ff_tt = P.turntable_frames(cfg, ff, cams); iv_tt = P.turntable_frames(cfg, spr, cams)
    inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    hy = (u.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rai = (recov.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, hy, f, iv, rai], 1) for f, iv in zip(ff_tt, iv_tt)]
    imageio.mimsave(f"{OUT}/{tag}.mp4", frames, fps=16, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/{tag}_strip.png")
    print(f"saved {tag}.mp4 (obs | hypothesis | feed-forward 3D | Infer3D 3D | recovered @ input pose)")


if __name__ == "__main__":
    main()
