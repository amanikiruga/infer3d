"""Infer3D inverse-graphics analysis-by-synthesis with an EqM (ImageNet EBM) prior.

Given a single observation I (real photo or Objaverse render), recover BOTH the latent of the
generative prior AND the camera pose so that the lifted, re-rendered 3D explains I:

    min_{z, R}  || M o ( Render( Rotate_R( Phi(dec(z)) ) ) - I ) ||_{MSE+LPIPS+DINO}
    with z kept on the EqM energy manifold via its equilibrium gradient (prior).

Infer3D machinery (multi-latent + multi-rotation start, prune-to-top-k, refine) but a FAST
schedule. EqM energy gradient used directly (no score distillation) for the prior term; data
term backprops through renderer -> frozen lifter -> VAE. DINOv2 feature loss bridges the
render-vs-real domain gap (paper's appearance-shift strategy).
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
from utils.abs_utils import symmetric_orthogonalization
import lpips as lpips_lib

_DINO = {}


def dino():
    if "m" not in _DINO:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(P.DEV).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _DINO["m"] = m
    return _DINO["m"]


def dino_feats(img01):
    x = F.interpolate(img01, (224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    f = dino().forward_features((x - mean) / std)["x_norm_patchtokens"]  # [B,256,384]
    return F.normalize(f, dim=-1)


def rotate_splats(sp, R):
    """Rotate object about world origin by R (3x3). Rotates xyz + gaussian orientation quats."""
    out = dict(sp)
    out["xyz"] = sp["xyz"] @ R.T
    q = matrix_to_quaternion(R).unsqueeze(0).expand(sp["rotation"].shape[0], -1)
    out["rotation"] = quaternion_raw_multiply(q, sp["rotation"])
    return out


def rand_rot(n, device):
    A = torch.randn(n, 3, 3, device=device)
    return symmetric_orthogonalization(A)


def load_obs(spec, cfg):
    from PIL import Image
    kind, arg = spec.split(":", 1)
    if kind == "png":
        im = np.asarray(Image.open(arg).convert("RGB").resize((128, 128))).astype(np.float32) / 255.
        I = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(P.DEV)
        M = (I < 0.985).any(1, keepdim=True).float()
        return I, M, os.path.splitext(os.path.basename(arg))[0], None
    if kind in ("objaverse", "objuid"):
        it = objaverse_item(cfg, int(arg)) if kind == "objaverse" else objaverse_item_uid(cfg, arg)
        I = it["gt_images"][0].unsqueeze(0).to(P.DEV)
        M = (I < 0.985).any(1, keepdim=True).float()
        gt = {"imgs": it["gt_images"].to(P.DEV), "wv": it["world_view_transforms"].to(P.DEV),
              "fp": it["full_proj_transforms"].to(P.DEV), "cc": it["camera_centers"].to(P.DEV)}
        oid = arg if kind == "objuid" else it.get("object_id", "")
        return I, M, f"obj_{oid[:10]}", gt
    raise ValueError(spec)


_DS = {}


def _ds(cfg):
    if "d" not in _DS:
        from splatter_image_datasets.objaverse import ObjaverseDataset
        _DS["d"] = ObjaverseDataset(cfg, "val")   # non-train -> deterministic arange view order
    return _DS["d"]


def objaverse_item(cfg, idx):
    from splatter_image_datasets.objaverse import ObjaverseDataset
    return ObjaverseDataset(cfg, "test")[idx]


def objaverse_item_uid(cfg, uid, src_idx=0):
    """Load a specific objaverse uid's item directly (bypasses the split), reusing the
    dataset's own camera-conversion. src_idx picks which of the 12 views is the SOURCE
    (observation); all cameras are made relative to it. src_idx>0 lets us observe an
    unnatural (e.g. top-down) view and verify recovery on the remaining natural views."""
    import glob as _g
    ds = _ds(cfg)
    paths = sorted(_g.glob(f"{ds.root_dir}/{uid}/*.png"))
    it = ds.load_imgs_and_convert_cameras(paths, len(paths))
    if src_idx != 0:
        n = it["gt_images"].shape[0]
        order = [src_idx] + [i for i in range(n) if i != src_idx]
        it = {k: (v[order] if isinstance(v, torch.Tensor) and v.shape[0] == n else v) for k, v in it.items()}
    it = ds.make_poses_relative_to_first(it)
    it["source_cv2wT_quat"] = ds.get_source_cw2wT(it["view_to_world_transforms"])
    it["object_id"] = uid
    return it


def objaverse_topdown_idx(cfg, uid):
    """Index of the most top-down (highest-elevation camera) among the 12 views + per-view
    elevation in degrees."""
    import glob as _g
    ds = _ds(cfg)
    paths = sorted(_g.glob(f"{ds.root_dir}/{uid}/*.png"))
    it = ds.load_imgs_and_convert_cameras(paths, len(paths))
    cc = it["camera_centers"]                       # [N,3], object at origin
    elev = torch.asin((cc[:, 2] / (cc.norm(dim=1) + 1e-8)).clamp(-1, 1)) * 180 / np.pi
    return int(elev.argmax()), elev.tolist()


def invert(I, M, gp, eqm, vae, cams, lp, cfg, use_dino=True, K=8, nrot=2, steps1=70, steps2=110,
           keep=3, eta=0.0017, lam=250., pose=True, warm=True):
    P0 = K * nrot
    z = torch.randn(P0, 4, 32, 32, device=P.DEV)
    if warm:
        with torch.no_grad():
            z[0] = P.vae_encode(vae, F.interpolate(I, (256, 256), mode="bilinear", align_corners=False))[0]
    R = torch.eye(3, device=P.DEV).unsqueeze(0).repeat(P0, 1, 1)
    if pose:
        R[1:] = rand_rot(P0 - 1, P.DEV)          # multi-rotation start (particle 0 = identity+warm)
    y = torch.full((P0,), 1000, device=P.DEV)
    Iexp = I.expand(P0, -1, -1, -1); Mexp = M.expand(P0, -1, -1, -1)
    Idino = dino_feats(I * M + (1 - M)) if use_dino else None

    def forward(zc, Rc, idx):
        idxt = torch.tensor(idx, device=P.DEV)
        img = P.vae_decode(vae, zc[idxt])                      # decode only alive particles
        ui = F.interpolate(img, (128, 128), mode="bilinear") * Mexp[idxt] + (1 - Mexp[idxt])
        losses, rends = [], []
        for j, k in enumerate(idx):
            sp = P.lift(gp, ui[j:j+1])
            spr = rotate_splats(sp, Rc[k]) if pose else sp
            r = P.render_view(cfg, spr, cams[0][0], cams[1][0], cams[2][0])
            rends.append(r)
            L = ((Mexp[k] * (r - Iexp[k])) ** 2).mean() + 0.3 * lp(r.unsqueeze(0) * 2 - 1, Iexp[k:k+1] * 2 - 1).mean()
            if use_dino:
                L = L + 0.5 * (1 - (dino_feats((r * Mexp[k] + (1 - Mexp[k])).unsqueeze(0)) * Idino).sum(-1).mean())
            losses.append(L)
        return torch.stack(losses), rends

    alive = list(range(P0))
    for stage, nsteps in [(1, steps1), (2, steps2)]:
        zp = z.clone().detach().requires_grad_(True)
        Rp = R.clone().detach().requires_grad_(pose)
        params = [zp] + ([Rp] if pose else [])
        opt = torch.optim.Adam(params, lr=0.04)
        for t in range(nsteps):
            opt.zero_grad()
            per, _ = forward(zp, Rp, alive)
            per.sum().backward()
            opt.step()
            with torch.no_grad():
                g = P.eqm_grad(eqm, zp.detach()[alive], y[alive])
                zp.data[alive] += eta * g
                if pose:
                    Rp.data[alive] = symmetric_orthogonalization(Rp.data[alive])
        z, R = zp.detach(), Rp.detach()
        with torch.no_grad():
            per, _ = forward(z, R, alive)
        order = [alive[i] for i in torch.argsort(per).tolist()]
        if stage == 1:
            alive = order[:keep]
    best = order[0]
    with torch.no_grad():
        img = P.vae_decode(vae, z[best:best+1])
        recon = F.interpolate(img, (128, 128), mode="bilinear")[0]
        u = recon * M[0] + (1 - M[0])
        sp = P.lift(gp, u.unsqueeze(0))
        spr = rotate_splats(sp, R[best]) if pose else sp
    return spr, sp, recon, R[best]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--tt", type=int, default=60)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--nrot", type=int, default=2)
    ap.add_argument("--steps1", type=int, default=70)
    ap.add_argument("--steps2", type=int, default=110)
    a = ap.parse_args()
    import imageio.v2 as imageio
    from PIL import Image
    OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/ig_out"
    os.makedirs(OUT, exist_ok=True)
    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae()
    cams = P.build_turntable(cfg, num=a.tt)
    lp = lpips_lib.LPIPS(net="vgg").to(P.DEV).eval()
    I, M, tag, gt = load_obs(a.obs, cfg); tag = a.tag or tag
    print("observation", tag, "dino", not a.no_dino, "pose", not a.no_pose)
    spr, sp_canon, recon, Rbest = invert(I, M, gp, eqm, vae, cams, lp, cfg,
                                          use_dino=not a.no_dino, pose=not a.no_pose,
                                          K=a.K, nrot=a.nrot, steps1=a.steps1, steps2=a.steps2)
    if gt is not None:
        ps = []
        for v in range(1, gt["wv"].shape[0]):
            with torch.no_grad():
                rr = P.render_view(cfg, sp_canon, gt["wv"][v], gt["fp"][v], gt["cc"][v])
            ps.append(float(-10 * torch.log10(((rr - gt["imgs"][v]) ** 2).mean() + 1e-12)))
        print(f"[{tag}] inverted novel-view PSNR {np.mean(ps):.2f}")
    tt = P.turntable_frames(cfg, sp_canon, cams)
    inp = (I[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rec = (recon.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [np.concatenate([inp, rec, f], 1) for f in tt]
    imageio.mimsave(f"{OUT}/ig_{tag}.mp4", frames, fps=20, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/ig_{tag}_strip.png")
    print(f"saved ig_{tag}.mp4  (input | inverted recon | recovered 3D turntable)")
