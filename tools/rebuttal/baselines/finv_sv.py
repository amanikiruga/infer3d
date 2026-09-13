"""FINV-SV: FINV (Nguyen-Phuoc et al., 3DV'24) multi-start particle inversion with
pruning, adapted to the SINGLE-VIEW reconstruction case (the AC's explicit ask:
"FINV ... also employs multi-start particle inversion with pruning").

FINV's pipeline = sample N latent particles -> optimize each (latent+pose) -> FILTER
(prune) the worst by a consistency signal -> refine survivors -> keep best -> PTI.
FINV's original filtering signal is CROSS-VIEW consistency (multi-view input). In the
single-image setting the faithful analog is INPUT-image consistency (each particle's
render vs the single input). We instantiate on the same EG3D ShapeNet-cars generator
FINV builds on (GET3D/EG3D family), so the machinery is faithful; only the filtering
supervision is the single-view analog. This tests whether multi-start+pruning — the
ingredient the AC cites — is what closes the OOD gap.

Outputs per object (both tasks): renders/ (NVS at GT targets, own-pose), geometry .npy
(ICP-Chamfer), reproj, meta. Run oracle NVS + orbit separately (reuse eg3d tooling).
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
device = "cuda"
FLIP = np.diag([1., -1., -1., 1.]).astype(np.float32)  # EG3D convention calibrated on control


def sample_particles(G, n, restarts, seed=0):
    """N particles = (w from a random z, azimuth restart). Spreads seeds x poses."""
    g = torch.Generator(device=device).manual_seed(seed)
    parts = []
    for i in range(n):
        z = torch.randn(1, G.z_dim, device=device, generator=g)
        az0 = restarts[i % len(restarts)]
        c0, _ = E.make_c(torch.tensor(float(az0), device=device),
                         torch.tensor(float(np.pi / 2), device=device), G)
        with torch.no_grad():
            w = G.mapping(z, c0, truncation_psi=0.7)
        parts.append({"w": w.clone(), "az": float(az0), "el": float(np.pi / 2)})
    return parts


def opt_particle(G, lp, target, p, steps, opt_pose=True):
    """Optimize one particle's latent (+pose) vs the input image; return final loss."""
    ws = p["w"].clone().detach().requires_grad_(True)
    az = torch.tensor(p["az"], device=device, requires_grad=opt_pose)
    el = torch.tensor(p["el"], device=device, requires_grad=opt_pose)
    params = [{"params": [ws], "lr": 8e-3}]
    if opt_pose:
        params.append({"params": [az, el], "lr": 7e-3})
    opt = torch.optim.Adam(params)
    for _ in range(steps):
        c, _ = E.make_c(az, el, G)
        img = E.synth(G, ws, c)
        loss = lp(img, target).mean() + 0.1 * F.mse_loss(img, target)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        c, _ = E.make_c(az, el, G)
        fl = lp(E.synth(G, ws, c), target).mean().item()
    p.update({"w": ws.detach(), "az": az.detach().item(), "el": el.detach().item(), "loss": fl})
    return fl


def finv_invert(G, lp, target, n_particles=16, restarts=(0., np.pi / 2, np.pi, 3 * np.pi / 2),
                stage1_steps=80, keep1=4, stage2_steps=250, pti_steps=200):
    parts = sample_particles(G, n_particles, restarts)
    # Stage 1: brief opt of every particle, then FILTER to top-keep1 by input consistency
    for p in parts:
        opt_particle(G, lp, target, p, stage1_steps)
    parts.sort(key=lambda p: p["loss"])
    survivors = parts[:keep1]
    filt_losses = [round(p["loss"], 4) for p in parts]
    # Stage 2: refine survivors longer, prune to the single best
    for p in survivors:
        opt_particle(G, lp, target, p, stage2_steps)
    survivors.sort(key=lambda p: p["loss"])
    best = survivors[0]
    ws = best["w"]; az_f = torch.tensor(best["az"], device=device); el_f = torch.tensor(best["el"], device=device)
    # PTI on the winning particle (fine-tune G)
    c_fixed, c2w_hat = E.make_c(az_f, el_f, G); c_fixed = c_fixed.detach()
    for p in G.parameters():
        p.requires_grad_(True)
    optG = torch.optim.Adam(G.parameters(), lr=3e-4)
    for _ in range(pti_steps):
        out = G.synthesis(ws, c_fixed, noise_mode="const", force_fp32=True)
        img, img_raw = out["image"], out["image_raw"]
        tgt_raw = F.interpolate(target, size=img_raw.shape[-1], mode="area")
        loss = (lp(img, target).mean() + F.mse_loss(img, target)
                + lp(img_raw, tgt_raw).mean() + F.mse_loss(img_raw, tgt_raw))
        optG.zero_grad(); loss.backward(); optG.step()
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return ws, c2w_hat.detach()[0].cpu().numpy(), {"filt_losses": filt_losses,
            "best_loss": best["loss"], "survivor_az": [round(p["az"], 3) for p in survivors]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_runs/finv")
    ap.add_argument("--n_particles", type=int, default=16)
    ap.add_argument("--realcars", action="store_true")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(device)
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    inp = "rgb_white_128.png" if args.realcars else "input.png"
    for oid in ids:
        bdir = f"{args.bundles}/{oid}"; odir = f"{args.out}/{oid}"
        os.makedirs(f"{odir}/renders", exist_ok=True)
        G = E.load_G()
        target = E.load_input(f"{bdir}/{inp}")
        ws, c2w_hat, info = finv_invert(G, lp, target, n_particles=args.n_particles)
        # render at GT target cameras, anchored at estimated pose (same convention as eg3d_task_a)
        cabs = np.load(f"{bdir}/cams_absolute.npz")
        c_in_cv = cabs["input_v2w"].T.astype(np.float32); targets_v2w = cabs["targets_v2w"]
        c_hat = np.eye(4, dtype=np.float32); c_hat[:3, :3] = c2w_hat[:3, :3]; c_hat[:3, 3] = c2w_hat[:3, 3]
        for i in range(targets_v2w.shape[0]):
            c_tgt_cv = targets_v2w[i].T.astype(np.float32)
            ct = c_hat @ FLIP @ np.linalg.inv(c_in_cv) @ c_tgt_cv @ FLIP
            cc = np.concatenate([ct.reshape(16), E.INTR.numpy().reshape(9)])[None].astype(np.float32)
            with torch.no_grad():
                img = E.synth(G, ws, torch.from_numpy(cc).to(device))[0]
            E.save_rgb(torch.clamp(img, -1, 1), f"{odir}/renders/{i:02d}.png")
        json.dump(info, open(f"{odir}/finv_info.json", "w"))
        print(f"done {oid} best_loss={info['best_loss']:.4f} survivors_az={info['survivor_az']}")
        del G; torch.cuda.empty_cache()
    print("ALL DONE")


if __name__ == "__main__":
    main()
