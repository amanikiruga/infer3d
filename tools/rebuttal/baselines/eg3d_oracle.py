"""EG3D ORACLE-POSE NVS (give-GT-pose oracle, globally-calibrated frame).

Same idea as nfi_oracle: EG3D-world relates to NMR-world by a fixed rigid W, calibrated
from the CONTROL runs' estimated poses c2w_hat. For each SO(3) object: fix the input pose
to the GT pose in EG3D frame (radius-normalized to EG3D's training radius ~1.7 since EG3D's
generator is pose-conditioned), optimize w + PTI (no pose opt), render at GT targets.
gap(own-pose, oracle) = pose-estimation error, isolated. Fair: uses GT poses only.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
from scorer import score_views, bundle_targets
device = "cuda"; R = f"{WT}/rebuttal/results"
Fm = np.diag([1., -1, -1, 1]).astype(np.float32)


def calibrate_W():
    Ws = []
    for mp in sorted(glob.glob(f"{R}/task_a_runs/eg3d_control/*/meta.json")):
        oid = os.path.basename(os.path.dirname(mp)); d = json.load(open(mp))
        if "c2w_hat" not in d:
            continue
        c2w = np.array(d["c2w_hat"])
        c_in = np.load(f"{R}/task_a_inputs_control/{oid}/cams_absolute.npz")["input_v2w"].T
        Ws.append(c2w @ np.linalg.inv(c_in @ Fm))
    Ws = np.array(Ws)
    # SVD-project mean rotation to SO(3); reject rot-trace outliers (bad estimates)
    tr = np.array([np.trace(w[:3, :3]) for w in Ws]); med = np.median(tr)
    Wk = Ws[np.abs(tr - med) < 0.6]
    U, _, Vt = np.linalg.svd(Wk[:, :3, :3].mean(0)); Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1; Rp = U @ Vt
    W = np.eye(4, dtype=np.float32); W[:3, :3] = Rp; W[:3, 3] = Wk[:, :3, 3].mean(0)
    return W, len(Wk), len(Ws)


def renorm_radius(c2w, radius):
    c2w = c2w.copy(); c = c2w[:3, 3]; c2w[:3, 3] = c / (np.linalg.norm(c) + 1e-9) * radius
    return c2w


def invert_fixed(G, lp, target, c_fixed_25, main_steps=350, pti_steps=200):
    wa = E.w_avg(G)
    ws = wa.clone().detach().requires_grad_(True)
    c = torch.from_numpy(c_fixed_25)[None].to(device).float()
    opt = torch.optim.Adam([{"params": [ws], "lr": 8e-3}])
    for _ in range(main_steps):
        img = E.synth(G, ws, c)
        loss = lp(img, target).mean() + 0.1 * F.mse_loss(img, target)
        opt.zero_grad(); loss.backward(); opt.step()
    ws = ws.detach()
    for p in G.parameters():
        p.requires_grad_(True)
    optG = torch.optim.Adam(G.parameters(), lr=3e-4)
    for _ in range(pti_steps):
        out = G.synthesis(ws, c, noise_mode="const", force_fp32=True)
        img, img_raw = out["image"], out["image_raw"]
        tgt_raw = F.interpolate(target, size=img_raw.shape[-1], mode="area")
        loss = (lp(img, target).mean() + F.mse_loss(img, target)
                + lp(img_raw, tgt_raw).mean() + F.mse_loss(img_raw, tgt_raw))
        optG.zero_grad(); loss.backward(); optG.step()
    G.eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{R}/task_a_inputs")
    ap.add_argument("--out", default=f"{R}/task_a_oracle/eg3d")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(device)
    W, nk, nt = calibrate_W(); print(f"W_eg3d from {nk}/{nt}\n{np.round(W,3)}")
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    rows = {}
    for oid in ids:
        bdir = f"{args.bundles}/{oid}"; odir = f"{args.out}/{oid}"; rdir = f"{odir}/renders"
        os.makedirs(rdir, exist_ok=True)
        G = E.load_G()
        radius = G.rendering_kwargs.get("avg_camera_radius", 1.7)
        cabs = np.load(f"{bdir}/cams_absolute.npz"); targets_v2w = cabs["targets_v2w"]
        c_in_cv = cabs["input_v2w"].T.astype(np.float32)
        cam_gt = renorm_radius((W @ (c_in_cv @ Fm)).astype(np.float32), radius)
        c25 = np.concatenate([cam_gt.reshape(16), E.INTR.numpy().reshape(9)]).astype(np.float32)
        target = E.load_input(f"{bdir}/input.png")
        ws = invert_fixed(G, lp, target, c25)
        rpaths = []
        for i in range(targets_v2w.shape[0]):
            c_tgt_cv = targets_v2w[i].T.astype(np.float32)
            ct = renorm_radius((W @ (c_tgt_cv @ Fm)).astype(np.float32), radius)
            cc = np.concatenate([ct.reshape(16), E.INTR.numpy().reshape(9)]).astype(np.float32)
            with torch.no_grad():
                img = E.synth(G, ws, torch.from_numpy(cc)[None].to(device))[0]
            p = f"{rdir}/{i:02d}.png"; E.save_rgb(torch.clamp(img, -1, 1), p); rpaths.append(p)
        m = score_views(rpaths, bundle_targets(bdir)); rows[oid] = m
        json.dump(m, open(f"{odir}/metrics.json", "w"))
        print(f"done {oid} PSNR={m['PSNR']:.2f}")
        del G; torch.cuda.empty_cache()
    if rows:
        agg = {k: float(np.mean([r[k] for r in rows.values()])) for k in ["PSNR", "SSIM", "LPIPS"]}
        agg["n"] = len(rows); json.dump({"agg": agg, "per_obj": rows}, open(f"{args.out}/summary_shard{args.shard}.json", "w"), indent=1)
        print("AGG", agg)
    print("ALL DONE")


if __name__ == "__main__":
    main()
