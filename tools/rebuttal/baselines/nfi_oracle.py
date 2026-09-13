"""NFI ORACLE-POSE NVS (give-GT-pose oracle, globally calibrated frame).

NFI is trained with pose supervision on SRN cars, so its world frame relates to the
NMR benchmark frame by ONE fixed rigid transform W (empirically ~90 deg about x:
NFI z-up vs NMR y-up). We calibrate W ONCE from the CONTROL runs (ID pose, where
NFI's own PnP estimate is reliable), then for each object:
  - re-invert with the input pose FIXED to the GT pose  cam_gt = W @ (c_in_cv @ FLIP)
    (optimize latent only, same 15-aug LPIPS objective / steps as own-pose),
  - render at the GT target cameras  W @ (c_tgt_cv @ FLIP),
  - score vs GT target images (same scorer).
This hands NFI the pose it fails to estimate under SO(3). gap(own-pose, oracle)=pose error.
No per-object ICP -> no symmetric-car branch ambiguity. W uses GT geometry/poses only
(never target images) -> fair oracle.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, glob, json, os, sys
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import nfi_task_a as N
from scorer import score_views, bundle_targets
dev = "cuda"; R = f"{WT}/rebuttal/results"


def calibrate_W(max_trace_dev=0.6):
    """Robust W = SVD-projected mean of per-control-object W_i = cam_est @ inv(c_in@FLIP).
    Reject outliers whose rotation trace deviates from the inlier median (bad PnP)."""
    Ws = []
    for mp in sorted(glob.glob(f"{R}/task_a_runs/nfi_control/*/meta.json")):
        oid = os.path.basename(os.path.dirname(mp)); d = json.load(open(mp))
        if "cam_est_300" not in d or d.get("pnp_err_300", 1) > 0.05:
            continue
        cam_est = np.array(d["cam_est_300"])
        c_in_cv = np.load(f"{R}/task_a_inputs_control/{oid}/cams_absolute.npz")["input_v2w"].T
        Ws.append(cam_est @ np.linalg.inv(c_in_cv @ N.FLIP))
    Ws = np.array(Ws)
    traces = np.array([np.trace(w[:3, :3]) for w in Ws])
    med = np.median(traces)
    keep = np.abs(traces - med) < max_trace_dev
    Wk = Ws[keep]
    Rmean = Wk[:, :3, :3].mean(0)
    U, _, Vt = np.linalg.svd(Rmean)
    Rproj = U @ Vt
    if np.linalg.det(Rproj) < 0:
        U[:, -1] *= -1; Rproj = U @ Vt
    W = np.eye(4); W[:3, :3] = Rproj; W[:3, 3] = Wk[:, :3, 3].mean(0)
    return W.astype(np.float32), int(keep.sum()), len(Ws)


def invert_fixed(G, enc, lp, target_img, steps, focal, cam_fixed):
    with torch.no_grad():
        _, _, w = enc(target_img.permute(0, 3, 1, 2))
    z = (w.expand(-1, G.mapping_network.get_average_w().shape[1], -1).contiguous() / 5).requires_grad_()
    opt = torch.optim.Adam([z], lr=2e-3, betas=(0.9, 0.95))
    cam = cam_fixed.to(dev).float(); foc = torch.tensor([focal], device=dev).float()
    for it in range(steps):
        rgb, _, _, _ = N.render(G, N.RES, N.RES, cam, foc, z * 5)
        pred = rgb.permute(0, 3, 1, 2); targ = target_img[..., :3].permute(0, 3, 1, 2)
        cat = torch.cat((pred, targ), 1).unsqueeze(1).expand(-1, 15, -1, -1, -1).contiguous().flatten(0, 1)
        cat = N.augment_impl(cat, 1.0)
        pa = torch.cat((pred, cat[:, :3]), 0); ta = torch.cat((targ, cat[:, 3:]), 0)
        loss = lp(pa, ta).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return (z * 5).detach(), foc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{R}/task_a_inputs")
    ap.add_argument("--out", default=f"{R}/task_a_oracle/nfi")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    W, nk, nt = calibrate_W(); print(f"calibrated W from {nk}/{nt} control objs\n{np.round(W,3)}")
    G, enc, lp = N.load_models()
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    rows = {}
    for oid in ids:
        bdir = f"{args.bundles}/{oid}"; odir = f"{args.out}/{oid}"; rdir = f"{odir}/renders"
        os.makedirs(rdir, exist_ok=True)
        cams = np.load(f"{bdir}/cams_relative.npz"); fov = float(cams["fov_deg"])
        focal = (N.RES / 2) / np.tan(np.deg2rad(fov) / 2) / N.RES
        cabs = np.load(f"{bdir}/cams_absolute.npz"); targets_v2w = cabs["targets_v2w"]
        c_in_cv = cabs["input_v2w"].T.astype(np.float32)
        cam_gt = (W @ (c_in_cv @ N.FLIP)).astype(np.float32)
        target_img = N.load_input(f"{bdir}/input.png")
        torch.manual_seed(0); np.random.seed(0)
        ws, foc = invert_fixed(G, enc, lp, target_img, args.steps, focal, torch.from_numpy(cam_gt)[None])
        rpaths = []
        for i in range(targets_v2w.shape[0]):
            c_tgt_cv = targets_v2w[i].T.astype(np.float32)
            ct = (W @ (c_tgt_cv @ N.FLIP)).astype(np.float32)
            with torch.no_grad():
                rgb, _, _, _ = N.render(G, N.RES, N.RES, torch.from_numpy(ct)[None].to(dev), foc, ws, randomize=False)
            p = f"{rdir}/{i:02d}.png"; N.save_rgb(torch.clamp(rgb[0], -1, 1), p); rpaths.append(p)
        m = score_views(rpaths, bundle_targets(bdir)); rows[oid] = m
        json.dump(m, open(f"{odir}/metrics.json", "w"))
        print(f"done {oid} PSNR={m['PSNR']:.2f} SSIM={m['SSIM']:.3f} LPIPS={m['LPIPS']:.3f}")
    if rows:
        agg = {k: float(np.mean([r[k] for r in rows.values()])) for k in ["PSNR", "SSIM", "LPIPS"]}
        agg["n"] = len(rows)
        json.dump({"agg": agg, "per_obj": rows}, open(f"{args.out}/summary_shard{args.shard}.json", "w"), indent=1)
        print("AGG", agg)
    print("ALL DONE")


if __name__ == "__main__":
    main()
