"""Calibrate the CV<->OpenGL camera-convention flip for Task-A novel-view eval.

On the side-top CONTROL object (where NFI reconstructs well, so recon is not the
confound), enumerate diagonal flips F=diag(sx,sy,sz,1) and composition variants,
render target cameras, and score vs GT targets. The correct convention maximizes
PSNR. One-time; result is hardcoded thereafter.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import itertools
import os
import sys

import numpy as np
import torch

WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import nfi_task_a as N
from scorer import score_views, bundle_targets

OBJ = sys.argv[1] if len(sys.argv) > 1 else "1c1a3dc04b6f1f8fd8162cce87567b4"
BDIR = f"{WT}/rebuttal/results/task_a_inputs_control/{OBJ}"
device = "cuda"


def main():
    G, enc, lp = N.load_models()
    cams = np.load(f"{BDIR}/cams_relative.npz")
    fov = float(cams["fov_deg"])
    focal_guess = (N.RES / 2) / np.tan(np.deg2rad(fov) / 2) / N.RES
    target_img = N.load_input(f"{BDIR}/input.png")
    torch.manual_seed(0); np.random.seed(0)
    ws, cam, foc, _ = N.invert(G, enc, lp, target_img, 30, focal_guess)
    cam_np = cam[0].cpu().numpy()
    cabs = np.load(f"{BDIR}/cams_absolute.npz")
    c_in_cv = cabs["input_v2w"].T.astype(np.float32)
    tv = cabs["targets_v2w"]
    tps = bundle_targets(BDIR)
    idxs = list(range(0, tv.shape[0], 3))  # subsample for speed

    def render_set(compose_fn):
        rpaths = []
        os.makedirs("/tmp/calib", exist_ok=True)
        for k, i in enumerate(idxs):
            ct = compose_fn(cam_np, c_in_cv, tv[i].T.astype(np.float32))
            ct_t = torch.from_numpy(ct.astype(np.float32))[None].to(device)
            with torch.no_grad():
                rgb, _, _, _ = N.render(G, N.RES, N.RES, ct_t, foc, ws, randomize=False)
            p = f"/tmp/calib/{k:02d}.png"
            N.save_rgb(torch.clamp(rgb[0], -1, 1), p)
            rpaths.append(p)
        return rpaths

    results = []
    for sx, sy, sz in itertools.product([1, -1], repeat=3):
        Fc = np.diag([sx, sy, sz, 1]).astype(np.float32)
        # variant A: C_hat @ F @ inv(C_in) @ C_tgt @ F
        def cA(ch, cin, ct, Fc=Fc):
            return ch @ Fc @ np.linalg.inv(cin) @ ct @ Fc
        # variant B: C_hat @ inv(F@C_in) @ (F@C_tgt)  == same as A actually; use conj other side
        def cB(ch, cin, ct, Fc=Fc):
            return ch @ np.linalg.inv(Fc @ cin) @ (Fc @ ct)
        for name, fn in [("A", cA), ("B", cB)]:
            try:
                rp = render_set(fn)
                m = score_views(rp, [tps[i] for i in idxs])
            except Exception as e:
                results.append((-1.0, f"F=({sx:+d},{sy:+d},{sz:+d}) {name} FAIL({type(e).__name__})", {"SSIM": 0, "LPIPS": 1}))
                continue
            results.append((m["PSNR"], f"F=({sx:+d},{sy:+d},{sz:+d}) {name}", m))
    results.sort(reverse=True)
    print("\n=== convention calibration (control object, recon good) ===")
    for psnr, tag, m in results[:8]:
        print(f"  {tag}: PSNR {psnr:.2f} SSIM {m['SSIM']:.3f} LPIPS {m['LPIPS']:.3f}")
    print(f"\nBEST: {results[0][1]}  PSNR {results[0][0]:.2f}")


if __name__ == "__main__":
    main()
