"""Test EG3D with GT input pose: place NMR cameras into EG3D frame preserving
full orientation (incl. roll) at radius 1.7. If reproj matches input, gauge/flip
are correct and we can run EG3D-PTI† with GT poses (fair upper bound)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import sys, os, json
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
VIZ = f"{WT}/rebuttal/viz"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
import lpips as lpips_lib
dev = "cuda"
R_EG3D = 1.7
FLIPS = {"nfi": np.diag([1., -1, -1, 1]), "id": np.eye(4), "yz": np.diag([-1., 1, -1, 1])}


def nmr_to_eg3d(c2w_cv, F, radius=R_EG3D):
    gl = (c2w_cv @ F).astype(np.float64)        # CV->GL camera-axis flip
    R = gl[:3, :3].copy(); pos = gl[:3, 3].copy()
    n = np.linalg.norm(pos)
    if n > 1e-6:
        pos = pos / n * radius                   # fix radius, keep orientation (roll preserved)
    m = np.eye(4, dtype=np.float32); m[:3, :3] = R; m[:3, 3] = pos
    return m


def main():
    obj = sys.argv[1] if len(sys.argv) > 1 else sorted(os.listdir(f"{WT}/rebuttal/results/task_a_inputs_control"))[0]
    bdir = f"{WT}/rebuttal/results/task_a_inputs_control/{obj}"
    G = E.load_G(); lp = lpips_lib.LPIPS(net="vgg").to(dev)
    target = E.load_input(f"{bdir}/input.png")
    cabs = np.load(f"{bdir}/cams_absolute.npz")
    c_in_cv = cabs["input_v2w"].T.astype(np.float32)
    for fname, F in FLIPS.items():
        c_hat = nmr_to_eg3d(c_in_cv, F)
        # w-projection at FIXED GT pose (no pose opt), few steps
        wa = E.w_avg(G)
        ws = wa.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([ws], lr=8e-3)
        c = np.concatenate([c_hat.reshape(16), E.INTR.numpy().reshape(9)]).astype(np.float32)
        c_t = torch.from_numpy(c)[None].to(dev)
        for _ in range(120):
            img = E.synth(G, ws, c_t)
            loss = lp(img, target).mean() + 0.1 * torch.mean((img - target) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            rep = E.synth(G, ws, c_t)
            l = lp(rep, target).mean().item()
        E.save_rgb(torch.clamp(rep[0], -1, 1), f"{_EXT}/splatter-image-rebuttal/rebuttal/viz/eg3d_reproj_{fname}.png")
        print(f"flip={fname}: reproj LPIPS vs input = {l:.3f}")
    # save input for comparison
    import shutil; shutil.copy(f"{bdir}/input.png", f"{_EXT}/splatter-image-rebuttal/rebuttal/viz/eg3d_input.png")
    print(f"saved {VIZ}/eg3d_reproj_*.png and {VIZ}/eg3d_input.png")


if __name__ == "__main__":
    main()
