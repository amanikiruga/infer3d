"""Calibrate EG3D target-camera convention: invert the control object once,
then sweep flip F=diag(±1,±1,±1,1) x {A,B} rendering the GT target cameras,
score vs GT. Pick the max-PSNR convention (recon is fixed, so recon isn't the
confound)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import itertools, os, sys, json
import numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eg3d_task_a as E
from scorer import score_views, bundle_targets
import lpips as lpips_lib

R_EG3D = 1.7  # EG3D shapenet-cars camera radius


def renorm_lookat(c2w, radius=R_EG3D):
    """Keep camera orientation/direction from origin but place it at `radius`
    looking at the origin (EG3D's fixed convention), fixing the NMR/EG3D radius
    mismatch. c2w: 4x4 numpy (EG3D-convention cam2world after flip)."""
    pos = c2w[:3, 3].astype(np.float64)
    n = np.linalg.norm(pos)
    if n < 1e-6:
        return c2w
    pos = pos / n * radius
    fwd = -pos / np.linalg.norm(pos)           # look toward origin
    up = np.array([0, 1., 0])
    if abs(np.dot(fwd, up)) > 0.99:
        up = np.array([0, 0., 1.])
    right = np.cross(up, fwd); right /= np.linalg.norm(right)
    up2 = np.cross(fwd, right)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = right; m[:3, 1] = up2; m[:3, 2] = fwd; m[:3, 3] = pos
    return m

OBJ = sys.argv[1] if len(sys.argv) > 1 else sorted(os.listdir(f"{WT}/rebuttal/results/task_a_inputs_control"))[0]
BDIR = f"{WT}/rebuttal/results/task_a_inputs_control/{OBJ}"
dev = "cuda"


def main():
    G = E.load_G()
    lp = lpips_lib.LPIPS(net="vgg").to(dev)
    target = E.load_input(f"{BDIR}/input.png")
    ws, c2w_hat, _ = E.invert(G, lp, target, main_steps=200, pti_steps=0)
    cabs = np.load(f"{BDIR}/cams_absolute.npz")
    c_in_cv = cabs["input_v2w"].T.astype(np.float32)
    tv = cabs["targets_v2w"]
    tps = bundle_targets(BDIR)
    idxs = list(range(0, tv.shape[0], 4))
    os.makedirs("/tmp/eg3dcalib", exist_ok=True)

    def render(ct):
        c = np.concatenate([ct.reshape(16), E.INTR.numpy().reshape(9)]).astype(np.float32)
        with torch.no_grad():
            img = E.synth(G, ws, torch.from_numpy(c)[None].to(dev))[0]
        return torch.clamp(img, -1, 1)

    results = []
    for sx, sy, sz in itertools.product([1, -1], repeat=3):
        Fm = np.diag([sx, sy, sz, 1]).astype(np.float32)
        for name in ("A", "B"):
            rps = []
            ok = True
            for k, i in enumerate(idxs):
                ct_cv = tv[i].T.astype(np.float32)
                if name == "A":
                    ct = c2w_hat @ Fm @ np.linalg.inv(c_in_cv) @ ct_cv @ Fm
                else:
                    ct = c2w_hat @ np.linalg.inv(Fm @ c_in_cv) @ (Fm @ ct_cv)
                ct = renorm_lookat(ct)  # fix NMR/EG3D radius mismatch
                try:
                    img = render(ct)
                except Exception:
                    ok = False; break
                p = f"/tmp/eg3dcalib/{k:02d}.png"
                E.save_rgb(img, p); rps.append(p)
            if not ok:
                results.append((-1, f"F=({sx:+d},{sy:+d},{sz:+d}){name}", {})); continue
            m = score_views(rps, [tps[i] for i in idxs])
            results.append((m["PSNR"], f"F=({sx:+d},{sy:+d},{sz:+d}){name}", m))
    results.sort(reverse=True)
    for psnr, tag, m in results[:8]:
        print(f"  {tag}: PSNR {psnr:.2f}" + (f" SSIM {m['SSIM']:.3f}" if m else ""))
    print("BEST:", results[0][1], round(results[0][0], 2))


if __name__ == "__main__":
    main()
