"""FINV-SV homefield visual — invert REAL in-distribution control cars and render a
novel-view ORBIT of the reconstruction (input | orbit), analogous to pi-GAN's homefield
clips. This is the honest FINV homefield visual: it shows FINV reconstructing the correct
car in its own domain, WITHOUT relying on the control-camera pose anchor (which is the
gauge-broken quantity — see JOURNAL). Orbit is around the FINV-estimated input pose, so it
is anchor-free and always shows a car.

  python finv_control_orbit.py [--n 8]   # first n control cars

Writes report/videos/homefield_real/finv_<oid>.mp4 (input | orbit) + a recon PSNR summary.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, math, os, sys
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import homefield_gan  # noqa: F401 — patches E.make_c to the z-up camera family (real cars are z-up);
                      # reroutes finv_sv's pose search too, matching eg3d_control_zup for fairness
import eg3d_task_a as E
import finv_sv as FV
dev = "cuda"
R = f"{WT}/rebuttal/results"
INP = f"{R}/task_a_inputs_control"
OUTV = f"{WT}/rebuttal/report/videos/homefield_real"
os.makedirs(OUTV, exist_ok=True)


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def render(G, ws, c2w):
    c = np.concatenate([np.asarray(c2w, np.float32).reshape(16), E.INTR.numpy().reshape(9)])[None].astype(np.float32)
    with torch.no_grad():
        img = E.synth(G, ws, torch.from_numpy(c).to(dev))[0]
    return np.clip((img.permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=8)
    a = ap.parse_args()
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(dev)
    ids = sorted(os.listdir(INP))[:a.count]
    recon = {}
    for oid in ids:
        target = E.load_input(f"{INP}/{oid}/input.png")
        inp128 = np.asarray(Image.open(f"{INP}/{oid}/input.png").convert("RGB").resize((128, 128)))
        G = E.load_G()
        ws, c2w_hat, info = FV.finv_invert(G, lp, target)
        # reproj at estimated pose
        rep = render(G, ws, c2w_hat)
        recon[oid] = psnr(rep, inp128)
        # orbit: rotate object about vertical, anchored at estimated pose (anchor-free visual)
        el = math.acos(np.clip(c2w_hat[2, 3] / 1.7, -1, 1))
        frames = []
        for az in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            _, c2w = E.make_c(torch.tensor(float(az), device=dev),
                              torch.tensor(float(el), device=dev), G)
            frames.append(np.concatenate([inp128, render(G, ws, c2w[0].detach().cpu().numpy())], 1))
        imageio.mimsave(f"{OUTV}/finv_{oid}.mp4", frames, fps=8, codec="libx264",
                        output_params=["-pix_fmt", "yuv420p"])
        print(f"done {oid} recon={recon[oid]:.2f}", flush=True)
        del G; torch.cuda.empty_cache()
    json.dump({"mean_recon_psnr": float(np.mean(list(recon.values()))), "per_object": recon},
              open(f"{R}/homefield/finv_control_orbit_summary.json", "w"), indent=1)
    print(f"[finv homefield] mean recon PSNR = {np.mean(list(recon.values())):.2f} over {len(recon)}")


if __name__ == "__main__":
    main()
