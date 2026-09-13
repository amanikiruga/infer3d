"""FINV-SV homefield input-view reconstruction on REAL control cars (fast, no orbit).
Matches EG3D's homefield montage: input | FINV reproj at its estimated pose. Saves per-car
reproj pngs + a 6-car montage + recon PSNR summary. z-up patch applied (real cars are z-up).
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os, sys
import numpy as np, torch
from PIL import Image, ImageDraw
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import homefield_gan  # noqa: F401 — z-up make_c patch (reroutes finv_sv too)
import eg3d_task_a as E
import finv_sv as FV
dev = "cuda"
R = f"{WT}/rebuttal/results"
INP = f"{R}/task_a_inputs_control"
OUTV = f"{WT}/rebuttal/report/videos/homefield_real"
os.makedirs(f"{OUTV}/img", exist_ok=True)


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def render(G, ws, c2w):
    c = np.concatenate([np.asarray(c2w, np.float32).reshape(16), E.INTR.numpy().reshape(9)])[None].astype(np.float32)
    with torch.no_grad():
        img = E.synth(G, ws, torch.from_numpy(c).to(dev))[0]
    return np.clip((img.permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)


def main():
    import lpips as lpips_lib
    lp = lpips_lib.LPIPS(net="vgg").to(dev)
    ids = sorted(os.listdir(INP))  # all 25 control cars (montage uses first 6)
    recon = {}
    reps = {}
    for oid in ids:
        target = E.load_input(f"{INP}/{oid}/input.png")
        inp128 = np.asarray(Image.open(f"{INP}/{oid}/input.png").convert("RGB").resize((128, 128)))
        G = E.load_G()
        ws, c2w_hat, info = FV.finv_invert(G, lp, target)
        rep = render(G, ws, c2w_hat)
        reps[oid] = rep; recon[oid] = psnr(rep, inp128)
        print(f"done {oid} recon={recon[oid]:.2f}", flush=True)
        del G; torch.cuda.empty_cache()
    # 6-car montage: input | FINV recon
    def lab(a, t, c):
        im = Image.fromarray(a.copy()); ImageDraw.Draw(im).text((3, 2), t, fill=c); return np.asarray(im)
    gap = np.full((128, 4, 3), 255, np.uint8)
    rows = []
    for oid in ids[:6]:
        inp = np.asarray(Image.open(f"{INP}/{oid}/input.png").convert("RGB").resize((128, 128)))
        rows.append(np.concatenate([lab(inp, "real input", (20, 20, 20)), gap,
                                    lab(reps[oid], "FINV-SV recon", (20, 60, 200))], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(f"{OUTV}/img/finv_recon_montage.png")
    json.dump({"mean_recon_psnr": float(np.mean(list(recon.values()))), "per_object": recon},
              open(f"{R}/homefield/finv_control_recon_summary.json", "w"), indent=1)
    print(f"[finv homefield recon] mean = {np.mean(list(recon.values())):.2f}, montage saved")


if __name__ == "__main__":
    main()
