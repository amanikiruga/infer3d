"""pi-GAN homefield on REAL CARLA images (the actual data its generator was trained on).

pi-GAN's released car model is trained on the GRAF CARLA renders. That IS its home
distribution — so we invert real CARLA images (NOT generator samples) and report:
  - input-view reconstruction PSNR (recon vs the real input) — does inversion fit real data?
  - a qualitative novel-view orbit (CARLA has NO per-image camera labels, so there is no
    novel-view GT and no NVS metric even in-domain — a property of pi-GAN's training data,
    which is exactly why the paper reports no NVS PSNR either).
Contrast target: on ShapeNet/RealCars (OOD) pi-GAN inverts to blobs; here on its own CARLA
domain it should reconstruct clean cars — isolating domain gap from a broken model.

  python homefield_pigan_real.py [--n 15] [--iters 700]
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, math, os, sys
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image
WT = f"{_EXT}/splatter-image-rebuttal"
RB = f"{_EXT}/rebuttal_infer3d_baselines"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import pigan_geom as P
dev = "cuda"
HF = f"{WT}/rebuttal/results/homefield"
V = math.pi / 2


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)


def render(G, wf, wp, h, v):
    o = dict(P.OPT); o["h_mean"] = h; o["v_mean"] = v
    o["num_steps"] = 36; o["lock_view_dependence"] = True
    with torch.no_grad():
        try:
            px, _ = G.staged_forward_with_frequencies(wf, wp, max_batch_size=400000, **o)
        except Exception:
            px, _ = G.forward_with_frequencies(wf, wp, **o)
    return np.clip((px[0].permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)


def invert_hv(G, target, iters):
    with torch.no_grad():
        z = torch.randn(10000, 256, device=dev)
        wf, wp = G.siren.mapping_network(z)
        wf = wf.mean(0, keepdim=True); wp = wp.mean(0, keepdim=True)
    off_f = torch.zeros_like(wf, requires_grad=True)
    off_p = torch.zeros_like(wp, requires_grad=True)
    h = torch.tensor(math.pi / 2, device=dev, requires_grad=True)
    v = torch.tensor(math.pi / 2, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [off_f, off_p], "lr": 1e-2, "weight_decay": 1e-4},
                            {"params": [h, v], "lr": 2e-3}])
    for i in range(iters):
        n = 0.03 * torch.randn_like(wf) * (iters - i) / iters
        o = dict(P.OPT); o["h_mean"] = h; o["v_mean"] = v
        frame, _ = G.forward_with_frequencies(wf + n + off_f, wp + n + off_p, **o)
        loss = torch.mean((frame - target) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    return (wf + off_f).detach(), (wp + off_p).detach(), h.item(), v.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--iters", type=int, default=700)
    a = ap.parse_args()
    G = P.load_gen()
    # evenly spaced real CARLA images (fixed indices -> reproducible)
    idx = [int(round(k)) for k in np.linspace(0, 9000, a.n)]
    recon = {}
    for j, ii in enumerate(idx):
        oid = f"r{j:02d}"; od = f"{HF}/pigan_real/{oid}"; os.makedirs(od, exist_ok=True)
        if os.path.exists(f"{od}/meta.json"):
            recon[oid] = json.load(open(f"{od}/meta.json"))["recon_psnr"]; continue
        src = f"{RB}/datasets/{ii:06d}.png"
        inp128 = np.asarray(Image.open(src).convert("RGB").resize((128, 128)))
        Image.fromarray(inp128).save(f"{od}/input.png")
        target = P.load_img(src)
        wf, wp, h_hat, v_hat = invert_hv(G, target, a.iters)
        rep = render(G, wf, wp, h_hat, v_hat)
        Image.fromarray(rep).save(f"{od}/reproj.png")
        rp = psnr(rep, inp128)
        # qualitative novel-view orbit (no GT to score against)
        frames = [render(G, wf, wp, h_hat + dh, v_hat)
                  for dh in np.linspace(-math.pi, math.pi, 24, endpoint=False)]
        imageio.mimsave(f"{od}/orbit.mp4", [np.concatenate([inp128, f], 1) for f in frames],
                        fps=8, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
        json.dump({"recon_psnr": rp, "src_idx": ii, "h_hat": h_hat, "v_hat": v_hat},
                  open(f"{od}/meta.json", "w"))
        recon[oid] = rp
        print(f"done {oid} (carla {ii:06d}) recon={rp:.2f}", flush=True)
    m = float(np.mean(list(recon.values())))
    json.dump({"mean_recon_psnr": m, "per_object": recon, "note": "input-view recon only; "
               "CARLA has no per-image poses so no novel-view GT / no NVS metric exists"},
              open(f"{HF}/pigan_real_summary.json", "w"), indent=1)
    print(f"[pigan_real] mean input-view recon PSNR = {m:.2f} over {len(recon)}")


if __name__ == "__main__":
    main()
