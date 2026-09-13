"""pi-GAN homefield NVS — same protocol as homefield_gan.py but in pi-GAN's own
CARLA domain. Sample K latents from the released CARLA generator, render input +
24 ring targets at known (h, v), invert the input with pi-GAN's own pipeline
(frequency/phase offsets + pose), render targets anchored at the estimated pose,
PSNR vs the generator's own target renders. Input pose is EXACTLY the inverter's
init (h=v=pi/2) so pi-GAN's assumed-pose convention is exactly satisfied — its best
case. (First run used inputs offset up to ±20°: pose opt barely moved from init and
PSNR degraded linearly with the offset — kept as homefield/pigan_offpose ablation.)
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, math, os, sys
import numpy as np, torch
from PIL import Image
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import pigan_geom as P
dev = "cuda"
HF = f"{WT}/rebuttal/results/homefield"
K = 15
N_TGT = 24
V = math.pi / 2  # input AND target ring at the inverter's own init pose convention


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def render(G, wf, wp, h, v):
    o = dict(P.OPT); o["h_mean"] = h; o["v_mean"] = v
    o["num_steps"] = 36; o["lock_view_dependence"] = True
    with torch.no_grad():
        try:
            px, _ = G.staged_forward_with_frequencies(wf, wp, max_batch_size=400000, **o)
        except Exception:
            px, _ = G.forward_with_frequencies(wf, wp, **o)
    return np.clip((px[0].permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)


def invert_hv(G, target, iters=700):
    """pigan_geom.invert but also returns the estimated (h, v)."""
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
    ap.add_argument("--iters", type=int, default=700)
    a = ap.parse_args()
    G = P.load_gen()
    all_psnr = {}
    for k in range(K):
        oid = f"s{k:02d}"
        idir = f"{HF}/inputs_pigan/{oid}"; od = f"{HF}/pigan/{oid}"
        os.makedirs(f"{idir}/targets", exist_ok=True); os.makedirs(f"{od}/renders", exist_ok=True)
        if os.path.exists(f"{od}/meta.json"):
            all_psnr[oid] = json.load(open(f"{od}/meta.json"))["psnr"]
            print(f"skip {oid} (done, {all_psnr[oid]:.2f})"); continue
        torch.manual_seed(2000 + k)
        z = torch.randn(1, 256, device=dev)
        with torch.no_grad():
            wf_t, wp_t = G.siren.mapping_network(z)
        h_in = math.pi / 2  # exactly the inverter's init: assumption exactly satisfied
        az_tgt = [2 * math.pi * i / N_TGT for i in range(N_TGT)]
        Image.fromarray(render(G, wf_t, wp_t, h_in, V)).save(f"{idir}/input.png")
        gts = []
        for i, hh in enumerate(az_tgt):
            g = render(G, wf_t, wp_t, hh, V)
            Image.fromarray(g).save(f"{idir}/targets/{i:02d}.png")
            gts.append(g)
        json.dump({"h_in": h_in, "v_in": V, "h_tgt": az_tgt, "v_tgt": V},
                  open(f"{idir}/cams.json", "w"))
        target = P.load_img(f"{idir}/input.png")
        wf, wp, h_hat, v_hat = invert_hv(G, target, a.iters)
        ps = []
        for i, hh in enumerate(az_tgt):
            r = render(G, wf, wp, h_hat + (hh - h_in), v_hat)  # targets share the input's v
            Image.fromarray(r).save(f"{od}/renders/{i:02d}.png")
            ps.append(psnr(r, gts[i]))
        m = float(np.mean(ps))
        json.dump({"psnr": m, "per_view": [round(p, 2) for p in ps],
                   "h_hat": h_hat, "v_hat": v_hat, "h_err": h_hat - h_in,
                   "v_err": v_hat - V}, open(f"{od}/meta.json", "w"))
        all_psnr[oid] = m
        print(f"done {oid} psnr={m:.2f} h_err={h_hat - h_in:+.3f}")
    json.dump({"mean": float(np.mean(list(all_psnr.values()))), "per_object": all_psnr},
              open(f"{HF}/pigan_summary.json", "w"), indent=1)
    print(f"[pigan] mean PSNR = {np.mean(list(all_psnr.values())):.2f} over {len(all_psnr)}")


if __name__ == "__main__":
    main()
