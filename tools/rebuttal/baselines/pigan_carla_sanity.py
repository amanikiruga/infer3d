"""pi-GAN CARLA-domain sanity: sample random latents from the released CARLA generator and
render them at a few azimuths. Shows the prior produces clean cars IN ITS OWN DOMAIN — so the
blobs on ShapeNet/RealCars are a genuine domain gap, not a broken model. Saves a montage +
a short orbit video for one sample."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import math, os, sys
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import pigan_geom as P
dev = "cuda"
OUT = f"{WT}/rebuttal/report/videos/pigan_carla"; os.makedirs(OUT, exist_ok=True)


def render(G, wf, wp, h):
    o = dict(P.OPT); o["h_mean"] = h; o["v_mean"] = math.pi / 2 - 0.2
    o["num_steps"] = 36; o["lock_view_dependence"] = True
    with torch.no_grad():
        try:
            px, _ = G.staged_forward_with_frequencies(wf, wp, max_batch_size=400000, **o)
        except Exception:
            px, _ = G.forward_with_frequencies(wf, wp, **o)
    return np.clip((px[0].permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)


def main():
    G = P.load_gen()
    torch.manual_seed(0)
    zs = torch.randn(6, 256, device=dev)
    azdeg = [math.pi / 2 + a for a in np.deg2rad([-45, -15, 15, 45])]
    rows = []
    for k in range(6):
        with torch.no_grad():
            wf, wp = G.siren.mapping_network(zs[k:k + 1])
        cells = [render(G, wf, wp, h) for h in azdeg]
        rows.append(np.concatenate([np.asarray(Image.fromarray(c).resize((128, 128))) for c in cells], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(f"{OUT}/carla_samples.png")
    # short orbit video of sample 0
    with torch.no_grad():
        wf, wp = G.siren.mapping_network(zs[0:1])
    frames = [render(G, wf, wp, float(h)) for h in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    imageio.mimsave(f"{OUT}/carla_orbit.mp4", [np.asarray(Image.fromarray(f).resize((128, 128))) for f in frames],
                    fps=8, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
    print("CARLA sanity saved:", OUT)


if __name__ == "__main__":
    main()
