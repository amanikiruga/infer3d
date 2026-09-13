"""pi-GAN (Chan et al. CVPR'21) geometry for Task A/B — gauge-free ICP-Chamfer.

CARLA-pretrained SIREN 3D-GAN. Invert a single 128px car image by optimizing the
SIREN frequency/phase offsets AND camera (h,v) [pose optimization ON], then extract
a density grid -> marching cubes -> surface point cloud. ICP-Chamfer scored separately.
CARLA is the only released car model => synthetic driving-sim prior (inherent limit).
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, math, os, sys
import numpy as np, torch, trimesh
from skimage import measure
from PIL import Image

PIGAN = f"{_EXT}/rebuttal_infer3d_baselines/pi-GAN"
CKDIR = f"{_EXT}/rebuttal_infer3d_baselines/checkpoints/pigan_carla/CARLA"
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, PIGAN)
dev = "cuda"

# CARLA render options (curriculums.py) for inversion + extraction
OPT = dict(img_size=128, fov=30, ray_start=0.7, ray_end=1.3, num_steps=24,
           hierarchical_sample=False, h_stddev=0, v_stddev=0, sample_dist=None,
           nerf_noise=0, white_back=True, clamp_mode='relu', last_back=False,
           lock_view_dependence=False)


def _stub_torch_ema():
    """Minimal torch_ema stub so ema.pth unpickles without installing the package."""
    import types
    if "torch_ema" in sys.modules:
        return
    m = types.ModuleType("torch_ema"); mm = types.ModuleType("torch_ema.ema")
    class ExponentialMovingAverage:
        def __init__(self, *a, **k):
            self.shadow_params = []
        def copy_to(self, params):
            for s, p in zip(self.shadow_params, params):
                p.data.copy_(s.data)
    m.ExponentialMovingAverage = ExponentialMovingAverage
    mm.ExponentialMovingAverage = ExponentialMovingAverage
    m.ema = mm
    sys.modules["torch_ema"] = m; sys.modules["torch_ema.ema"] = mm


def load_gen():
    _stub_torch_ema()
    G = torch.load(f"{CKDIR}/generator.pth", map_location=dev, weights_only=False)
    try:
        ema = torch.load(f"{CKDIR}/ema.pth", map_location=dev, weights_only=False)
        for s, p in zip(ema.shadow_params, G.parameters()):
            p.data.copy_(s.data.to(dev))
        print("applied EMA weights")
    except Exception as e:
        print(f"EMA skip ({e}); using raw generator.pth")
    G.set_device(dev); G.eval()
    return G


def load_img(p):
    a = np.asarray(Image.open(p).convert("RGB").resize((128, 128))).astype(np.float32) / 255.
    return (torch.from_numpy(a).permute(2, 0, 1)[None].to(dev) * 2 - 1)  # [-1,1]


def invert(G, target, iters=700):
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
        o = dict(OPT); o["h_mean"] = h; o["v_mean"] = v
        frame, _ = G.forward_with_frequencies(wf + n + off_f, wp + n + off_p, **o)
        loss = torch.mean((frame - target) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    return (wf + off_f).detach(), (wp + off_p).detach()


def extract_pc(G, wf, wp, N=96, cube=0.6, n_surface=50000):
    g = np.linspace(-cube / 2, cube / 2, N)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts_t = torch.from_numpy(pts).float().to(dev).unsqueeze(0)
    dirs = torch.zeros_like(pts_t); dirs[..., 2] = -1
    sig = []
    with torch.no_grad():
        for i in range(0, pts_t.shape[1], 200000):
            out = G.siren.forward_with_frequencies_phase_shifts(
                pts_t[:, i:i+200000], wf, wp, ray_directions=dirs[:, i:i+200000])
            sig.append(out[..., -1].reshape(-1))
    sig = torch.cat(sig).reshape(N, N, N).cpu().numpy()
    sig = np.maximum(sig, 0)
    lvl = float(np.percentile(sig[sig > 0], 50)) if (sig > 0).any() else 1.0
    if not (sig.min() < lvl < sig.max()):
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(sig, level=lvl)
    except Exception:
        return None
    verts = verts / (N - 1) * cube - cube / 2
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if m.area == 0:
        return None
    p, _ = trimesh.sample.sample_surface(m, n_surface)
    return np.asarray(p, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_geom/pigan")
    ap.add_argument("--realcars", action="store_true")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    G = load_gen()
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    ids = [o for o in ids if not os.path.exists(f"{args.out}/{o}.npy")]
    inp = "rgb_white_128.png" if args.realcars else "input.png"
    for oid in ids:
        target = load_img(f"{args.bundles}/{oid}/{inp}")
        wf, wp = invert(G, target, args.iters)
        pc = extract_pc(G, wf, wp)
        if pc is None:
            print(f"FAIL {oid}"); continue
        np.save(f"{args.out}/{oid}.npy", pc)
        print(f"done {oid} ({pc.shape[0]} pts)")
    print("ALL DONE")


if __name__ == "__main__":
    main()
