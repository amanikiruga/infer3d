"""Minimal EG3D shapenetcars load + render test (torch-2.9 custom-op risk check)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import sys, numpy as np, torch, pickle
REPO = f"{_EXT}/rebuttal_infer3d_baselines/3D-GAN-Inversion"
PKL = f"{_EXT}/rebuttal_infer3d_baselines/checkpoints/shapenetcars128-64.pkl"
sys.path.insert(0, REPO)

# Force pure-pytorch reference paths for custom ops (no compiler on torch-2.9 env):
# _init()->False makes bias_act/upfirdn2d take their 'ref' branch.
from torch_utils.ops import bias_act, upfirdn2d
bias_act._init = lambda: False
upfirdn2d._init = lambda: False

device = "cuda"
with open(PKL, "rb") as f:
    G = pickle.load(f)["G_ema"].to(device).eval().float()
print("loaded G:", type(G).__name__)
print("rendering_kwargs:", {k: G.rendering_kwargs.get(k) for k in
      ["box_warp", "ray_start", "ray_end", "avg_camera_radius", "white_back",
       "c_gen_conditioning_zero", "image_resolution"]})
print("neural_rendering_resolution:", getattr(G, "neural_rendering_resolution", None))
print("z_dim,c_dim,w_dim:", G.z_dim, G.c_dim, G.w_dim,
      "backbone.num_ws:", G.backbone.num_ws)

# build a frontal camera c (cam2world at radius 1.7 + car intrinsics)
r = G.rendering_kwargs.get("avg_camera_radius", 1.7)
try:
    from utils.camera_utils import LookAtPoseSampler
    piv = torch.tensor(G.rendering_kwargs.get("avg_camera_pivot", [0, 0, 0]), device=device)
    c2w = LookAtPoseSampler.sample(np.pi/2, np.pi/2, piv, radius=r, device=device)  # (1,4,4)
    print("LookAtPoseSampler ok, cam center:", c2w[0, :3, 3].cpu().numpy().round(3))
except Exception as e:
    print("LookAtPoseSampler failed:", e); c2w = torch.eye(4, device=device)[None]; c2w[0,2,3]=r
intr = torch.tensor([[1.0254, 0, 0.5], [0, 1.0254, 0.5], [0, 0, 1.]], device=device)
c = torch.cat([c2w.reshape(-1, 16), intr.reshape(-1, 9)], 1)

z = torch.randn(1, G.z_dim, device=device)
with torch.no_grad():
    ws = G.mapping(z, c, truncation_psi=0.7)
    out = G.synthesis(ws, c)
print("synthesis keys:", list(out.keys()))
print("image:", tuple(out["image"].shape), "range", float(out["image"].min()), float(out["image"].max()))
# geometry sanity
with torch.no_grad():
    pts = (torch.rand(1, 1000, 3, device=device) - 0.5) * G.rendering_kwargs["box_warp"]
    sig = G.sample_mixed(pts, torch.zeros_like(pts), ws, noise_mode="const")["sigma"]
print("sample_mixed sigma:", tuple(sig.shape), "mean", float(sig.mean()))
print("EG3D LOADTEST OK")
