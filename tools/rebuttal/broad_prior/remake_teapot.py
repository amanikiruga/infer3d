"""Rebuild teapot turntable with 3 columns: EqM input image | masked-centered input | recovered 3D."""
import os, sys
import numpy as np, torch, torch.nn.functional as F, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
D = f"{HERE}/gallery_eqm/tt"
raw = np.asarray(Image.open(f"{D}/849_teapot_eqm.png").convert("RGB").resize((128, 128)))   # EqM sample
inp = np.asarray(Image.open(f"{D}/849_teapot_input.png").convert("RGB"))                     # masked 128

cfg = P.objaverse_cfg()
gp = P.load_lifter(cfg)
cams = P.build_turntable(cfg, num=60)
u = torch.from_numpy(inp).float().permute(2, 0, 1).to(P.DEV) / 255.
with torch.no_grad():
    splats = P.lift(gp, u)
tt = P.turntable_frames(cfg, splats, cams)
frames = [np.concatenate([raw, inp, f], axis=1) for f in tt]
imageio.mimsave(f"{D}/849_teapot_3col.mp4", frames, fps=20, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
Image.fromarray(np.concatenate([frames[i] for i in [0, 15, 30, 45]], axis=0)).save(f"{HERE}/SHOW_teapot_3col.png")
print("wrote 849_teapot_3col.mp4 (EqM input | masked input | recovered 3D)")
