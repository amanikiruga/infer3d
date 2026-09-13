"""Sanity: fuse a CLEAN feed-forward lift of an Objaverse view-0 and CD it to the true GLB mesh.
If the fusion/frame/ICP are correct, a clean full-view lift reconstructs the object -> LOW CD,
and the fused cloud visually overlaps GT. Saves a top-view scatter of pred(red) vs GT(gray)."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P, fuse_cd as FC
from PIL import Image

uid = sys.argv[1] if len(sys.argv) > 1 else "45db143ee7cf48f4803a5d7be9d9b865"
cfg = P.objaverse_cfg()
gp = P.load_lifter(cfg)
it = ig.objaverse_item_uid(cfg, uid, src_idx=0)               # clean view-0
I = it["gt_images"][0].unsqueeze(0).to(FC.DEV)
with torch.no_grad():
    ff = P.lift(gp, I)
pred = FC.fuse(cfg, ff, n_views=120)
gt = FC.gt_cloud(uid, 50000)
cd, Xt, Y, _ = FC.align_cd(pred, gt)
print(f"[{uid[:8]}] clean feed-forward fused CD = {cd:.4f}  (pred pts {pred.shape[0]}, gt pts {gt.shape[0]})")

# visual: 3 orthographic views; GT(gray) then pred(red) with 2x2 point blocks
def scat(A, B, ax0, ax1, res=256):
    img = np.full((res, res, 3), 255, np.uint8)
    allp = torch.cat([A, B], 0); c = allp.mean(0); sc = (allp - c).abs().max() * 1.1 + 1e-9
    def draw(pts, col):
        p = ((pts - c) / sc * 0.5 + 0.5)[:, [ax0, ax1]].clamp(0, 0.995)
        xy = (p * res).long().cpu().numpy()
        for dx in (0, 1):
            for dy in (0, 1):
                img[np.clip(res - 1 - xy[:, 1] - dy, 0, res - 1), np.clip(xy[:, 0] + dx, 0, res - 1)] = col
    draw(B, np.array([150, 150, 150], np.uint8))
    draw(A, np.array([220, 30, 30], np.uint8))
    return img
views = np.concatenate([scat(Xt, Y, 0, 1), scat(Xt, Y, 2, 1), scat(Xt, Y, 0, 2)], 1)
Image.fromarray(views).save(f"{os.path.dirname(os.path.abspath(__file__))}/SANITY_cd_{uid[:8]}.png")
print("saved SANITY (red=fused pred aligned to GT, gray=GT mesh); views: front | side | top")
