"""ICP-corrected NVS re-render (Part B of eval/fused_cd_and_icp_nvs.md): apply the CD-alignment
Sim(3) to the Infer3D gaussians and render a turntable in the GT-aligned frame, with the GT mesh
overlaid (gray) so the geometry match is visible from all angles. Also renders Splatter-direct
ICP-corrected for comparison. Per object: [GT+Splatter-aligned | GT+Infer3D-aligned] turntable."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, os, sys
import numpy as np, torch, imageio.v2 as imageio
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig, pipeline as P, fuse_cd as FC
from ig2 import make_visibility, invert_partial
from gaussian_renderer import render_predicted
from utils.camera_utils import get_loop_cameras
from utils.graphics_utils import fov2focal, getProjectionMatrix
import lpips as lpips_lib

OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/icp_render"
DEV = FC.DEV


def gt_frame_cams(gtr, n=60, res=128):
    fov = 49.0
    focal = float(fov2focal(fov * np.pi / 180, res)); cx = res / 2
    proj = getProjectionMatrix(znear=0.01 * gtr, zfar=20 * gtr, fovX=fov * np.pi / 180, fovY=fov * np.pi / 180).transpose(0, 1).to(DEV)
    out = []
    for c2w in get_loop_cameras(n, radius=2.6 * gtr, max_elevation=np.pi / 9):
        c2w = torch.from_numpy(c2w).float().to(DEV)
        w2c = torch.inverse(c2w)
        out.append((w2c.transpose(0, 1), (w2c.transpose(0, 1) @ proj), c2w[:3, 3], c2w, focal, cx))
    return out, focal, cx


def render_pair(cfg, g_icp, gt_pts, cams, focal, cx, res=128):
    bg = torch.tensor([1., 1, 1], device=DEV)
    frames = []
    for wv, full, cc, c2w, foc, cxx in cams:
        with torch.no_grad():
            im = render_predicted(g_icp, wv, full, cc, bg, cfg, focals_pixels=None)["render"].clamp(0, 1)
        img = (im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        # overlay GT points (gray) projected into this camera
        w2c = torch.inverse(c2w)
        pc = gt_pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2].clamp(min=1e-4)
        u = (foc * pc[:, 0] / z + cxx).long(); v = (foc * pc[:, 1] / z + cxx).long()
        m = (u >= 0) & (u < res) & (v >= 0) & (v < res) & (pc[:, 2] > 0)
        uu = u[m].cpu().numpy(); vv = v[m].cpu().numpy()
        gtimg = np.full((res, res, 3), 255, np.uint8); gtimg[vv, uu] = [130, 130, 130]
        frames.append(np.concatenate([gtimg, img], 1))     # [GT points | ICP-aligned gaussian render]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=999)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/cd_sel25.json"))
    S = {x["tag"]: x for x in json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/cd_out/summary_0.json"))
         + json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/cd_out/summary_30.json"))}
    pool = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/cd_pool.json")); cats = pool["cats"]
    items = [S[t] for t in sel][a.start:a.start + a.count]

    cfg = P.objaverse_cfg()
    gp = P.load_lifter(cfg); eqm = P.load_eqm(); vae = P.load_vae()
    cams8 = P.build_turntable(cfg, num=8)
    lp = lpips_lib.LPIPS(net="vgg").to(DEV).eval()

    for r in items:
        try:
            uid, cat = r["uid"], r["cat"]; cid = cats[cat]
            gt = FC.gt_cloud(uid, 50000)
            it = ig.objaverse_item_uid(cfg, uid, src_idx=0)
            I = it["gt_images"][0].unsqueeze(0).to(DEV); M = (I < 0.985).any(1, keepdim=True).float()
            V, O = make_visibility(M, "right", 0.45)
            spr, recon, R = invert_partial(I, M, V, gp, eqm, vae, cams8, lp, cfg, use_dino=True,
                                           pose=False, warm=True, n_prior=2, K=6, nrot=2, steps1=45, steps2=70, classid=cid)
            Pinv = FC.fuse(cfg, spr, n_views=120)
            cd, Xt, Y, T = FC.align_cd(Pinv, gt)
            if T is None:
                print("skip (fusion empty)", uid); continue
            g_icp = FC.apply_sim3_to_gauss(spr, T)
            gtr = float(FC._rms_radius(Y))
            cams, foc, cx = gt_frame_cams(gtr, n=60)
            frames = render_pair(cfg, g_icp, Y, cams, foc, cx)
            tag = r["tag"]
            imageio.mimsave(f"{OUT}/{tag}.mp4", frames, fps=18, codec="libx264", output_params=["-pix_fmt", "yuv420p"])
            Image.fromarray(np.concatenate([frames[i] for i in [0, len(frames)//4, len(frames)//2, 3*len(frames)//4]], 0)).save(f"{OUT}/{tag}_strip.png")
            print(f"[{tag}] ICP-corrected re-render done (CD {cd:.3f})")
        except Exception as e:
            import traceback; traceback.print_exc(); print("skip", r["tag"], e)
    print("done shard")


if __name__ == "__main__":
    main()
