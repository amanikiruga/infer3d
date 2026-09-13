"""Fused Chamfer Distance for Objaverse, adapting eval/fused_cd_and_icp_nvs.md to Objaverse:
 - GT = the actual GLB mesh (objaverse.load_objects), sampled to a surface cloud (true GT).
 - PRED = fuse the predicted gaussians to a SURFACE cloud (render a sphere of views with depth,
   back-project foreground pixels) -- NOT bare gaussian centers (the doc's key point).
 - ALIGN pred->GT with centroid-init + scale-guarded Sim(3) ICP (pytorch3d); symmetric CD after.
Also returns the ICP transform so the gaussians can be re-rendered ICP-corrected (Part B).
"""
import os, sys, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
from gaussian_renderer import render_predicted
from utils.graphics_utils import fov2focal, getProjectionMatrix
from pytorch3d.ops import iterative_closest_point, sample_farthest_points
from pytorch3d.loss import chamfer_distance
import trimesh

DEV = "cuda"


def gt_cloud(uid, n=50000):
    import objaverse
    p = objaverse.load_objects([uid])[uid]
    m = trimesh.load(p, force="mesh")
    pts, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(np.asarray(pts), dtype=torch.float32, device=DEV)


def sphere_cams(n=120, radius=2.0):
    """Fibonacci-sphere camera centers looking at origin; return list of 4x4 c2w (OpenCV:
    x right, y down, z from camera toward the scene)."""
    cams = []
    ga = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0, 1 - y * y))
        th = ga * i
        pos = np.array([math.cos(th) * r, y, math.sin(th) * r], np.float32) * radius
        z = -pos / (np.linalg.norm(pos) + 1e-9)          # OpenCV: +z points from camera to origin
        up = np.array([0, 1, 0], np.float32)
        if abs(np.dot(z, up)) > 0.99:
            up = np.array([1, 0, 0], np.float32)
        x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-9
        yv = np.cross(z, x)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = x; c2w[:3, 1] = yv; c2w[:3, 2] = z; c2w[:3, 3] = pos
        cams.append(c2w)
    return cams


def fuse(cfg, splats, n_views=120, res=128):
    """TSDF-fuse the gaussians (render a sphere of views -> depth -> Open3D ScalableTSDFVolume ->
    mesh -> sample), the eval-scripts way (robust to noisy single-view depth). Returns [M,3]
    in the lifter world frame. Falls back to [] on empty mesh."""
    import open3d as o3d
    fov = cfg.data.fov
    focal = float(fov2focal(fov * np.pi / 180.0, res))
    cx = cy = res / 2.0
    proj = getProjectionMatrix(znear=cfg.data.znear, zfar=cfg.data.zfar,
                               fovX=fov * 2 * np.pi / 360, fovY=fov * 2 * np.pi / 360).transpose(0, 1).to(DEV)
    bg = torch.tensor([1., 1, 1], device=DEV)
    intr = o3d.camera.PinholeCameraIntrinsic(res, res, focal, focal, cx, cy)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025, sdf_trunc=0.10,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    n_int = 0
    for c2w in sphere_cams(n_views):
        c2w_t = torch.from_numpy(c2w).to(DEV)
        w2c = torch.inverse(c2w_t)
        wv = w2c.transpose(0, 1); full = wv @ proj; cc = c2w_t[:3, 3]
        with torch.no_grad():
            out = render_predicted(splats, wv, full, cc, bg, cfg, focals_pixels=None)
        img = out["render"].clamp(0, 1); invd = out["invdepths"][0]
        fg = (img < 0.9).any(0) & (invd > 1e-3) & (invd < 1e3)
        fg = (-torch.nn.functional.max_pool2d(-fg.float()[None, None], 3, 1, 1)[0, 0]) > 0.5
        if fg.sum() < 30:
            continue
        depth = torch.where(fg, 1.0 / invd.clamp(min=1e-4), torch.zeros_like(invd))
        col = (img.permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        dep = depth.cpu().numpy().astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(col)), o3d.geometry.Image(dep),
            depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, w2c.cpu().numpy().astype(np.float64))  # extrinsic = w2c (col-vec)
        n_int += 1
    mesh = vol.extract_triangle_mesh()
    if len(mesh.vertices) < 50:
        return torch.zeros((0, 3), device=DEV)
    pcd = mesh.sample_points_uniformly(number_of_points=40000)
    return torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=DEV)


def _rms_radius(x):
    return (x - x.mean(0)).pow(2).sum(1).mean().sqrt() + 1e-9


def align_cd(pred, gt):
    """Doc-faithful: pre-scale pred to GT's size (RMS radius), centroid init, then guarded ICP
    (identity / rigid / Sim3 with 0.6<s<1.7); pick min symmetric CD. CD is in GT (mesh) units.
    Returns (cd, aligned_pred[N,3], gt_centered[M,3])."""
    if pred.shape[0] < 50:
        return 999.0, pred, gt - gt.mean(0), None
    gc = gt.mean(0); Y = (gt - gc)
    gtr = _rms_radius(Y)                                  # GT scale, for scale-free CD
    pm = pred.mean(0); s0 = gtr / _rms_radius(pred)
    Pp = (pred - pm) * s0                                 # pre-scaled to GT size, centered
    Xb = (Pp + (Y.mean(0) - Pp.mean(0)))                  # centroid init
    # transform of a lifter-frame point x into the GT-centered frame Y:
    #   identity: s0*(x-pm);  rigid/sim3: RTs.s*((s0*(x-pm))@RTs.R)+RTs.T
    cands = [(Xb, "identity", {"pm": pm, "s0": s0, "R": torch.eye(3, device=DEV), "s": 1.0, "T": torch.zeros(3, device=DEV)})]
    for scale in (False, True):
        try:
            sol = iterative_closest_point(Xb.unsqueeze(0), Y.unsqueeze(0), estimate_scale=scale,
                                          max_iterations=80, verbose=False)
            si = float(sol.RTs.s[0])
            if scale and not (0.6 < si < 1.7):
                continue
            cands.append((sol.Xt[0], "sim3" if scale else "rigid",
                          {"pm": pm, "s0": s0, "R": sol.RTs.R[0], "s": si if scale else 1.0, "T": sol.RTs.T[0]}))
        except Exception:
            pass
    best = None
    for Xt, name, T in cands:
        cd = float(chamfer_distance(Xt.unsqueeze(0), Y.unsqueeze(0))[0])
        if best is None or cd < best[0]:
            best = (cd, Xt, name, T)
    return best[0] / float(gtr) ** 2, best[1], Y, best[3]   # scale-free CD, aligned pred, GT, transform


def apply_sim3_to_gauss(g, T):
    """Apply the align_cd transform (lifter frame -> GT-centered frame) to gaussians:
    xyz -> s*((s0*(xyz-pm))@R)+T ; scaling -> s*s0*scaling ; rotation -> q(R^T) o q."""
    from utils.general_utils import matrix_to_quaternion, quaternion_raw_multiply
    out = dict(g)
    xyz = (g["xyz"] - T["pm"]) * T["s0"]
    out["xyz"] = T["s"] * (xyz @ T["R"]) + T["T"]
    if "scaling" in g:
        out["scaling"] = g["scaling"] * (T["s"] * T["s0"])
    q = matrix_to_quaternion(T["R"].transpose(0, 1)).unsqueeze(0).expand(g["rotation"].shape[0], -1)
    out["rotation"] = quaternion_raw_multiply(q, g["rotation"])
    return out
