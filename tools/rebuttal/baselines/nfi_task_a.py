"""nerf-from-image (Pavllo et al., CVPR'23) on Task A (NMR cars, SO(3) inputs).

Faithful programmatic port of the repo's `--run_inversion` shapenet path:
encoder bootstrap -> PnP pose/focal -> 30-step W+ optimization with the
LPIPS-VGG + 15-affine-augmentation loss (all numerics copied from run.py).
NFI reconstructs in the CANONICAL object frame with an ESTIMATED input camera
(no GT pose used) — exactly our protocol's information budget.

For each bundle we then render the exported relative target cameras composed
with the estimated input camera:  C_t^GL = C_hat^GL @ F @ C_rel^CV @ F, with
F = diag(1,-1,-1,1) (SRN/NMR CV convention <-> NFI OpenGL convention).

Outputs per object: renders_native30/ renders_ext300/ (24 PNGs each),
diag grid PNG, orbit.mp4, meta.json.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse
import json
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

NFI = f"{_EXT}/rebuttal_infer3d_baselines/nerf-from-image"
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, NFI)

from models import generator as nfi_generator
from models import encoder as nfi_encoder
from lib import pose_estimation, pose_utils, nerf_utils, metrics

device = "cuda"
SCENE_RANGE = 0.55
WHITE_BG = True
RES = 128
DEPTH_SAMPLES = 64
FLIP = np.diag([1., -1., -1., 1.]).astype(np.float32)


# ---------------- faithful render() (run.py:176, constants inlined) ----------
def render(model, height, width, tform_cam2world, focal_length, model_input,
           randomize=True, compute_normals=False):
    ray_origins, ray_directions = nerf_utils.get_ray_bundle(
        height, width, focal_length, tform_cam2world, None, None)
    ray_directions = F.normalize(ray_directions, dim=-1)
    with torch.no_grad():
        near_thresh, far_thresh = nerf_utils.compute_near_far_planes(
            ray_origins.detach(), ray_directions.detach(), SCENE_RANGE)
    query_points, depth_values = nerf_utils.compute_query_points_from_rays(
        ray_origins, ray_directions, near_thresh, far_thresh, DEPTH_SAMPLES,
        randomize=randomize)
    viewdirs = None  # use_viewdir=False for shapenet
    model_outputs = model(viewdirs, model_input, ['sampler'])
    sampler = model_outputs['sampler']
    req = ['sigma', 'rgb'] + (['normals'] if compute_normals else [])
    out_c = sampler(query_points, req)
    sigma = out_c['sigma'].view(*query_points.shape[:-1], -1)
    rgb = out_c['rgb'].view(*query_points.shape[:-1], -1)
    normals = out_c['normals'].view(*query_points.shape[:-1], -1) if compute_normals else None

    # fine sampling (run.py:261-335)
    z_vals = depth_values
    with torch.no_grad():
        weights = nerf_utils.render_volume_density_weights_only(
            sigma.squeeze(-1), ray_origins, ray_directions,
            depth_values).flatten(0, 2)
        weights = F.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
        weights = F.avg_pool1d(weights, 2, 1).squeeze()
        weights = weights + 0.01
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = nerf_utils.sample_pdf(z_vals_mid.flatten(0, 2),
                                          weights[..., 1:-1], DEPTH_SAMPLES,
                                          deterministic=not randomize)
        z_samples = z_samples.view(*z_vals.shape[:3], z_samples.shape[-1])
    z_values_sorted, z_indices_sorted = torch.sort(
        torch.cat((z_vals, z_samples), dim=-1), dim=-1)
    query_points_fine = ray_origins[..., None, :] + \
        ray_directions[..., None, :] * z_samples[..., :, None]
    out_f = sampler(query_points_fine, req)
    sigma_f = out_f['sigma'].view(*query_points_fine.shape[:-1], -1)
    rgb_f = out_f['rgb'].view(*query_points_fine.shape[:-1], -1)
    sigma = torch.cat((sigma, sigma_f), dim=-2).gather(
        -2, z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, sigma.shape[-1]))
    rgb = torch.cat((rgb, rgb_f), dim=-2).gather(
        -2, z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, rgb.shape[-1]))
    if compute_normals:
        normals_f = out_f['normals'].view(*query_points_fine.shape[:-1], -1)
        normals = torch.cat((normals, normals_f), dim=-2).gather(
            -2, z_indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, normals.shape[-1]))
    depth_values = z_values_sorted

    rgb_p, depth_p, mask_p, normals_p, sem_p = nerf_utils.render_volume_density(
        sigma.squeeze(-1), rgb, ray_origins, ray_directions, depth_values,
        normals, None, white_background=WHITE_BG)
    return rgb_p, depth_p, mask_p, normals_p


# ---------------- faithful augment (run.py:720-796, white_bg inlined) --------
def augment_impl(img, p):
    bs = img.shape[0]
    dev = img.device
    rot = (torch.rand((bs,), device=dev) - 0.5) * 2 * np.pi
    rot = rot * (torch.rand((bs,), device=dev) < p).float()
    scale = torch.exp2(torch.randn((bs,), device=dev) * 0.2)
    scale = torch.lerp(torch.ones_like(scale), scale,
                       (torch.rand((bs,), device=dev) < p).float())
    translation = torch.randn((bs, 2), device=dev) * 0.1
    translation = torch.lerp(torch.zeros_like(translation), translation,
                             (torch.rand((bs, 1), device=dev) < p).float())
    mat = torch.zeros((bs, 2, 3), device=dev)
    mat[:, 0, 0] = torch.cos(rot); mat[:, 0, 1] = -torch.sin(rot)
    mat[:, 0, 2] = translation[:, 0]
    mat[:, 1, 0] = torch.sin(rot); mat[:, 1, 1] = torch.cos(rot)
    mat[:, 1, 2] = -translation[:, 1]
    mat_scaled = mat.clone()
    mat_scaled *= scale[:, None, None]
    mat_scaled[:, :, 2] = torch.sum(mat[:, :2, :2] *
                                    mat_scaled[:, :, 2].unsqueeze(-2), dim=-1)
    grid = F.affine_grid(mat_scaled, img.shape, align_corners=False)
    img = img - 1  # white background adjustment
    img_t = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros',
                          align_corners=False)
    return img_t + 1


# ---------------- model loading ----------------------------------------------
def load_models():
    ck = torch.load(f"{NFI}/gan_checkpoints/g_shapenet_cars_pretrained/"
                    "checkpoint_latest.pth", map_location='cpu', weights_only=False)
    G = nfi_generator.Generator(
        latent_dim=512, scene_range=SCENE_RANGE, attention_values=10,
        use_viewdir=False, use_encoder=False, disable_stylegan_noise=True,
        use_sdf=True, num_classes=None).to(device)
    G.load_state_dict(ck['model_ema'])
    G.eval().requires_grad_(False)
    enc = nfi_encoder.BootstrapEncoder(
        512, pose_regressor=True, latent_regressor=True,
        separate_backbones=False, pretrained=False).to(device)
    sd = torch.load(f"{NFI}/coords_checkpoints/g_shapenet_cars_pretrained/"
                    "c_it300000_latest.pth", map_location='cpu', weights_only=False)['model_coord']
    enc.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    enc.eval().requires_grad_(False)
    lp = metrics.LPIPSLoss().to(device) if hasattr(metrics, 'LPIPSLoss') else None
    if lp is None:
        import lpips
        lp = lpips.LPIPS(net='vgg').to(device)
    return G, enc, lp


def load_input(png_path):
    from PIL import Image
    a = np.asarray(Image.open(png_path).convert('RGB')).astype(np.float32)
    img = torch.from_numpy(a / 255. * 2 - 1)[None].to(device)  # (1,H,W,3)
    return img


def estimate_poses_batch(target_coords, target_mask, focal_guesses):
    """Verbatim port of run.py:1709 (non-ortho branch, focal_guesses given)."""
    target_mask = target_mask > 0.9
    world2cam_mat, estimated_focal, errors = pose_estimation.compute_pose_pnp(
        target_coords.cpu().numpy(), target_mask.cpu().numpy(), focal_guesses)
    c2w = pose_utils.invert_space(
        torch.from_numpy(world2cam_mat).float()).to(target_coords.device)
    focal = torch.from_numpy(estimated_focal).float().to(target_coords.device)
    return c2w, focal, errors


def invert(G, enc, lp, target_img, steps, focal_guess):
    with torch.no_grad():
        coords, mask, w = enc(target_img.permute(0, 3, 1, 2))
        c2w, focal, err = estimate_poses_batch(
            coords, mask, np.array([focal_guess]))
    z_avg = G.mapping_network.get_average_w()
    z = (w.expand(-1, z_avg.shape[1], -1).contiguous() / 5).requires_grad_()
    opt = torch.optim.Adam([z], lr=2e-3, betas=(0.9, 0.95))
    cam = c2w.to(device).float()            # already (bs,4,4) torch tensor
    foc = focal.to(device).float()          # already (bs,) torch tensor
    for it in range(steps):
        rgb, _, _, _ = render(G, RES, RES, cam, foc, z * 5)
        pred = rgb.permute(0, 3, 1, 2)
        targ = target_img[..., :3].permute(0, 3, 1, 2)
        cat = torch.cat((pred, targ), dim=1)
        cat = cat.unsqueeze(1).expand(-1, 15, -1, -1, -1).contiguous().flatten(0, 1)
        cat = augment_impl(cat, 1.0)
        pa = torch.cat((pred, cat[:, :3]), dim=0)
        ta = torch.cat((targ, cat[:, 3:]), dim=0)
        loss = lp(pa, ta).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (z * 5).detach(), cam, foc, float(np.asarray(err)[0])


def compose_target_cam(cam_est_gl, c_in_cv, c_tgt_cv):
    """Target cam2world in NFI/GL frame, anchored at the estimated input pose.
    G maps NMR-world -> NFI-world: C_t^GL = C_hat^GL @ F @ inv(C_in^CV) @ C_tgt^CV @ F,
    where C^CV = view_to_world_transforms_absolute.T (splatter->standard cam2world)
    and F = diag(1,-1,-1,1) converts the CV camera frame to NFI's OpenGL frame."""
    return cam_est_gl @ FLIP @ np.linalg.inv(c_in_cv) @ c_tgt_cv @ FLIP


def save_rgb(rgb, path):  # rgb (H,W,3) in [-1,1]
    from PIL import Image
    a = np.clip((rgb.cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def run_object(G, enc, lp, bdir, odir, steps_list=(30, 300)):
    cams = np.load(f"{bdir}/cams_relative.npz")
    fov = float(cams["fov_deg"])
    focal_guess = (RES / 2) / np.tan(np.deg2rad(fov) / 2) / RES
    target_img = load_input(f"{bdir}/input.png")
    meta = {"fov_deg": fov, "focal_guess": focal_guess}
    cabs = np.load(f"{bdir}/cams_absolute.npz")
    c_in_cv = cabs["input_v2w"].T.astype(np.float32)        # standard cam2world (CV)
    targets_v2w = cabs["targets_v2w"]                        # (N,4,4) splatter conv

    for steps in steps_list:
        torch.manual_seed(0); np.random.seed(0)
        ws, cam, foc, pnp_err = invert(G, enc, lp, target_img, steps, focal_guess)
        cam_np = cam[0].cpu().numpy()
        rdir = f"{odir}/renders_{'native30' if steps == 30 else f'ext{steps}'}"
        os.makedirs(rdir, exist_ok=True)
        for i in range(targets_v2w.shape[0]):
            c_tgt_cv = targets_v2w[i].T.astype(np.float32)
            ct = compose_target_cam(cam_np, c_in_cv, c_tgt_cv)
            ct_t = torch.from_numpy(ct)[None].to(device)
            with torch.no_grad():
                rgb, _, _, _ = render(G, RES, RES, ct_t, foc, ws, randomize=False)
            save_rgb(torch.clamp(rgb[0], -1, 1), f"{rdir}/{i:02d}.png")
        # reprojection of the input view (diagnostic)
        with torch.no_grad():
            rgb_in, _, _, _ = render(G, RES, RES, cam, foc, ws, randomize=False)
        save_rgb(torch.clamp(rgb_in[0], -1, 1), f"{odir}/reproj_{steps}.png")
        meta[f"pnp_err_{steps}"] = pnp_err
        meta[f"cam_est_{steps}"] = cam_np.tolist()
    json.dump(meta, open(f"{odir}/meta.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_runs/nfi")
    ap.add_argument("--only", default=None, help="single object id")
    ap.add_argument("--steps", default="30,300")
    args = ap.parse_args()
    steps_list = tuple(int(s) for s in args.steps.split(","))
    G, enc, lp = load_models()
    ids = sorted(os.listdir(args.bundles))
    if args.only:
        ids = [args.only]
    for oid in ids:
        odir = f"{args.out}/{oid}"
        os.makedirs(odir, exist_ok=True)
        run_object(G, enc, lp, f"{args.bundles}/{oid}", odir, steps_list)
        print(f"done {oid}")
    print("ALL DONE")


if __name__ == "__main__":
    main()
