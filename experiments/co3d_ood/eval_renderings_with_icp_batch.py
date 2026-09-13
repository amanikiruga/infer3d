#!/usr/bin/env python3
"""
Batch rendering evaluation script with ICP alignment.

Reads checkpoint paths and pointcloud PLY paths from text files, matches by object_id, then:
1. Loads pred and GT pointcloud PLY files from disk
2. Runs Sim(3) ICP alignment on the PLY pointclouds
3. Loads checkpoint and regenerates splats
4. Aligns pred pointcloud to GT pointcloud centroid
5. Applies ICP transformation to the rendered splats
6. Renders all three versions (baseline_indist, pred, pred+ICP) and computes PSNR/SSIM/LPIPS
7. Logs pointcloud visualizations + rendering videos to wandb
8. Saves metrics to CSV

Supports two pred_method options:
- "ours": Evaluates optimized method with learned rotation/translation (default)
- "baseline_ood": Evaluates baseline method with OOD rotation applied

All methods are compared against baseline_indist (pseudo-GT).
"""

import argparse
import csv
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import copy
import glob
from datetime import datetime

# PyTorch3D for ICP
from pytorch3d.ops import iterative_closest_point

# scikit-learn for Chamfer Distance
from sklearn.neighbors import NearestNeighbors

# Open3D for loading pointclouds
import open3d as o3d

# Plotly for interactive visualization
import plotly.graph_objects as go

# Image/video libraries
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Splatter Image imports
from infer3d import config as _cfg

from infer3d.model import GaussianSplatPredictor
from infer3d.utils.optim import render_with_custom_camera, render_with_custom_camera_align, foreground_mask
from infer3d.renderer import render_predicted
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
import hydra

# ICP and metrics
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from infer3d.utils.loss import ssim as ssim_fn
import lpips as lpips_lib
from infer3d.utils.general import matrix_to_quaternion, quaternion_raw_multiply
from infer3d.data.co3d import CO3DDataset

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


# ============================================================================
# POINTCLOUD UTILITIES
# ============================================================================

def load_pointcloud(ply_path, num_points=None):
    """Load pointcloud from .ply file and optionally subsample."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Pointcloud file not found: {ply_path}")
    
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        raise ValueError(f"Pointcloud is empty: {ply_path}")
    
    if num_points is not None and len(points) > num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    
    return points


def compute_chamfer_distance(X, Y):
    """
    Compute Chamfer Distance between two point clouds using sklearn.
    X, Y: [N, 3] and [M, 3] numpy arrays
    
    Returns:
        chamfer: Total Chamfer Distance (pred_to_gt + gt_to_pred)
        pred_to_gt: Mean distance from pred to gt
        gt_to_pred: Mean distance from gt to pred
    """
    # Pred -> GT: For each point in X, find nearest in Y
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Y)
    distances, _ = nbrs.kneighbors(X)
    pred_to_gt = np.mean(distances)
    
    # GT -> Pred: For each point in Y, find nearest in X
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    distances, _ = nbrs.kneighbors(Y)
    gt_to_pred = np.mean(distances)
    
    chamfer = pred_to_gt + gt_to_pred
    
    return chamfer, pred_to_gt, gt_to_pred


def visualize_pointclouds_plotly(pc_pred, pc_gt, title="3D Pointcloud Comparison",
                                 color_pred='red', color_gt='green', max_points=25000):
    """
    Visualize two pointclouds (Pred and GT) together using Plotly.
    Returns the figure object for wandb logging.
    """
    if not isinstance(pc_pred, np.ndarray):
        pc_pred = np.array(pc_pred)
    if not isinstance(pc_gt, np.ndarray):
        pc_gt = np.array(pc_gt)
    
    # Downsample both pointclouds if too large
    if len(pc_pred) > max_points:
        idx = np.random.choice(len(pc_pred), max_points, replace=False)
        pc_pred = pc_pred[idx]
    if len(pc_gt) > max_points:
        idx = np.random.choice(len(pc_gt), max_points, replace=False)
        pc_gt = pc_gt[idx]
    
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pc_pred[:, 0], y=pc_pred[:, 1], z=pc_pred[:, 2],
        mode='markers',
        name='Pred',
        marker=dict(size=2, color=color_pred, opacity=0.8)
    ))

    fig.add_trace(go.Scatter3d(
        x=pc_gt[:, 0], y=pc_gt[:, 1], z=pc_gt[:, 2],
        mode='markers',
        name='GT',
        marker=dict(size=2, color=color_gt, opacity=0.8)
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.02, y=0.98)
    )

    return fig


# ============================================================================
# CHECKPOINT AND SPLAT UTILITIES
# ============================================================================

def load_checkpoint(checkpoint_path, device):
    """Load the saved checkpoint from training"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def load_gaussian_predictor(cfg, device):
    """Load the Gaussian Splatter Image predictor model"""
    gaussian_predictor = GaussianSplatPredictor(cfg)
    gaussian_predictor = gaussian_predictor.to(memory_format=torch.channels_last)
    gaussian_predictor = gaussian_predictor.to(device)
    
    if cfg.opt.pretrained_ckpt is not None:
        model_path = cfg.opt.pretrained_ckpt
        model_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        gaussian_predictor.load_state_dict(model_checkpoint["model_state_dict"])
        print(f'Loaded splatter image model from {model_path}')
    
    gaussian_predictor.eval()
    return gaussian_predictor


def regenerate_baseline_indist_splats(checkpoint, gaussian_predictor, cfg, device):
    """
    Regenerate the baseline in-distribution splats (GT).
    This uses the OOD image from the original in-distribution view.
    """
    ood_data = checkpoint["ood_data"]
    ood_image_index = checkpoint["ood_image_index"]
    
    # Move to device
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    
    # Use the in-distribution image (not the OOD rotated one)
    ood_image = ood_data["gt_images"][0, ood_image_index].to(device)
    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    
    # Prepare input
    input_images = torch.cat([ood_image.unsqueeze(0).unsqueeze(1), 
                              ood_origin_distances.unsqueeze(0).unsqueeze(1)], dim=2)
    focals_pixels_pred = ood_focals_pixels.unsqueeze(0).unsqueeze(0)
    
    # Run splatter image predictor
    with torch.no_grad():
        pred_splats = gaussian_predictor(
            input_images, 
            ood_data["view_to_world_transforms"][:1, 0:cfg.data.input_images], 
            ood_data["source_cv2wT_quat"][:1, 0:cfg.data.input_images], 
            focals_pixels_pred   
        )
    
    # Extract batch 0
    pred_splats = {k: v[0] for k, v in pred_splats.items()}
    
    print(f"Baseline in-dist splats: {pred_splats['xyz'].shape[0]} points")
    return pred_splats


def regenerate_baseline_ood_splats(checkpoint, baseline_indist_splats, gaussian_predictor, cfg, device):
    """
    Regenerate baseline OOD splats.
    Renders baseline_indist with OOD rotation, feeds back to predictor, undoes rotation.
    """
    background = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], 
                              dtype=torch.float32, device=device)
    ood_data = checkpoint['ood_data']
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    ood_image_index = checkpoint['ood_image_index']
    ood_random_rotation = checkpoint["ood_random_rotation"].to(device)
    assert ood_random_rotation is not None, "ood_random_rotation is not set"
    ood_zgt = checkpoint["zgt"]
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    ood_centroid = checkpoint.get("ood_centroid")
    if ood_centroid is None:  # StyleGAN ckpts omit it; centroid is [0,0,zgt] by construction
        ood_centroid = torch.tensor([0., 0., float(checkpoint["zgt"])], device=device)
    assert ood_centroid is not None, "ood_centroid not found in checkpoint"
    # Render baseline_indist with OOD rotation
    ood_render_image = render_with_custom_camera(baseline_indist_splats, background, cfg, 
                                                   ood_focals_pixels, 
                                                   ood_random_rotation, ood_zgt, device=device, zgt_ood=ood_zgt, translation=None)
    ood_render_image = ood_render_image.squeeze(0)
    
    # Use the rendered OOD image
    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    
    # Prepare input
    input_images = torch.cat([ood_render_image.unsqueeze(0).unsqueeze(1), 
                              ood_origin_distances.unsqueeze(0).unsqueeze(1)], dim=2)
    focals_pixels_pred = ood_focals_pixels.unsqueeze(0).unsqueeze(0)
    
    # Generate splats
    with torch.no_grad():
        pred_splats = gaussian_predictor(
            input_images, 
            ood_data["view_to_world_transforms"][:1, 0:cfg.data.input_images], 
            ood_data["source_cv2wT_quat"][:1, 0:cfg.data.input_images], 
            focals_pixels_pred   
        )
        pred_splats = {k: v[0] for k, v in pred_splats.items()}
    
    # Undo the OOD rotation
    if ood_random_rotation is not None:
        inverse_rotation = ood_random_rotation.T  # Transpose for inverse rotation
        pred_splats = render_with_custom_camera(
            pred_splats, background, cfg, ood_focals_pixels,
            inverse_rotation, ood_zgt, device=device, return_splats=True, translation=None, zgt_ood=ood_zgt
        )
    
    return pred_splats, ood_data


# `regenerate_ours_splats` lives in experiments/co3d_ood/regenerate_ours.py so both eval
# entrypoints share one implementation (they had silently drifted apart).
# Variant is selected by $INFER3D_OURS_REGEN ("paper" default, or "aligned").
from experiments.co3d_ood.regenerate_ours import regenerate_ours_splats  # noqa: E402



def apply_se3_to_splats(splats, R, t, scale=1.0, device=None):
    """
    Apply SE(3) transformation (rotation + translation + scale) to Gaussian splats.
    
    Args:
        splats: Dictionary with 'xyz', 'rotation', etc.
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        scale: Scale factor
        device: torch device
    
    Returns:
        transformed_splats: Splats with transformed positions and rotations
    """
    transformed_splats = {}
    
    R_tensor = torch.tensor(R, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
    
    for k, v in splats.items():
        if k == 'xyz':
            # Transform positions: scale * (R @ xyz) + t
            transformed_xyz = scale * (R_tensor @ v.T).T + t_tensor
            transformed_splats[k] = transformed_xyz.contiguous()
        
        elif k == 'rotation':
            # Transform rotations (quaternions)
            R_quat = matrix_to_quaternion(R_tensor)
            R_quat_expanded = R_quat.unsqueeze(0).expand(*v.shape)
            transformed_rotation = quaternion_raw_multiply(R_quat_expanded, v)
            transformed_splats[k] = transformed_rotation.contiguous()
        
        else:
            # Keep other attributes unchanged
            transformed_splats[k] = v.contiguous()
    
    return transformed_splats


def apply_icp_to_splats(splats, icp_solution, centroid_offset, device):
    """
    Apply ICP transformation to splats, accounting for centroid alignment.

    Since ICP was computed on centroid-aligned pointclouds, we need to:
    1. Apply centroid offset to move splats to centroid-aligned space
    2. Apply ICP transformation

    Args:
        splats: Dictionary with 'xyz', 'rotation', etc.
        icp_solution: ICP solution from pytorch3d
        centroid_offset: Offset applied before ICP (gt_centroid - pred_centroid)
        device: torch device
    """
    assert icp_solution is not None, "ICP solution is None"

    # Extract ICP transformation parameters
    R_icp = icp_solution.RTs.R.squeeze(0).T.cpu().numpy()  # 3, 3
    T_icp = icp_solution.RTs.T.squeeze(0).cpu().numpy()  # 3
    s_icp = icp_solution.RTs.s.item()  # 1

    # Step 1: Apply centroid offset
    splats_centered = apply_se3_to_splats(splats, np.eye(3), centroid_offset, 1.0, device=device)

    # Step 2: Apply ICP transformation
    transformed_splats = apply_se3_to_splats(splats_centered, R_icp, T_icp, s_icp, device=device)

    return transformed_splats






# ============================================================================
# RENDERING AND EVALUATION UTILITIES
# ============================================================================

def render_mask(splats, world_view_transform, full_proj_transform, camera_center,
                cfg, device, focals_pixels=None):
    """Render binary mask by overriding splat colors to white."""
    white_color = torch.ones_like(splats["features_dc"])
    background_white = torch.ones(3, dtype=torch.float32, device=device)

    mask_output = render_predicted(
        splats,
        world_view_transform,
        full_proj_transform,
        camera_center,
        background_white,
        cfg,
        override_color=white_color,
        focals_pixels=focals_pixels
    )
    return foreground_mask(mask_output["render"], thresh=0.98)


def compute_masked_metrics(image, gt_image, mask_pred, mask_gt, lpips_fn, ssim_fn):
    """Compute metrics on union of masks."""
    # Union for metrics, intersection for IOU
    mask_union = mask_pred | mask_gt
    mask_intersection = mask_pred & mask_gt

    # IOU
    iou = mask_intersection.sum().float() / (mask_union.sum().float() + 1e-8)

    # Masked PSNR: MSE over union pixels only
    diff = (image - gt_image) ** 2  # [3, H, W]
    masked_diff = diff[:, mask_union]  # [3, N_union]
    mse = masked_diff.mean()
    psnr_masked = -10 * torch.log10(mse + 1e-8)

    # Masked SSIM: compute on masked regions
    # Create masked images (set background to 0)
    image_masked = image.clone()
    gt_masked = gt_image.clone()
    image_masked[:, ~mask_union] = 0
    gt_masked[:, ~mask_union] = 0
    ssim_masked = ssim_fn(image_masked, gt_masked)

    # Masked LPIPS: apply mask after computation
    lpips_masked = lpips_fn(
        image.unsqueeze(0) * 2 - 1,
        gt_image.unsqueeze(0) * 2 - 1
    )

    return {
        'psnr_masked': psnr_masked.item(),
        'ssim_masked': ssim_masked.item(),
        'lpips_masked': lpips_masked.item(),
        'mask_iou': iou.item()
    }

def make_data_relative_to_idx( images_and_camera_poses, image_index):
    inverse_idx_camera = images_and_camera_poses["world_view_transforms"][:, image_index].inverse().clone()
    for c in range(images_and_camera_poses["world_view_transforms"].shape[1]):
        images_and_camera_poses["world_view_transforms"][:, c] = torch.bmm(
                                        inverse_idx_camera,
                                        images_and_camera_poses["world_view_transforms"][:, c])
        images_and_camera_poses["view_to_world_transforms"][:, c] = torch.bmm(
                                            images_and_camera_poses["view_to_world_transforms"][:, c],
                                            inverse_idx_camera.inverse())
        images_and_camera_poses["full_proj_transforms"][:, c] = torch.bmm(
                                            inverse_idx_camera,
                                            images_and_camera_poses["full_proj_transforms"][:, c])
        images_and_camera_poses["camera_centers"][:, c] = images_and_camera_poses["world_view_transforms"][:, c].inverse()[:, 3, :3]
    
    return images_and_camera_poses

def render_splats_and_compute_metrics(splats, ood_data, cfg, device, method_name="baseline", 
                                     output_dir=None, wandb_run=None, object_id=None, ood_image_index=0):
    """
    Render Gaussian splats from saved camera poses and compute metrics.
    
    Returns:
        scores: Dictionary with PSNR, SSIM, LPIPS
        video_path: Path to saved video
        invdepth_video_path: Path to saved inverse depth video
    """
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)


    if ood_image_index != 0: 
        from copy import deepcopy
        deep_clone_ood_data = deepcopy(ood_data)
        relative_ood_data = make_data_relative_to_idx(deep_clone_ood_data, ood_image_index)
    else:
        relative_ood_data = ood_data
    
    # Move data to device
    relative_ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in relative_ood_data.items()}
    splats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in splats.items()}
    
    # Set up rendering background
    background = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    
    psnr_all = []
    ssim_all = []
    lpips_all = []
    psnr_masked_all = []
    ssim_masked_all = []
    lpips_masked_all = []
    mask_iou_all = []
    rendered_images = []
    rendered_invdepths = []

    num_cameras = relative_ood_data["world_view_transforms"].shape[1]
    

    # Render all views
    for r_idx in range(num_cameras):
        world_view_transforms = relative_ood_data["world_view_transforms"][0, r_idx].unsqueeze(0)
        full_proj_transforms = relative_ood_data["full_proj_transforms"][0, r_idx].unsqueeze(0)
        camera_centers = relative_ood_data["camera_centers"][0, r_idx].unsqueeze(0)

        # Get focal pixels for this camera
        cur_focals_pixels = relative_ood_data["focals_pixels"][0, r_idx] if "focals_pixels" in relative_ood_data else None

        # Render
        with torch.no_grad():
            render_output = render_predicted(
                splats,
                world_view_transforms,
                full_proj_transforms,
                camera_centers,
                background,
                cfg,
                focals_pixels=cur_focals_pixels
            )
            image = render_output["render"]
            invdepth = render_output["invdepths"]

        # Compute metrics
        gt_image = relative_ood_data["gt_images"][0, r_idx].to(device)
        psnr = -10 * torch.log10(torch.mean((image - gt_image) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, gt_image).item()
        lpips = lpips_fn(image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).item()

        psnr_all.append(psnr)
        ssim_all.append(ssim)
        lpips_all.append(lpips)

        # Render masks and compute masked metrics
        with torch.no_grad():
            mask_pred = render_mask(splats, world_view_transforms, full_proj_transforms,
                                   camera_centers, cfg, device, cur_focals_pixels)
            # For GT mask, we need to render GT splats - but we don't have them here
            # So we'll use the GT image to generate a mask
            mask_gt = foreground_mask(gt_image, thresh=0.98)

            masked_metrics = compute_masked_metrics(image, gt_image, mask_pred, mask_gt,
                                                   lpips_fn, ssim_fn)
            psnr_masked_all.append(masked_metrics['psnr_masked'])
            ssim_masked_all.append(masked_metrics['ssim_masked'])
            lpips_masked_all.append(masked_metrics['lpips_masked'])
            mask_iou_all.append(masked_metrics['mask_iou'])
        
        # Convert RGB to numpy
        rendered_image = torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rendered_images.append(rendered_image)
        
        # Store inverse depth
        invdepth_squeezed = invdepth.squeeze(0) if invdepth.dim() == 3 else invdepth
        rendered_invdepths.append(invdepth_squeezed.detach().cpu())
    
    # Compute aggregate scores
    scores = {
        "PSNR": np.mean(psnr_all),
        "SSIM": np.mean(ssim_all),
        "LPIPS": np.mean(lpips_all),
        "PSNR_masked": np.mean(psnr_masked_all),
        "SSIM_masked": np.mean(ssim_masked_all),
        "LPIPS_masked": np.mean(lpips_masked_all),
        "mask_IOU": np.mean(mask_iou_all),
    }
    
    # Normalize inverse depths globally
    invdepths_tensor = torch.stack(rendered_invdepths)
    invdepth_min = invdepths_tensor.min()
    invdepth_max = invdepths_tensor.max()
    
    if invdepth_max > invdepth_min:
        invdepths_normalized = (invdepths_tensor - invdepth_min) / (invdepth_max - invdepth_min)
    else:
        invdepths_normalized = torch.zeros_like(invdepths_tensor)
    
    # Apply colormap for depth visualization
    cmap = plt.get_cmap('turbo')
    invdepth_images = []
    for i in range(len(invdepths_normalized)):
        invdepth_np = invdepths_normalized[i].numpy()
        invdepth_colored = (cmap(invdepth_np)[:, :, :3] * 255).astype(np.uint8)
        invdepth_images.append(invdepth_colored)
    
    # Save videos if output_dir provided
    video_path = None
    invdepth_video_path = None
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        video_filename = f"{object_id}_{method_name}.mp4" if object_id else f"{method_name}.mp4"
        video_path = os.path.join(output_dir, f"rendered_{video_filename}")
        imageio.mimsave(video_path, rendered_images, fps=4, codec='libx264')
        
        invdepth_video_path = os.path.join(output_dir, f"invdepth_{video_filename}")
        imageio.mimsave(invdepth_video_path, invdepth_images, fps=4, codec='libx264')
    
    return scores, video_path, invdepth_video_path


# ============================================================================
# PATH UTILITIES
# ============================================================================

def extract_object_id_from_path(ply_path):
    """
    Extract object_id from pointcloud path.
    Expected format: .../{object_id}_{...}_{...}_pointcloud.ply
    
    Returns object_id string.
    """
    filename = os.path.basename(ply_path)
    # Remove _pointcloud.ply suffix
    if filename.endswith('_pointcloud.ply'):
        filename = filename[:-len('_pointcloud.ply')]
    
    # Split by underscore and take first part as object_id
    parts = filename.split('_')
    assert len(parts) == 3, f"Expected 3 parts in filename: {filename}"
    return "_".join(parts)


def load_pointcloud_paths(txt_path):
    """
    Load pointcloud paths from text file.
    Returns dict mapping object_id -> ply_path.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Path list file not found: {txt_path}")
    
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) == 0:
        raise ValueError(f"No paths found in {txt_path}")
    
    path_dict = {}
    for line in lines:
        if not os.path.exists(line):
            print(f"WARNING: Path does not exist, skipping: {line}")
            continue
        
        object_id = extract_object_id_from_path(line)
        
        if object_id in path_dict:
            raise ValueError(f"Duplicate object_id found: {object_id} in {txt_path}")
        
        path_dict[object_id] = line
    
    return path_dict


def parse_checkpoint_folder_name(folder_name):
    """
    Parse folder name to extract object_id and datetime.
    Format: {object_id}-{DD-MM-YY-HH-MM-SS}
    
    Returns:
        (object_id, datetime_obj) or (None, None) if parsing fails
    """
    # Parse datetime: DD-MM-YY-HH-MM-SS
    try:
        parts = folder_name.split('-')
        if len(parts) < 7:
            raise ValueError(f"Could not parse folder name: {folder_name}")
            return None, None
        
        object_id = parts[0]
        
        day, month, year, hour, minute, second = parts[1:7]
        # Convert 2-digit year to 4-digit (assuming 2000s)
        year_full = f"20{year}"
        dt = datetime(int(year_full), int(month), int(day), int(hour), int(minute), int(second))
        return object_id, dt
    except Exception as e:
        print(f"Error parsing folder name: {folder_name}")
        raise e
        # return None, None


def find_latest_checkpoints(checkpoint_dir):
    """
    Find all checkpoints in checkpoint_dir, group by object_id, and return latest for each.
    
    Returns:
        List of (object_id, checkpoint_path) tuples
    """
    # Find all subdirectories with topk_best_everything_latest.pth
    checkpoint_pattern = os.path.join(checkpoint_dir, "*", "topk_best_everything_latest.pth")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Parse and group by object_id
    object_checkpoints = {}  # object_id -> [(datetime, checkpoint_path)]
    
    for ckpt_path in checkpoint_files:
        folder_name = os.path.basename(os.path.dirname(ckpt_path))
        object_id, dt = parse_checkpoint_folder_name(folder_name)
        
        if object_id is None:
            print(f"WARNING: Could not parse folder name: {folder_name}")
            raise ValueError(f"Could not parse folder name: {folder_name}")
        
        if object_id not in object_checkpoints:
            object_checkpoints[object_id] = []
        
        object_checkpoints[object_id].append((dt, ckpt_path))
    
    # For each object, take the latest checkpoint
    latest_checkpoints = []
    for object_id, checkpoints in object_checkpoints.items():
        # Sort by datetime (latest last)
        checkpoints.sort(key=lambda x: x[0])
        latest_dt, latest_path = checkpoints[-1]
        latest_checkpoints.append((object_id, latest_path))
        print(f"Object {object_id}: {len(checkpoints)} checkpoints found, latest from {latest_dt}")
    
    return latest_checkpoints


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_checkpoint(checkpoint_path, object_id, pred_ply, gt_ply, 
                       gaussian_predictor, cfg, device,
                       pred_method="ours",
                       num_points=100000, 
                       max_iterations=100, 
                       estimate_scale=True,
                       output_dir=None,
                       wandb_run=None):
    """

    
    Evaluate a single checkpoint:
    1. Load pred and GT pointcloud PLY files from disk
    2. Run ICP alignment on the PLY pointclouds
    3. Load checkpoint and regenerate splats
    4. Apply ICP transformation to pred splats (ours or baseline_ood)
    5. Render all three versions (baseline_indist, pred, pred+ICP) and compute metrics
    6. Log everything to wandb
    
    Args:
        pred_method: "ours" or "baseline_ood" - which method to evaluate
    
    Returns dict with evaluation results.

    """
    print(f"\n{'='*80}")
    print(f"EVALUATING CHECKPOINT: {object_id}")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Pred PLY: {pred_ply}")
    print(f"GT PLY:   {gt_ply}")
    
    # Load pointclouds from PLY files
    print(f"\nLoading pointclouds from PLY files...")
    pred_points = load_pointcloud(pred_ply, num_points)
    gt_points = load_pointcloud(gt_ply, num_points)
    print(f"  Pred: {len(pred_points)} points")
    print(f"  GT:   {len(gt_points)} points")

    # Compute centroids and align pred to GT centroid
    pred_centroid = np.mean(pred_points, axis=0)
    gt_centroid = np.mean(gt_points, axis=0)
    centroid_offset = gt_centroid - pred_centroid

    print(f"\nCentroid alignment:")
    print(f"  Pred centroid: {pred_centroid}")
    print(f"  GT centroid:   {gt_centroid}")
    print(f"  Offset:        {centroid_offset}")

    # Apply centroid alignment to pred pointcloud (helps ICP convergence)
    pred_points_aligned = pred_points + centroid_offset
    # NOTE: We must later apply this same centroid_offset to splats before ICP transformation

    # Convert to torch tensors for ICP
    X = torch.from_numpy(pred_points_aligned).float().unsqueeze(0).to(device)  # [1, N, 3]
    Y = torch.from_numpy(gt_points).float().unsqueeze(0).to(device)    # [1, M, 3]
    
    # Run ICP
    print(f"\nRunning ICP alignment...")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Estimate scale: {estimate_scale}")
    
    icp_solution = iterative_closest_point(
        X, Y,
        estimate_scale=estimate_scale,
        max_iterations=max_iterations,
        verbose=False
    )
    
    # Extract ICP results
    converged = icp_solution.converged.item() if torch.is_tensor(icp_solution.converged) else icp_solution.converged
    rmse = icp_solution.rmse.item()
    rotation = icp_solution.RTs.R[0].cpu().numpy()
    translation = icp_solution.RTs.T[0].cpu().numpy()
    scale = icp_solution.RTs.s[0].item()
    
    print(f"\nICP Results:")
    print(f"  Converged: {converged}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Scale: {scale:.6f}")
    
    # Get aligned pointcloud
    Xt = icp_solution.Xt
    Xt_np = Xt[0].cpu().numpy()
    Y_np = Y[0].cpu().numpy()
    
    # Compute Chamfer Distance
    print(f"\nComputing Chamfer Distance...")
    chamfer, pred_to_gt, gt_to_pred = compute_chamfer_distance(Xt_np, Y_np)
    
    print(f"  Pred -> GT: {pred_to_gt:.6f}")
    print(f"  GT -> Pred: {gt_to_pred:.6f}")
    print(f"  Chamfer Distance: {chamfer:.6f}")
    
    # NOW load checkpoint and regenerate splats
    print(f"\nLoading checkpoint and regenerating splats...")
    checkpoint = load_checkpoint(checkpoint_path, device)
    example_id = checkpoint['example_id']
    if 'ood_data' not in checkpoint:
        print(f"ood_data not found in checkpoint, reconstructing from val_dataset using example_id: {example_id}")
        
        # Load validation dataset (use_hq=False for standard 128x128 resolution)
        val_dataset = CO3DDataset(cfg, "test", use_hq=False)
        
        # Find and load the sequence
        ood_seq_index, found_example_id = val_dataset.find_index_for_sequence_prefix(example_id)
        ood_data = val_dataset[ood_seq_index]

        # add view batch dimension to ood_data
        ood_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in ood_data.items()}
        # Add to checkpoint for consistency with rest of the code
        checkpoint['ood_data'] = ood_data
        
        print(f"Reconstructed ood_data for example_id: {found_example_id}")
    else:
        print(f"Using ood_data from checkpoint (old format)")
    
    gt_splats = regenerate_baseline_indist_splats(checkpoint, gaussian_predictor, cfg, device)

    # Regenerate pred splats based on method
    if pred_method == "ours":
        print(f"Regenerating OURS splats (optimized with rotation/translation)...")
        # Trivial-TTT support: load per-scene fine-tuned weights for OURS only.
        pretrained_state_dict_backup = None
        if "finetuned_state_dict" in checkpoint:
            print("Loading per-scene finetuned_state_dict for OURS regeneration")
            pretrained_state_dict_backup = {k: v.detach().clone() for k, v in gaussian_predictor.state_dict().items()}
            gaussian_predictor.load_state_dict({k: v.to(device) for k, v in checkpoint["finetuned_state_dict"].items()})
        pred_splats, ood_data = regenerate_ours_splats(checkpoint, gaussian_predictor, cfg, device)
        if pretrained_state_dict_backup is not None:
            gaussian_predictor.load_state_dict(pretrained_state_dict_backup)
            print("Restored pretrained weights")
    elif pred_method == "baseline_ood":
        print(f"Regenerating BASELINE_OOD splats (baseline with OOD rotation)...")
        pred_splats, ood_data = regenerate_baseline_ood_splats(checkpoint, gt_splats, gaussian_predictor, cfg, device)
    else:
        raise ValueError(f"Unknown pred_method: {pred_method}. Must be 'ours' or 'baseline_ood'")

    # Reconstruct the actual OOD image used during optimization (random-rotation render)
    ood_image_tensor = None
    if "ood_random_rotation" in checkpoint:
        ood_image_index = checkpoint["ood_image_index"]
        ood_random_rotation = checkpoint["ood_random_rotation"].to(device)
        zgt = checkpoint["zgt"]
        ood_centroid = checkpoint["ood_centroid"].to(device) if checkpoint.get("ood_centroid") is not None \
            else torch.tensor([0., 0., float(zgt)], device=device)  # StyleGAN ckpts omit it
        ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
        background = torch.tensor(
            [1, 1, 1] if cfg.data.white_background else [0, 0, 0],
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            ood_image_tensor, _centroid, _depth = render_with_custom_camera(
                gt_splats,
                background,
                cfg,
                ood_focals_pixels,
                ood_random_rotation,
                zgt,
                device=device,
                return_splats=False, 
                translation=None, 
                zgt_ood=zgt,
                return_centroid = True, 
                return_depth=True,
                # override_centroid=ood_centroid,
            )
            ood_image_tensor = ood_image_tensor.squeeze(0).detach().cpu()
    else:
        raise ValueError(f"ood_random_rotation not found in checkpoint")
    # Apply ICP transformation to pred splats
    print(f"\nApplying ICP transformation to {pred_method} splats...")
    pred_splats_icp = apply_icp_to_splats(pred_splats, icp_solution, centroid_offset, device)
    
    # Render and evaluate all three versions
    print(f"\nRendering and evaluating splats...")
    
    scores_gt, video_gt, invdepth_video_gt = render_splats_and_compute_metrics(
        gt_splats, ood_data, cfg, device, "baseline_indist", output_dir, wandb_run, object_id, 
        ood_image_index=ood_image_index
    )
    print(f"  Baseline GT - PSNR: {scores_gt['PSNR']:.4f}, SSIM: {scores_gt['SSIM']:.4f}, LPIPS: {scores_gt['LPIPS']:.4f}")
    
    scores_pred, video_pred, invdepth_video_pred = render_splats_and_compute_metrics(
        pred_splats, ood_data, cfg, device, pred_method, output_dir, wandb_run, object_id,
        ood_image_index=ood_image_index
    )
    print(f"  {pred_method.upper()} - PSNR: {scores_pred['PSNR']:.4f}, SSIM: {scores_pred['SSIM']:.4f}, LPIPS: {scores_pred['LPIPS']:.4f}")
    
    scores_pred_icp, video_pred_icp, invdepth_video_pred_icp = render_splats_and_compute_metrics(
        pred_splats_icp, ood_data, cfg, device, f"{pred_method}_icp_aligned", output_dir, wandb_run, object_id,
        ood_image_index=ood_image_index
    )
    print(f"  {pred_method.upper()} ICP - PSNR: {scores_pred_icp['PSNR']:.4f}, SSIM: {scores_pred_icp['SSIM']:.4f}, LPIPS: {scores_pred_icp['LPIPS']:.4f}")
    
    # Prepare results
    results = {
        'object_id': object_id,
        'pred_method': pred_method,
        'chamfer': chamfer,
        'pred_to_gt': pred_to_gt,
        'gt_to_pred': gt_to_pred,
        'icp_converged': converged,
        'icp_rmse': rmse,
        'icp_scale': scale,
        'baseline_psnr': scores_gt['PSNR'],
        'baseline_ssim': scores_gt['SSIM'],
        'baseline_lpips': scores_gt['LPIPS'],
        'baseline_psnr_masked': scores_gt['PSNR_masked'],
        'baseline_ssim_masked': scores_gt['SSIM_masked'],
        'baseline_lpips_masked': scores_gt['LPIPS_masked'],
        'baseline_mask_iou': scores_gt['mask_IOU'],
        f'{pred_method}_psnr': scores_pred['PSNR'],
        f'{pred_method}_ssim': scores_pred['SSIM'],
        f'{pred_method}_lpips': scores_pred['LPIPS'],
        f'{pred_method}_psnr_masked': scores_pred['PSNR_masked'],
        f'{pred_method}_ssim_masked': scores_pred['SSIM_masked'],
        f'{pred_method}_lpips_masked': scores_pred['LPIPS_masked'],
        f'{pred_method}_mask_iou': scores_pred['mask_IOU'],
        f'{pred_method}_icp_psnr': scores_pred_icp['PSNR'],
        f'{pred_method}_icp_ssim': scores_pred_icp['SSIM'],
        f'{pred_method}_icp_lpips': scores_pred_icp['LPIPS'],
        f'{pred_method}_icp_psnr_masked': scores_pred_icp['PSNR_masked'],
        f'{pred_method}_icp_ssim_masked': scores_pred_icp['SSIM_masked'],
        f'{pred_method}_icp_lpips_masked': scores_pred_icp['LPIPS_masked'],
        f'{pred_method}_icp_mask_iou': scores_pred_icp['mask_IOU'],
    }
    
    # Log to wandb if provided
    if wandb_run is not None:
        wandb_log_dict = {
            f'{object_id}/pred_method': pred_method,
            f'{object_id}/chamfer': chamfer,
            f'{object_id}/pred_to_gt': pred_to_gt,
            f'{object_id}/gt_to_pred': gt_to_pred,
            f'{object_id}/icp_converged': int(converged),
            f'{object_id}/icp_rmse': rmse,
            f'{object_id}/icp_scale': scale,
            f'{object_id}/baseline_psnr': scores_gt['PSNR'],
            f'{object_id}/baseline_ssim': scores_gt['SSIM'],
            f'{object_id}/baseline_lpips': scores_gt['LPIPS'],
            f'{object_id}/baseline_psnr_masked': scores_gt['PSNR_masked'],
            f'{object_id}/baseline_ssim_masked': scores_gt['SSIM_masked'],
            f'{object_id}/baseline_lpips_masked': scores_gt['LPIPS_masked'],
            f'{object_id}/baseline_mask_iou': scores_gt['mask_IOU'],
            f'{object_id}/{pred_method}_psnr': scores_pred['PSNR'],
            f'{object_id}/{pred_method}_ssim': scores_pred['SSIM'],
            f'{object_id}/{pred_method}_lpips': scores_pred['LPIPS'],
            f'{object_id}/{pred_method}_psnr_masked': scores_pred['PSNR_masked'],
            f'{object_id}/{pred_method}_ssim_masked': scores_pred['SSIM_masked'],
            f'{object_id}/{pred_method}_lpips_masked': scores_pred['LPIPS_masked'],
            f'{object_id}/{pred_method}_mask_iou': scores_pred['mask_IOU'],
            f'{object_id}/{pred_method}_icp_psnr': scores_pred_icp['PSNR'],
            f'{object_id}/{pred_method}_icp_ssim': scores_pred_icp['SSIM'],
            f'{object_id}/{pred_method}_icp_lpips': scores_pred_icp['LPIPS'],
            f'{object_id}/{pred_method}_icp_psnr_masked': scores_pred_icp['PSNR_masked'],
            f'{object_id}/{pred_method}_icp_ssim_masked': scores_pred_icp['SSIM_masked'],
            f'{object_id}/{pred_method}_icp_lpips_masked': scores_pred_icp['LPIPS_masked'],
            f'{object_id}/{pred_method}_icp_mask_iou': scores_pred_icp['mask_IOU'],
        }

        # Log OOD image rendered with the stored random rotation
        if ood_image_tensor is not None:
            wandb_log_dict[f'{object_id}/ood_image'] = wandb.Image(torch.clamp(ood_image_tensor, 0.0, 1.0), caption="OOD Image")
        
        # Log best_pred_image and best_input_image from checkpoint
        if "best_pred_image" in checkpoint and checkpoint["best_pred_image"] is not None:
            wandb_log_dict[f'{object_id}/best_pred_image'] = wandb.Image(torch.clamp(checkpoint["best_pred_image"], 0.0, 1.0), caption="Best Pred Image")
        if "best_input_image" in checkpoint:
            best_input_rgb = checkpoint["best_input_image"].squeeze(0)[:3, :, :]
            wandb_log_dict[f'{object_id}/best_input_image'] = wandb.Image(best_input_rgb, caption="Best Input Image")
        
        # Create and log GT video
        gt_video_frames = []
        for i in range(ood_data["gt_images"].shape[1]):
            gt_frame = torch.clamp(ood_data["gt_images"][0, i] * 255, 0.0, 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            gt_video_frames.append(gt_frame)
        if output_dir is not None:
            gt_video_path = os.path.join(output_dir, f"{object_id}_gt.mp4")
            imageio.mimsave(gt_video_path, gt_video_frames, fps=4, codec='libx264')
            wandb_log_dict[f'{object_id}/video_gt'] = wandb.Video(gt_video_path, fps=4, format="mp4")
        
        # Create pointcloud visualizations
        try:
            # Before ICP alignment (using centroid-aligned pred points)
            fig_before = visualize_pointclouds_plotly(
                pred_points_aligned, gt_points,
                title=f"{object_id} - Before ICP Alignment (after centroid alignment)"
            )
            wandb_log_dict[f'{object_id}/pointcloud_before_icp'] = wandb.Plotly(fig_before)
            
            # After ICP alignment
            fig_after = visualize_pointclouds_plotly(
                Xt_np, Y_np,
                title=f"{object_id} - After ICP Alignment"
            )
            wandb_log_dict[f'{object_id}/pointcloud_after_icp'] = wandb.Plotly(fig_after)
            
        except Exception as e:
            print(f"WARNING: Could not create pointcloud visualizations: {e}")
        
        # Log rendering videos
        if video_gt and os.path.exists(video_gt):
            wandb_log_dict[f'{object_id}/video_baseline'] = wandb.Video(video_gt, fps=4, format="mp4")
        if video_pred and os.path.exists(video_pred):
            wandb_log_dict[f'{object_id}/video_{pred_method}'] = wandb.Video(video_pred, fps=4, format="mp4")
        if video_pred_icp and os.path.exists(video_pred_icp):
            wandb_log_dict[f'{object_id}/video_{pred_method}_icp'] = wandb.Video(video_pred_icp, fps=4, format="mp4")
        
        if invdepth_video_gt and os.path.exists(invdepth_video_gt):
            wandb_log_dict[f'{object_id}/invdepth_baseline'] = wandb.Video(invdepth_video_gt, fps=4, format="mp4")
        if invdepth_video_pred and os.path.exists(invdepth_video_pred):
            wandb_log_dict[f'{object_id}/invdepth_{pred_method}'] = wandb.Video(invdepth_video_pred, fps=4, format="mp4")
        if invdepth_video_pred_icp and os.path.exists(invdepth_video_pred_icp):
            wandb_log_dict[f'{object_id}/invdepth_{pred_method}_icp'] = wandb.Video(invdepth_video_pred_icp, fps=4, format="mp4")
        
        wandb_run.log(wandb_log_dict)
        print(f"\nLogged results to wandb for {object_id}")
    
    return results


def main(checkpoint_dir, pred_list, gt_list, results_csv, 
         pred_method="ours",
         num_points=100000, 
         max_iterations=100, 
         estimate_scale=True,
         output_dir=None,
         config_overrides=None,
         wandb_project=None, 
         wandb_run_name=None,
         dataset_name="hydrants",
         pretrained_ckpt=None):
    """
    Main pipeline: Find latest checkpoints in directory, load pointcloud PLY lists, match by object_id and process them.
    
    Args:
        pred_method: "ours" or "baseline_ood" - which method to evaluate
    """
    print("="*80)
    print("BATCH RENDERING EVALUATION WITH ICP ALIGNMENT")
    print("="*80)
    
    # Initialize wandb if project name provided
    wandb_run = None
    if wandb_project is not None:
        if wandb_run_name is None:
            wandb_run_name = f"{dataset_name}_rendering_eval_{Path(checkpoint_dir).name}"
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "checkpoint_dir": checkpoint_dir,
                "pred_list": pred_list,
                "gt_list": gt_list,
                "results_csv": results_csv,
                "pred_method": pred_method,
                "num_points": num_points,
                "max_iterations": max_iterations,
                "estimate_scale": estimate_scale,
            }
        )
        print(f"Initialized wandb project: {wandb_project}, run: {wandb_run_name}")
    
    # Load config
    print(f"\nLoading configuration...")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=_cfg.CONFIGS_DIR)
    
    default_overrides = [
        "general.split=0", 
        "general.total_splits=1", 
        f"+dataset={dataset_name}", 
        "general.prefix=checkpoints-eval",
        "abs=diffae_abs", 
        f"opt.pretrained_ckpt={pretrained_ckpt}",
        "general.data_example_ids_path=not_needed.json"
    ]
    
    if config_overrides is not None:
        default_overrides.extend(config_overrides)
    
    cfg = compose(config_name="abs_config", overrides=default_overrides)
    print("Configuration loaded")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Gaussian predictor
    print(f"\nLoading Gaussian predictor model...")
    gaussian_predictor = load_gaussian_predictor(cfg, device)
    
    # Find latest checkpoints
    print(f"\nFinding latest checkpoints in {checkpoint_dir}...")
    latest_checkpoints = find_latest_checkpoints(checkpoint_dir)
    checkpoint_paths = dict(latest_checkpoints)  # Convert to dict: object_id -> checkpoint_path
    print(f"  Found {len(checkpoint_paths)} checkpoints")
    
    # Load pointcloud paths
    print(f"\nLoading pointcloud paths...")
    pred_paths = load_pointcloud_paths(pred_list)
    gt_paths = load_pointcloud_paths(gt_list)
    print(f"  Found {len(pred_paths)} pred pointclouds")
    print(f"  Found {len(gt_paths)} gt pointclouds")
    
    # Find matching triplets (checkpoint, pred PLY, gt PLY)
    common_ids = set(checkpoint_paths.keys()) & set(pred_paths.keys()) & set(gt_paths.keys())
    
    if len(common_ids) == 0:
        raise ValueError("No matching object_ids found between checkpoints, pred PLYs, and gt PLYs!")
    
    checkpoint_only = set(checkpoint_paths.keys()) - common_ids
    pred_only = set(pred_paths.keys()) - common_ids
    gt_only = set(gt_paths.keys()) - common_ids
    
    if checkpoint_only:
        print(f"\nWARNING: {len(checkpoint_only)} checkpoints without matching PLYs: {sorted(checkpoint_only)}")
    if pred_only:
        print(f"\nWARNING: {len(pred_only)} pred PLYs without matching checkpoint/gt: {sorted(pred_only)}")
    if gt_only:
        print(f"\nWARNING: {len(gt_only)} gt PLYs without matching checkpoint/pred: {sorted(gt_only)}")
    
    print(f"\nProcessing {len(common_ids)} matched triplets")
    
    # Sort for consistent ordering
    object_ids = sorted(common_ids)
    
    # Process each matched triplet
    all_results = []
    for object_id in tqdm(object_ids, desc="Processing checkpoints"):
        print(f"Processing {object_id}")
        # if object_id != "471_66544_130959": 
        #     continue 
        # if object_id != "134_15451_31119":
        #     continue
        # else: 
        try:
            results = evaluate_checkpoint(
                checkpoint_path=checkpoint_paths[object_id],
                object_id=object_id,
                pred_ply=pred_paths[object_id],
                gt_ply=gt_paths[object_id],
                gaussian_predictor=gaussian_predictor,
                cfg=cfg,
                device=device,
                pred_method=pred_method,
                num_points=num_points,
                max_iterations=max_iterations,
                estimate_scale=estimate_scale,
                output_dir=output_dir,
                wandb_run=wandb_run,
            )
            all_results.append(results)
            
            # Save results incrementally
            file_exists = os.path.exists(results_csv)
            with open(results_csv, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(results)
            
        except Exception as e:
            print(f"\nERROR processing {object_id}: {e}")
            raise  # No silent failures - re-raise the exception
    
    # Compute aggregate statistics
    if all_results:
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)

        chamfers = [r['chamfer'] for r in all_results]
        baseline_psnrs = [r['baseline_psnr'] for r in all_results]
        pred_psnrs = [r[f'{pred_method}_psnr'] for r in all_results]
        pred_icp_psnrs = [r[f'{pred_method}_icp_psnr'] for r in all_results]
        baseline_psnrs_masked = [r['baseline_psnr_masked'] for r in all_results]
        pred_psnrs_masked = [r[f'{pred_method}_psnr_masked'] for r in all_results]
        pred_icp_psnrs_masked = [r[f'{pred_method}_icp_psnr_masked'] for r in all_results]
        baseline_mask_ious = [r['baseline_mask_iou'] for r in all_results]
        pred_mask_ious = [r[f'{pred_method}_mask_iou'] for r in all_results]
        pred_icp_mask_ious = [r[f'{pred_method}_icp_mask_iou'] for r in all_results]

        print(f"Chamfer Distance:          {np.mean(chamfers):.6f} ± {np.std(chamfers):.6f}")
        print(f"Baseline PSNR:             {np.mean(baseline_psnrs):.4f} ± {np.std(baseline_psnrs):.4f}")
        print(f"{pred_method.upper()} PSNR:             {np.mean(pred_psnrs):.4f} ± {np.std(pred_psnrs):.4f}")
        print(f"{pred_method.upper()} ICP PSNR:         {np.mean(pred_icp_psnrs):.4f} ± {np.std(pred_icp_psnrs):.4f}")
        print(f"Baseline PSNR (masked):    {np.mean(baseline_psnrs_masked):.4f} ± {np.std(baseline_psnrs_masked):.4f}")
        print(f"{pred_method.upper()} PSNR (masked):    {np.mean(pred_psnrs_masked):.4f} ± {np.std(pred_psnrs_masked):.4f}")
        print(f"{pred_method.upper()} ICP PSNR (masked):{np.mean(pred_icp_psnrs_masked):.4f} ± {np.std(pred_icp_psnrs_masked):.4f}")
        print(f"Baseline Mask IOU:         {np.mean(baseline_mask_ious):.4f} ± {np.std(baseline_mask_ious):.4f}")
        print(f"{pred_method.upper()} Mask IOU:         {np.mean(pred_mask_ious):.4f} ± {np.std(pred_mask_ious):.4f}")
        print(f"{pred_method.upper()} ICP Mask IOU:     {np.mean(pred_icp_mask_ious):.4f} ± {np.std(pred_icp_mask_ious):.4f}")

        # Log aggregate stats to wandb
        if wandb_run is not None:
            wandb_run.log({
                'aggregate/chamfer_mean': np.mean(chamfers),
                'aggregate/chamfer_std': np.std(chamfers),
                'aggregate/baseline_psnr_mean': np.mean(baseline_psnrs),
                'aggregate/baseline_psnr_std': np.std(baseline_psnrs),
                f'aggregate/{pred_method}_psnr_mean': np.mean(pred_psnrs),
                f'aggregate/{pred_method}_psnr_std': np.std(pred_psnrs),
                f'aggregate/{pred_method}_icp_psnr_mean': np.mean(pred_icp_psnrs),
                f'aggregate/{pred_method}_icp_psnr_std': np.std(pred_icp_psnrs),
                'aggregate/baseline_psnr_masked_mean': np.mean(baseline_psnrs_masked),
                'aggregate/baseline_psnr_masked_std': np.std(baseline_psnrs_masked),
                f'aggregate/{pred_method}_psnr_masked_mean': np.mean(pred_psnrs_masked),
                f'aggregate/{pred_method}_psnr_masked_std': np.std(pred_psnrs_masked),
                f'aggregate/{pred_method}_icp_psnr_masked_mean': np.mean(pred_icp_psnrs_masked),
                f'aggregate/{pred_method}_icp_psnr_masked_std': np.std(pred_icp_psnrs_masked),
                'aggregate/baseline_mask_iou_mean': np.mean(baseline_mask_ious),
                'aggregate/baseline_mask_iou_std': np.std(baseline_mask_ious),
                f'aggregate/{pred_method}_mask_iou_mean': np.mean(pred_mask_ious),
                f'aggregate/{pred_method}_mask_iou_std': np.std(pred_mask_ious),
                f'aggregate/{pred_method}_icp_mask_iou_mean': np.mean(pred_icp_mask_ious),
                f'aggregate/{pred_method}_icp_mask_iou_std': np.std(pred_icp_mask_ious),
            })
    
    # Finish wandb run
    if wandb_run is not None:
        wandb_run.finish()
        print("\nWandb run finished.")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"Processed {len(all_results)} checkpoints")
    print(f"Results saved to: {results_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch rendering evaluation with ICP alignment"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        required=True,
        help="Directory containing checkpoint subdirectories with format {object_id}-{MM-DD-YY-HH-MM-SS}/topk_best_everything_latest.pth"
    )
    parser.add_argument(
        "--pred_list", 
        type=str, 
        required=True,
        help="Text file containing paths to predicted pointclouds (one per line)"
    )
    parser.add_argument(
        "--gt_list", 
        type=str, 
        required=True,
        help="Text file containing paths to ground-truth pointclouds (one per line)"
    )
    parser.add_argument(
        "--pred_method",
        type=str,
        choices=["ours", "baseline_ood"],
        default="ours",
        help="Which pred method to evaluate: 'ours' (optimized) or 'baseline_ood' (baseline with OOD rotation). Default: ours"
    )
    parser.add_argument(
        "--results_csv", 
        type=str, 
        required=True,
        help="Path to save results CSV file"
    )
    parser.add_argument(
        "--num_points", 
        type=int, 
        default=100000,
        help="Number of points to sample from each pointcloud (default: 100000)"
    )
    parser.add_argument(
        "--max_iterations", 
        type=int, 
        default=100,
        help="Maximum ICP iterations (default: 100)"
    )
    parser.add_argument(
        "--estimate_scale", 
        action="store_true", 
        default=True,
        help="Estimate scale in Sim(3) ICP (default: True)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output videos (optional)"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default=None,
        help="Wandb project name (optional, enables wandb logging)"
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None,
        help="Wandb run name (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str,
        default="hydrants",
        help="Dataset name (default: hydrants)"
    )
    parser.add_argument(
        "--pretrained_ckpt", 
        type=str, 
        default=None,
        help="Pretrained checkpoint path (default: None)"
    )
    
    args = parser.parse_args()
    
    main(
        checkpoint_dir=args.checkpoint_dir,
        pred_list=args.pred_list,
        gt_list=args.gt_list,
        results_csv=args.results_csv,
        pred_method=args.pred_method,
        num_points=args.num_points,
        max_iterations=args.max_iterations,
        estimate_scale=args.estimate_scale,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dataset_name=args.dataset_name,
        pretrained_ckpt=args.pretrained_ckpt,
    )



