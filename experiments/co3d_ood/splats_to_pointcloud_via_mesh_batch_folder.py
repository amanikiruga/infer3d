#!/usr/bin/env python3
"""
Batch version: Process all latest checkpoints for each object in a directory.

Finds all checkpoints in format {object_id}-{MM-DD-YY-HH-MM-SS}/topk_best_everything_latest.pth
Groups by object_id and takes the latest timestamp for each object.
Processes BOTH pseudo_GT (baseline_indist) and ours (optimized).
Saves as {object_id}.ply in respective output folders.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as scipy_R
from plyfile import PlyData, PlyElement
from datetime import datetime
import glob

# Add paths
from infer3d import config as _cfg
GS2MESH_ROOT = _cfg.GS2MESH_ROOT
sys.path.append(GS2MESH_ROOT)

from omegaconf import OmegaConf
from infer3d.model import GaussianSplatPredictor
from infer3d.utils.graphics import fov2focal, focal2fov
from infer3d.utils.optim import render_with_custom_camera, render_with_custom_camera_align
import hydra
from hydra import initialize_config_dir, compose
import wandb
import imageio
from PIL import Image
from tqdm import tqdm
from infer3d.renderer import render_predicted
from infer3d.data.co3d import CO3DDataset



def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)"""
    rot = scipy_R.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Return [w, x, y, z]


@torch.no_grad()
def regenerate_baseline_ood_splats(checkpoint, baseline_indist_splats, gaussian_predictor, cfg, device):
    """Regenerate baseline splats using in-distribution OOD image (pseudo_GT)"""
    background = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], 
                              dtype=torch.float32, device=device)
    ood_data = checkpoint['ood_data']
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    ood_image_index = checkpoint['ood_image_index']
    ood_random_rotation = checkpoint["ood_random_rotation"].to(device)
    ood_centroid = checkpoint.get("ood_centroid")
    if ood_centroid is None:  # StyleGAN ckpts omit it; centroid is [0,0,zgt] by construction
        ood_centroid = torch.tensor([0., 0., float(checkpoint["zgt"])], device=device)
    assert ood_centroid is not None, "ood_centroid not found in checkpoint"
    assert ood_random_rotation is not None, "ood_random_rotation is not set"
    ood_zgt = checkpoint["zgt"]
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    ood_render_image = render_with_custom_camera(baseline_indist_splats, background, cfg, 
                                                   ood_focals_pixels, 
                                                   ood_random_rotation, ood_zgt, device=device, zgt_ood=ood_zgt, translation=None)
    print(f"ood render image shape: {ood_render_image.shape}") # 1, 3, 128, 128
    ood_render_image = ood_render_image.squeeze(0)
    # Move to device
    
    # Use the in-distribution OOD image
    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    
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
    if ood_random_rotation is not None:
        inverse_rotation = ood_random_rotation.T  # Transpose for inverse rotation
        pred_splats = render_with_custom_camera(
            pred_splats, background, cfg, ood_focals_pixels,
            inverse_rotation, ood_zgt, device=device, return_splats=True, translation=None, zgt_ood=ood_zgt
        )
    return pred_splats, ood_data

@torch.no_grad()
def regenerate_baseline_indist_splats(checkpoint, gaussian_predictor, cfg, device):
    """Regenerate baseline splats using in-distribution OOD image (pseudo_GT)"""
    ood_data = checkpoint['ood_data']
    ood_image_index = checkpoint['ood_image_index']
    
    # Move to device
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    
    # Use the in-distribution OOD image
    ood_image = ood_data["gt_images"][0, ood_image_index].to(device)
    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    
    # Prepare input
    input_images = torch.cat([ood_image.unsqueeze(0).unsqueeze(1), 
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
    
    return pred_splats, ood_data


# `regenerate_ours_splats` lives in regenerate_ours.py so both eval
# entrypoints share one implementation (they had silently drifted apart).
# Variant is selected by $INFER3D_OURS_REGEN ("paper" default, or "aligned").
from experiments.co3d_ood.regenerate_ours import regenerate_ours_splats  # noqa: E402



def construct_list_of_attributes(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape):
    """Construct attribute list for PLY file (matching Gaussian Splatting format)"""
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # DC features
    for i in range(features_dc_shape[1] * features_dc_shape[2]):
        l.append(f'f_dc_{i}')
    # Rest features
    for i in range(features_rest_shape[1] * features_rest_shape[2]):
        l.append(f'f_rest_{i}')
    l.append('opacity')
    # Scaling
    for i in range(scaling_shape[1]):
        l.append(f'scale_{i}')
    # Rotation
    for i in range(rotation_shape[1]):
        l.append(f'rot_{i}')
    return l


def export_splats_to_ply(splats, output_path, activate=False, target_sh_degree=3):
    """
    Export splats to PLY format compatible with Gaussian Splatting.
    
    Args:
        splats: Dictionary with keys ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']
        output_path: Where to save the PLY file
        activate: If True, apply activation functions (exp, sigmoid, normalize). 
                  If False, save raw values (GS format expects raw values).
        target_sh_degree: Target SH degree (default 3 for GS2Mesh)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract data
    xyz = splats['xyz'].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    features_dc = splats['features_dc'].detach().cpu()
    features_rest = splats['features_rest'].detach().cpu()
    opacity = splats['opacity'].detach().cpu().numpy()
    scaling = splats['scaling'].detach().cpu().numpy()
    rotation = splats['rotation'].detach().cpu().numpy()
    
    # Apply inverse activations if the splats are activated
    # GS expects: raw opacity (before sigmoid), raw scaling (before exp), normalized rotation
    if activate:
        # If data is activated, we need to "unactivate" it for GS format
        # opacity: sigmoid^-1
        opacity = -np.log((1.0 / (opacity + 1e-8)) - 1.0)
        # scaling: log
        scaling = np.log(scaling + 1e-8)
        # rotation is already normalized
    
    # Transpose features_dc and features_rest to match GS format
    # From (N, C, SH) to (N, SH, C) then flatten
    features_dc = features_dc.transpose(1, 2).flatten(start_dim=1).contiguous()
    features_rest = features_rest.transpose(1, 2).flatten(start_dim=1).contiguous()
    
    # Pad or trim features_rest to match target SH degree
    # For SH degree d: num_rest_features = 3 * (d+1)^2 - 3
    target_rest_features = 3 * (target_sh_degree + 1) ** 2 - 3
    current_rest_features = features_rest.shape[1]
    
    if current_rest_features < target_rest_features:
        # Pad with zeros
        padding = torch.zeros(features_rest.shape[0], target_rest_features - current_rest_features)
        features_rest = torch.cat([features_rest, padding], dim=1)
    elif current_rest_features > target_rest_features:
        # Trim
        features_rest = features_rest[:, :target_rest_features]
    
    features_dc = features_dc.numpy()
    features_rest_np = features_rest.numpy()
    
    # Construct attribute list using actual shapes (after padding/trimming)
    # Need to create fake shapes that match the actual data
    dc_shape = (features_dc.shape[0], 1, features_dc.shape[1])  # (N, 1, 3)
    rest_shape = (features_rest_np.shape[0], 1, features_rest_np.shape[1])  # (N, 1, target_rest_features)
    
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(
        dc_shape, rest_shape, 
        splats['scaling'].shape, splats['rotation'].shape)]
    
    # Create PLY data
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, features_dc, features_rest_np, opacity, scaling, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)
    
    print(f"Saved {xyz.shape[0]} Gaussians to {output_path}")


def render_splats_and_log_to_wandb(splats, ood_data, cfg, wandb_run, object_id, method_name):
    """
    Render Gaussian splats and log as videos to wandb.
    
    Args:
        splats: Dictionary with Gaussian splat parameters
        ood_data: Dictionary with camera data
        cfg: Config object
        wandb_run: wandb run object
        object_id: Object identifier for logging
        method_name: Name for logging (e.g., "pseudo_gt", "ours")
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move data to device
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    splats = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in splats.items()}
    
    # Set up rendering background
    background = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    
    rendered_images = []
    rendered_invdepths = []
    num_cameras = ood_data["world_view_transforms"].shape[1]
    
    # Render all views
    for r_idx in range(num_cameras):
        world_view_transforms = ood_data["world_view_transforms"][0, r_idx].unsqueeze(0)
        full_proj_transforms = ood_data["full_proj_transforms"][0, r_idx].unsqueeze(0)
        camera_centers = ood_data["camera_centers"][0, r_idx].unsqueeze(0)
        cur_focals_pixels = ood_data["focals_pixels"][0, r_idx] if "focals_pixels" in ood_data else None
        
        with torch.no_grad():
            render_output = render_predicted(
                splats, world_view_transforms, full_proj_transforms,
                camera_centers, background, cfg, focals_pixels=cur_focals_pixels
            )
            image = render_output["render"]
            invdepth = render_output["invdepths"]
        
        # Convert RGB to numpy
        rendered_image = torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        rendered_images.append(rendered_image)
        
        # Store inverse depth
        invdepth_squeezed = invdepth.squeeze(0) if invdepth.dim() == 3 else invdepth
        rendered_invdepths.append(invdepth_squeezed.detach().cpu())
    
    # Normalize inverse depths globally
    invdepths_tensor = torch.stack(rendered_invdepths)
    invdepth_min = invdepths_tensor.min()
    invdepth_max = invdepths_tensor.max()
    
    if invdepth_max > invdepth_min:
        invdepths_normalized = (invdepths_tensor - invdepth_min) / (invdepth_max - invdepth_min)
    else:
        invdepths_normalized = torch.zeros_like(invdepths_tensor)
    
    # Apply colormap for depth visualization
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    cmap = plt.get_cmap('turbo')
    invdepth_images = []
    for i in range(len(invdepths_normalized)):
        invdepth_np = invdepths_normalized[i].numpy()
        invdepth_colored = (cmap(invdepth_np)[:, :, :3] * 255).astype(np.uint8)
        invdepth_images.append(invdepth_colored)
    
    # Create videos
    video_path = f"/tmp/rendered_splats_{object_id}_{method_name}.mp4"
    imageio.mimsave(video_path, rendered_images, fps=4, codec='libx264')
    
    invdepth_video_path = f"/tmp/rendered_invdepths_{object_id}_{method_name}.mp4"
    imageio.mimsave(invdepth_video_path, invdepth_images, fps=4, codec='libx264')
    
    # Log to wandb
    if wandb_run is not None:
        wandb_run.log({
            f"{object_id}/rendered_splats_{method_name}": wandb.Video(video_path, fps=4, format="mp4"),
            f"{object_id}/rendered_invdepths_{method_name}": wandb.Video(invdepth_video_path, fps=4, format="mp4"),
            f"{object_id}/num_gaussians_{method_name}": splats['xyz'].shape[0],
        })


def export_colmap_format(ood_data, output_dir, image_dir=None, use_synthetic_poses=True, camera_distance=2.5, example_id=None):
    """
    Export camera poses and parameters to COLMAP format.
    
    Args:
        ood_data: Dictionary with camera data
        output_dir: Where to save COLMAP files (will create sparse/0/ subdirectory)
        image_dir: Optional path to save placeholder images
        use_synthetic_poses: If True, create synthetic camera poses in a circular arc
                           If False and example_id provided, load original CO3D camera poses
        camera_distance: Distance of cameras from origin (used for synthetic poses)
        example_id: CO3D example ID (needed to load original camera poses)
    """
    sparse_dir = os.path.join(output_dir, 'sparse', '0')
    os.makedirs(sparse_dir, exist_ok=True)
    
    num_cameras = ood_data['view_to_world_transforms'].shape[1]
    
    # Get camera parameters (assume all cameras have same intrinsics)
    focal_pixels = ood_data['focals_pixels'][0, 0].cpu().numpy()  # [fx, fy]
    width = height = 128  # CO3D images are 128x128
    cx, cy = width / 2.0, height / 2.0
    
    # Write cameras.txt (PINHOLE model)
    cameras_path = os.path.join(sparse_dir, 'cameras.txt')
    with open(cameras_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        # Use single camera model for all views
        f.write(f"1 PINHOLE {width} {height} {focal_pixels[0]} {focal_pixels[1]} {cx} {cy}\n")
    
    # Write images.txt
    images_path = os.path.join(sparse_dir, 'images.txt')
    with open(images_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_IDX)\n")
        f.write(f"# Number of images: {num_cameras}, mean observations per image: 0\n")
        
        for i in range(num_cameras):
            # Use camera poses from ood_data (checkpoint has relative poses)
            w2c_tensor = ood_data['world_view_transforms'][0, i]
            c2w_tensor = w2c_tensor.inverse()
            
            # Camera center in world coordinates (row 3 because matrices are transposed in this codebase)
            camera_center = c2w_tensor[3, :3].cpu().numpy()
            
            # Build proper w2c with camera center
            # NOTE: Matrices are transposed in this codebase, so [:3, :3] gives R^T
            # We need to transpose it back to get the actual rotation matrix
            R = w2c_tensor[:3, :3].T.cpu().numpy()
            T = -R @ camera_center
            
            # Convert rotation matrix to quaternion (w, x, y, z)
            quat = rotation_matrix_to_quaternion(R)
            
            # Image ID, quaternion, translation, camera ID, image name
            image_id = i + 1
            camera_id = 1
            image_name = f"{i:06d}.png"
            
            f.write(f"{image_id} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {T[0]} {T[1]} {T[2]} {camera_id} {image_name}\n")
            f.write("\n")  # Empty line for 2D points

    print(f"Wrote cameras to {cameras_path}")
    print(f"Wrote images to {images_path}")
    
    # Optionally save images
    if image_dir is not None:
        os.makedirs(image_dir, exist_ok=True)
        gt_images = ood_data['gt_images'][0].cpu().numpy()  # (N, 3, H, W)
        for i in range(num_cameras):
            img = gt_images[i].transpose(1, 2, 0)  # (H, W, 3)
            img = (img * 255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(img).save(os.path.join(image_dir, f"{i:06d}.png"))
    
    # Write empty points3D.txt
    points_path = os.path.join(sparse_dir, 'points3D.txt')
    with open(points_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0, mean track length: 0\n")


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
            # continue
        
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


def process_single_checkpoint(checkpoint_path, object_id, gaussian_predictor, cfg, device, 
                              pseudo_gt_output_folder, ours_output_folder, baseline_ood_output_folder, wandb_run=None):
    """Process a single checkpoint and save both pseudo_GT and ours PLYs"""
    print("\n" + "=" * 80)
    print(f"PROCESSING OBJECT: {object_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    example_id = checkpoint['example_id']

    # Reconstruct ood_data if not in checkpoint (new format)
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

    # ========== Process PSEUDO_GT (baseline_indist) ==========
    print("\n" + "-" * 80)
    print("GENERATING PSEUDO_GT (baseline_indist)")
    print("-" * 80)
    
    pseudo_gt_splats, ood_data = regenerate_baseline_indist_splats(checkpoint, gaussian_predictor, cfg, device)
    baseline_ood_splats, _ = regenerate_baseline_ood_splats(checkpoint, pseudo_gt_splats, gaussian_predictor, cfg, device)
    print(f"Generated {pseudo_gt_splats['xyz'].shape[0]} Gaussians for pseudo_GT")
    
    # Set up directories for pseudo_GT
    method_name = "baseline_indist"
    
    
    if pseudo_gt_output_folder is not None:
        splatting_dir = os.path.join(pseudo_gt_output_folder, object_id,"splatting")
        colmap_dir = os.path.join(pseudo_gt_output_folder, object_id, "colmap")
        os.makedirs(splatting_dir, exist_ok=True)
        os.makedirs(colmap_dir, exist_ok=True)
        image_dir = os.path.join(colmap_dir, 'images')
    else:
        raise ValueError("pseudo_gt_output_folder is not set")
        # colmap_dir = os.path.join(GS2MESH_ROOT, 'data', 'custom', f'{example_id}_{method_name}')
        # splatting_dir = os.path.join(GS2MESH_ROOT, 'splatting_output', 'custom', f'{example_id}_{method_name}')
    # Export pseudo_GT splats to PLY
    pseudo_gt_ply_path = os.path.join(splatting_dir, f'{object_id}.ply')
    export_splats_to_ply(pseudo_gt_splats, pseudo_gt_ply_path, activate=True)
    
    # Export COLMAP format for pseudo_GT
    export_colmap_format(ood_data, colmap_dir, image_dir, use_synthetic_poses=False, camera_distance=2.5, example_id=None)
    
    # Render and log pseudo_GT
    if wandb_run is not None:
        print("\nRendering pseudo_GT splats...")
        render_splats_and_log_to_wandb(pseudo_gt_splats, ood_data, cfg, wandb_run, object_id, "pseudo_gt")
    
    # ========== Process BASELINE OOD ==========

    if baseline_ood_output_folder is not None:
        splatting_dir = os.path.join(baseline_ood_output_folder, object_id,"splatting")
        colmap_dir = os.path.join(baseline_ood_output_folder, object_id, "colmap")
        os.makedirs(splatting_dir, exist_ok=True)
        os.makedirs(colmap_dir, exist_ok=True)
        image_dir = os.path.join(colmap_dir, 'images')
    else:
        raise ValueError("baseline_ood_output_folder is not set")
        # colmap_dir = os.path.join(GS2MESH_ROOT, 'data', 'custom', f'{example_id}_{method_name}')
        # splatting_dir = os.path.join(GS2MESH_ROOT, 'splatting_output', 'custom', f'{example_id}_{method_name}')
    # Export baseline_ood splats to PLY
    baseline_ood_ply_path = os.path.join(splatting_dir, f'{object_id}.ply')
    export_splats_to_ply(baseline_ood_splats, baseline_ood_ply_path, activate=True)
    
    # Export COLMAP format for baseline_ood
    export_colmap_format(ood_data, colmap_dir, image_dir, use_synthetic_poses=False, camera_distance=2.5, example_id=None)
    
    # Render and log baseline_ood
    if wandb_run is not None:
        print("\nRendering baseline_ood splats...")
        render_splats_and_log_to_wandb(baseline_ood_splats, ood_data, cfg, wandb_run, object_id, "baseline_ood")
    
    # ========== Process OURS (optimized) ==========
    print("\n" + "-" * 80)
    print("GENERATING OURS (optimized with rotation)")
    print("-" * 80)

    # Trivial-TTT support: if the checkpoint shipped per-scene fine-tuned weights,
    # swap them in for "ours" only, then restore the pretrained weights afterwards
    # so subsequent scenes / pseudo_GT / baseline_ood remain unaffected.
    pretrained_state_dict_backup = None
    if "finetuned_state_dict" in checkpoint:
        print("Loading per-scene finetuned_state_dict for OURS regeneration")
        pretrained_state_dict_backup = {k: v.detach().clone() for k, v in gaussian_predictor.state_dict().items()}
        gaussian_predictor.load_state_dict({k: v.to(device) for k, v in checkpoint["finetuned_state_dict"].items()})

    ours_splats, ood_data = regenerate_ours_splats(checkpoint, gaussian_predictor, cfg, device)

    if pretrained_state_dict_backup is not None:
        gaussian_predictor.load_state_dict(pretrained_state_dict_backup)
        print("Restored pretrained weights")
    print(f"Generated {ours_splats['xyz'].shape[0]} Gaussians for ours")
    
    # Set up directories for ours
    method_name = "ours"

    
    if ours_output_folder is not None:
        colmap_dir = os.path.join(ours_output_folder, object_id, "colmap")
        splatting_dir = os.path.join(ours_output_folder, object_id, "splatting")
        os.makedirs(colmap_dir, exist_ok=True)
        os.makedirs(splatting_dir, exist_ok=True)
        image_dir = os.path.join(colmap_dir, 'images')
    else:
        raise ValueError("ours_output_folder is not set")
        # colmap_dir = os.path.join(GS2MESH_ROOT, 'data', 'custom', f'{example_id}_{method_name}')
        # splatting_dir = os.path.join(GS2MESH_ROOT, 'splatting_output', 'custom', f'{example_id}_{method_name}')
    
    # Export ours splats to PLY
    ours_ply_path = os.path.join(splatting_dir, f'{object_id}.ply')
    export_splats_to_ply(ours_splats, ours_ply_path, activate=True)
    
    # Export COLMAP format for ours
    export_colmap_format(ood_data, colmap_dir, image_dir, use_synthetic_poses=False, camera_distance=2.5, example_id=None)
    
    # Render and log ours
    if wandb_run is not None:
        print("\nRendering ours splats...")
        render_splats_and_log_to_wandb(ours_splats, ood_data, cfg, wandb_run, object_id, "ours")
    
    return pseudo_gt_ply_path, baseline_ood_ply_path, ours_ply_path

@torch.no_grad()
def main(checkpoint_dir, baseline_ood_output_folder, pseudo_gt_output_folder, ours_output_folder, object_id=None, max_objects=-1, dataset_name="hydrants", pretrained_ckpt=None):
    """
    Main pipeline: Process all latest checkpoints in a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoint subdirectories
        pseudo_gt_output_folder: Folder to save pseudo_GT PLYs
        ours_output_folder: Folder to save ours PLYs
    """
    # Initialize wandb
    wandb_run = wandb.init(
        project="gs2mesh-batch-export",
        name=f"batch_export_{Path(checkpoint_dir).name}",
        config={
            "checkpoint_dir": checkpoint_dir,
            "baseline_ood_output_folder": baseline_ood_output_folder,
            "pseudo_gt_output_folder": pseudo_gt_output_folder,
            "ours_output_folder": ours_output_folder,
            "pretrained_ckpt": pretrained_ckpt
        }
    )
    
    # Load config using Hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=_cfg.CONFIGS_DIR)
    cfg = compose(config_name="abs_config", overrides=[
        "general.split=0", 
        "general.total_splits=1", 
        f"+dataset={dataset_name}", 
        "abs=diffae_abs", 
        f"opt.pretrained_ckpt={pretrained_ckpt}", 
        "general.data_example_ids_path=not_needed.json"
    ])
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Gaussian predictor
    print("\n" + "=" * 80)
    print("LOADING GAUSSIAN PREDICTOR")
    print("=" * 80)
    gaussian_predictor = GaussianSplatPredictor(cfg)
    
    if cfg.opt.pretrained_ckpt is not None:
        model_path = cfg.opt.pretrained_ckpt
        model_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        gaussian_predictor.load_state_dict(model_checkpoint["model_state_dict"])
        print(f'Loaded model from {model_path}')
    
    gaussian_predictor = gaussian_predictor.to(device)
    gaussian_predictor.eval()
    
    # Find latest checkpoints
    print("\n" + "=" * 80)
    print("FINDING LATEST CHECKPOINTS")
    print("=" * 80)
    latest_checkpoints = find_latest_checkpoints(checkpoint_dir)
    if object_id is not None:
        latest_checkpoints = [checkpoint for checkpoint in latest_checkpoints if checkpoint[0] == object_id]
        assert len(latest_checkpoints) == 1, "Expected 1 checkpoint for object_id, got {len(latest_checkpoints)}"
    print(f"\nFound {len(latest_checkpoints)} objects to process")
    if max_objects > 0:
        latest_checkpoints = latest_checkpoints[:max_objects]
    # Process each checkpoint
    results = []
    for object_id, checkpoint_path in latest_checkpoints:
        pseudo_gt_ply_path, baseline_ood_ply_path, ours_ply_path = process_single_checkpoint(
            checkpoint_path, 
            object_id, 
            gaussian_predictor, 
            cfg, 
            device, 
            pseudo_gt_output_folder,
            ours_output_folder,
            baseline_ood_output_folder,
            wandb_run
        )
        results.append((object_id, pseudo_gt_ply_path, baseline_ood_ply_path, ours_ply_path))
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    for object_id, pseudo_gt_path, baseline_ood_path, ours_path in results:
        print(f"  {object_id}:")
        print(f"    pseudo_GT: {pseudo_gt_path}")
        print(f"    baseline_ood: {baseline_ood_path}")
        print(f"    ours:      {ours_path}")
    
    # Finish wandb run
    wandb_run.finish()
    print("\nWandb run finished.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch export Gaussian splats for GS2Mesh (both pseudo_GT and ours)")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint subdirectories")
    parser.add_argument("--object_id", type=str, required=False, help="Object ID (optional)")
    parser.add_argument("--baseline_ood_output_folder", type=str, required=True, help="Folder to save baseline_ood (baseline_indist) PLYs")
    parser.add_argument("--pseudo_gt_output_folder", type=str, required=True, help="Folder to save pseudo_GT (baseline_indist) PLYs")
    parser.add_argument("--ours_output_folder", type=str, required=True, help="Folder to save ours (optimized) PLYs")
    parser.add_argument("--max_objects", type=int, required=False, default=-1,help="Maximum number of objects to process (optional)")
    parser.add_argument("--dataset_name", type=str, required=False, default="hydrants", help="Dataset name (default: hydrants)")
    parser.add_argument("--pretrained_ckpt", type=str, required=False, default=None, help="Pretrained checkpoint path (default: None)")
    args = parser.parse_args()
    if args.object_id is not None:
        print(f"Processing object_id: {args.object_id}")
    # else:
    #     raise ValueError("object_id is required")
    
    # Run the batch pipeline
    main(
        checkpoint_dir=args.checkpoint_dir,
        baseline_ood_output_folder=args.baseline_ood_output_folder,
        pseudo_gt_output_folder=args.pseudo_gt_output_folder,
        ours_output_folder=args.ours_output_folder,
        object_id=args.object_id,
        max_objects=args.max_objects,
        dataset_name=args.dataset_name,
        pretrained_ckpt=args.pretrained_ckpt
    )
    