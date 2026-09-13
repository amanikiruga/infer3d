
import numpy as np
import torch
import math 
from scipy.spatial.transform import Rotation as scipy_R
import torchvision.transforms as transforms
import imageio
from scipy.spatial.transform import Rotation as scipy_R
from infer3d.utils.graphics import getProjectionMatrix, focal2fov
from infer3d.utils.general import matrix_to_quaternion, quaternion_raw_multiply
from infer3d.renderer import render_predicted
import torch.nn.functional as F
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper functions 
from infer3d.utils.loss import ssim as ssim_fn
import lpips as lpips_lib

class Metricator():
    def __init__(self, device):
        self.lpips_net = lpips_lib.LPIPS(net='vgg').to(device)
    def compute_metrics(self, image, target):
        lpips = self.lpips_net( image.unsqueeze(0) * 2 - 1, target.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - target) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, target).item()
        return psnr, ssim, lpips


def create_loop_and_eval(gaussian_splats, test_data_instance, test_data_name, cfg, save_loop_path = None, cond_idx = None, save_gt = False, ood_cond_image = None, offset_idx = 0, plot_number = None):
    # setup evaluator 
    metricator = Metricator(device)
    psnr_all_renders_cond = []
    ssim_all_renders_cond = []
    lpips_all_renders_cond = []


    psnr_all_renders_novel = []
    ssim_all_renders_novel = []
    lpips_all_renders_novel = []
    
    background = torch.tensor([1, 1, 1] , dtype=torch.float32, device=device)
    loop_renders = []
    gt_images = []
    test_data_instance = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in test_data_instance.items()}
    cur_plot_number = 0
    is_eval = True
    for r_idx in range(test_data_instance["world_view_transforms"].shape[1]-offset_idx ):
        world_view_transforms = test_data_instance["world_view_transforms"][0, r_idx+offset_idx].unsqueeze(0)
        full_proj_transforms = test_data_instance["full_proj_transforms"][0, r_idx+offset_idx].unsqueeze(0)
        camera_centers = test_data_instance["camera_centers"][0, r_idx+offset_idx].unsqueeze(0)

        image = render_predicted(gaussian_splats,
                                    world_view_transforms,
                                    full_proj_transforms,  
                                    camera_centers,
                                    background,
                                    cfg,
                                    focals_pixels=None)["render"]
        
        gt_idx = r_idx+offset_idx
        gt_image = test_data_instance["gt_images"][0, gt_idx].to(device)
        
        
        if is_eval: 
            psnr, ssim, lpips = metricator.compute_metrics(image, gt_image)
            if cond_idx is not None and r_idx == cond_idx+offset_idx: 
                psnr_all_renders_cond.append(psnr)
                ssim_all_renders_cond.append(ssim)
                lpips_all_renders_cond.append(lpips)
            else: 
                psnr_all_renders_novel.append(psnr)
                ssim_all_renders_novel.append(ssim)
                lpips_all_renders_novel.append(lpips)
        
        if plot_number is None or cur_plot_number < plot_number: 
            cur_plot_number+=1
            gt_images.append(torch.clamp(gt_image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    
    if ood_cond_image is not None:
        # render splats at identity 
        pred_identity_image = render_predicted(gaussian_splats,
                                    test_data_instance["world_view_transforms"][0, 0].unsqueeze(0),
                                    test_data_instance["full_proj_transforms"][0, 0].unsqueeze(0),  
                                    test_data_instance["camera_centers"][0, 0].unsqueeze(0),
                                    background,
                                    cfg,
                                    focals_pixels=None)["render"]
        ood_cond_image = ood_cond_image.to(device)
        psnr, ssim, lpips = metricator.compute_metrics(pred_identity_image, ood_cond_image)
        psnr_all_renders_cond.append(psnr)
        ssim_all_renders_cond.append(ssim)
        lpips_all_renders_cond.append(lpips)
    
    scores = None
    if is_eval: 
        scores = {
            "PSNR_cond": sum(psnr_all_renders_cond) / max(1, len(psnr_all_renders_cond)),
            "SSIM_cond": sum(ssim_all_renders_cond) / max(1, len(ssim_all_renders_cond)),
            "LPIPS_cond": sum(lpips_all_renders_cond) / max(1,len(lpips_all_renders_cond)),
            "PSNR_novel": sum(psnr_all_renders_novel) / max(1,len(psnr_all_renders_novel)),
            "SSIM_novel": sum(ssim_all_renders_novel) / max(1,len(ssim_all_renders_novel)),
            "LPIPS_novel": sum(lpips_all_renders_novel) / max(1,len(lpips_all_renders_novel))
            }
    if save_loop_path: 
        imageio.mimsave(save_loop_path, loop_renders, fps=25, codec='libx264')
        if save_gt: 
            imageio.mimsave(save_loop_path.replace(".mp4", "_gt.mp4"), gt_images, fps=25, codec='libx264')
    return loop_renders, gt_images, scores


def get_source_cw2wT( source_cameras_view_to_world):
    # Compute view to world transforms in quaternion representation.
    # Used for transforming predicted rotations
    qs = []
    for c_idx in range(source_cameras_view_to_world.shape[0]):
        qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
    return torch.stack(qs, dim=0)

def make_data_relative_to(target_data, reference_data, batch_size = 1, return_relative = True, include_absolute=False, include_gt_images = False): 
    """
    Combines the reference and target data into a single dictionary,
    and makes the poses relative to the first camera in the reference data.
    Args:
        target_data: The target data to be combined.
        reference_data: The reference data to be combined.
        batch_size: The batch size of the data.
        return_relative: Whether to return the relative data or not.    
    """

    comb_data = {
        
    }
    all_data = [reference_data, target_data]
    num_across_first_dim = reference_data["world_view_transforms"].shape[1]
    for data in all_data: 
        if "world_view_transforms" not in comb_data: 
            
            comb_data = {
                
                "world_view_transforms": data["world_view_transforms_absolute"],
                "view_to_world_transforms": data["view_to_world_transforms_absolute"],
                "full_proj_transforms": data["full_proj_transforms_absolute"],
                "camera_centers": data["camera_centers_absolute"],
            }

            if include_absolute: 
                comb_data["world_view_transforms_absolute"] = data["world_view_transforms_absolute"]
                comb_data["view_to_world_transforms_absolute"] = data["view_to_world_transforms_absolute"]
                comb_data["full_proj_transforms_absolute"] = data["full_proj_transforms_absolute"]
                comb_data["camera_centers_absolute"] = data["camera_centers_absolute"]
            if include_gt_images: 
                comb_data["gt_images"] = data["gt_images"]

        else: 
            comb_data["world_view_transforms"] = torch.cat([comb_data["world_view_transforms"], data["world_view_transforms_absolute"]], dim=1)
            comb_data["view_to_world_transforms"] = torch.cat([comb_data["view_to_world_transforms"], data["view_to_world_transforms_absolute"]], dim=1)
            comb_data["full_proj_transforms"] = torch.cat([comb_data["full_proj_transforms"], data["full_proj_transforms_absolute"]], dim=1)
            comb_data["camera_centers"] = torch.cat([comb_data["camera_centers"], data["camera_centers_absolute"]], dim=1)
            if include_absolute: 
                comb_data["world_view_transforms_absolute"] = torch.cat([comb_data["world_view_transforms_absolute"], data["world_view_transforms_absolute"]], dim=1)
                comb_data["view_to_world_transforms_absolute"] = torch.cat([comb_data["view_to_world_transforms_absolute"], data["view_to_world_transforms_absolute"]], dim=1)
                comb_data["full_proj_transforms_absolute"] = torch.cat([comb_data["full_proj_transforms_absolute"], data["full_proj_transforms_absolute"]], dim=1)
                comb_data["camera_centers_absolute"] = torch.cat([comb_data["camera_centers_absolute"], data["camera_centers_absolute"]], dim=1)
            if include_gt_images:
                comb_data["gt_images"] = torch.cat([comb_data["gt_images"], data["gt_images"]], dim=1)
    
    comb_data = make_poses_relative_to_first(comb_data)
    source_cw2wTs = []
    for batch_idx in range(batch_size): 
        cur_source_cw2wT = get_source_cw2wT(comb_data["view_to_world_transforms"][batch_idx])
        source_cw2wTs.append(cur_source_cw2wT)
        
    comb_data["source_cv2wT_quat"] = torch.stack(source_cw2wTs)  
    if return_relative:
        # return only the relative data
        comb_data["world_view_transforms"] = comb_data["world_view_transforms"][:, num_across_first_dim:]
        comb_data["view_to_world_transforms"] = comb_data["view_to_world_transforms"][:, num_across_first_dim:]
        comb_data["full_proj_transforms"] = comb_data["full_proj_transforms"][:, num_across_first_dim:]
        comb_data["camera_centers"] = comb_data["camera_centers"][:, num_across_first_dim:]
        comb_data["source_cv2wT_quat"] = comb_data["source_cv2wT_quat"][:, num_across_first_dim:]
        if not include_gt_images: 
            comb_data["gt_images"] = target_data["gt_images"]
        else: 
            comb_data["gt_images"] = comb_data["gt_images"][:, num_across_first_dim:]
            
        if include_absolute:
            comb_data["world_view_transforms_absolute"] = comb_data["world_view_transforms_absolute"][:, num_across_first_dim:]
            comb_data["view_to_world_transforms_absolute"] = comb_data["view_to_world_transforms_absolute"][:, num_across_first_dim:]
            comb_data["full_proj_transforms_absolute"] = comb_data["full_proj_transforms_absolute"][:, num_across_first_dim:]
            comb_data["camera_centers_absolute"] = comb_data["camera_centers_absolute"][:, num_across_first_dim:]

        assert comb_data["world_view_transforms"].shape[1] == target_data["world_view_transforms_absolute"].shape[1], f"world_view_transforms should have shape {target_data['world_view_transforms_absolute'].shape[1]} cf. {comb_data['world_view_transforms'].shape[1]}"
      
    
    return comb_data

def make_poses_relative_to_first( images_and_camera_poses):
    inverse_first_camera = images_and_camera_poses["world_view_transforms"][:, 0].inverse().clone()
    for c in range(images_and_camera_poses["world_view_transforms"].shape[1]):
        images_and_camera_poses["world_view_transforms"][:, c] = torch.bmm(
                                        inverse_first_camera,
                                        images_and_camera_poses["world_view_transforms"][:, c])
        images_and_camera_poses["view_to_world_transforms"][:, c] = torch.bmm(
                                            images_and_camera_poses["view_to_world_transforms"][:, c],
                                            inverse_first_camera.inverse())
        images_and_camera_poses["full_proj_transforms"][:, c] = torch.bmm(
                                            inverse_first_camera,
                                            images_and_camera_poses["full_proj_transforms"][:, c])
        images_and_camera_poses["camera_centers"][:, c] = images_and_camera_poses["world_view_transforms"][:, c].inverse()[:, 3, :3]
    
    return images_and_camera_poses






def symmetric_orthogonalization(x):
  """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

  x: should have size [batch_size, 9]

  Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
  """
  m = x.view(-1, 3, 3)
  u, s, v = torch.svd(m)
  vt = torch.transpose(v, 1, 2)
  det = torch.det(torch.matmul(u, vt))
  det = det.view(-1, 1, 1)
  vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
  r = torch.matmul(u, vt)
  return r



def sample_point_on_sphere(radius):
    """Uniformly sample a point on the surface of a sphere with a given radius using inverse transform sampling."""
    u = np.random.uniform(0, 1)
    v = np.random.uniform(0, 1)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


def sample_evenly_distributed_points_on_sphere(samples=40, radius=1.0):
    """ 
    Uses fibonacci lattice to sample points on a sphere
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        # Scale the point to the desired radius
        points.append((x * radius, y * radius, z * radius))

    return np.array(points)

def get_random_cameras_deterministic(num_rotations, zgt, device = "cuda", num_forward_rotations = 12): 
    rotations = []
    camera_positions = sample_evenly_distributed_points_on_sphere(samples=num_rotations, radius=zgt)

    
    for idx, camera_position in enumerate(camera_positions):
        # Camera position
        # camera_position = sample_point_on_sphere(zgt)
        # camera_position = np.array([0, 0, -zgt])

        # Compute the camera orientation to look at the object centroid
        forward = -camera_position 
        forward = forward / np.linalg.norm(forward) # normalize forward  
        
        # The "up" direction of the sphere is always 
        sphere_up = np.array([0, 1, 0])
        
        if np.abs(np.dot(forward, sphere_up)) > 0.9999:
            # If the forward vector is almost aligned with the up vector, we need to choose a different up vector
            sphere_up = np.array([0, 0, 1])
        
        # Project sphere_up onto the plane perpendicular to forward
        up = sphere_up - np.dot(sphere_up, forward) * forward
        
        # Normalize up vector
        up = up / np.linalg.norm(up)
        
        # Compute right vector to complete the orthonormal basis
        right = np.cross(up, forward)
        
        # Construct the rotation matrix
        R = np.column_stack((right, up, forward))
        # for i in range(num_forward_rotations):
        #     angle = 2 * np.pi * i / num_forward_rotations
        #     random_rotation = scipy_R.from_rotvec(angle * forward)
        #     R_rotated = random_rotation.apply(R)
        #     rotations.append(torch.tensor(R_rotated, dtype=torch.float32))
    
        for i in range(num_forward_rotations):
            angle = 2 * np.pi * i / num_forward_rotations

            # Define rotation explicitly around LOCAL Z-axis (camera forward)
            rotation_around_forward = scipy_R.from_rotvec(angle * np.array([0, 0, 1]))

            # Correctly rotate R around its local forward axis (Z-axis)
            R_rotated = R @ rotation_around_forward.as_matrix()

            rotations.append(torch.tensor(R_rotated, dtype=torch.float32))

    rotations = torch.stack(rotations).to(device)
    return rotations

def get_random_cameras(num_rotations, zgt, device = "cuda"): 
    rotations = []
    camera_positions = sample_evenly_distributed_points_on_sphere(samples=num_rotations, radius=zgt)

    for idx, camera_position in enumerate(camera_positions):
        # Camera position
        # camera_position = sample_point_on_sphere(zgt)
        # camera_position = np.array([0, 0, -zgt])

        # Compute the camera orientation to look at the object centroid
        forward = -camera_position 
        forward = forward / np.linalg.norm(forward) # normalize forward  
        
        # The "up" direction of the sphere is always 
        sphere_up = np.array([0, 1, 0])
        
        if np.abs(np.dot(forward, sphere_up)) > 0.9999:
            # If the forward vector is almost aligned with the up vector, we need to choose a different up vector
            sphere_up = np.array([0, 0, 1])
        
        # Project sphere_up onto the plane perpendicular to forward
        up = sphere_up - np.dot(sphere_up, forward) * forward
        
        # Normalize up vector
        up = up / np.linalg.norm(up)
        
        # Compute right vector to complete the orthonormal basis
        right = np.cross(up, forward)
        
        # Construct the rotation matrix
        R = np.column_stack((right, up, forward))
        
        # Apply a random rotation around the forward vector
        angle = np.random.uniform(0, 2 * np.pi)
        # angle = 4 * np.pi
        random_rotation = scipy_R.from_rotvec(angle * forward)

        # Apply the random rotation to the camera orientation
        R = random_rotation.apply(R)

        # turn to tensors
        R = torch.tensor(R, dtype=torch.float32)

        rotations.append(R)
    rotations = torch.stack(rotations).to(device)
    return rotations


def render_with_custom_camera_align(splats, background, cfg, focal_pixels, rotation, zgt, device = 'cuda', return_splats=False, translation=None, zgt_ood=None, return_centroid=False, override_centroid=None, return_depth=False): 
    assert zgt_ood is not None, "zgt_ood must be provided"
    
    # Compute the ACTUAL center of mass from the Gaussian splats
    # splats['xyz'] has shape [N, 3] where N is number of Gaussians
    if override_centroid is not None: 
        object_centroid = override_centroid.to(device)
    else: 
        object_centroid = splats['xyz'].mean(dim=0)  # [3] - actual mean (x, y, z) position
        
    # For OOD rendering, keep the x,y center but use zgt_ood for z
    # This preserves the object's lateral position while adjusting depth
    object_centroid_ood = object_centroid.clone()
    object_centroid_ood[2] = zgt_ood

    # Create the inverse rotation matrix
    R_inv = rotation.T
 
    # Create translation matrices
    # Step 1: Translate object center to origin for proper rotation
    T_to_origin = torch.eye(4, device=device)   
    T_to_origin[:3, 3] = -object_centroid

    # Step 2: Translate back to target position (with adjusted z)
    T_back = torch.eye(4, device=device)
    T_back[:3, 3] = object_centroid_ood

    # Create the rotation matrix (4x4)
    R_4x4 = torch.eye(4, device=device)
    R_4x4[:3, :3] = R_inv
    
    R_quat = matrix_to_quaternion(R_inv)
    
    # Optional additional translation
    T_translate = torch.eye(4, device=device)
    if translation is not None: 
        T_translate[:3, 3] = translation

    # Compute the full transformation matrix:
    # 1. T_to_origin: Move object to origin
    # 2. R_4x4: Rotate around origin (valid SO(3))
    # 3. T_back: Move to target position
    # 4. T_translate: Apply additional translation
    full_transform = T_translate @ T_back @ R_4x4 @ T_to_origin

    full_transform = full_transform


    # Apply the transformation to the splats
    transformed_splats_new = {}
    for k, v in splats.items():
        
        if k == 'xyz':  # Transform positions
            homogeneous_coords = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
            transformed_coords = (full_transform @ homogeneous_coords.T).T
            transformed_splats_new[k] = transformed_coords[:, :3].contiguous()
        
        elif k == 'rotation': 
            covar_rots = v
            R_quat_expanded = R_quat.unsqueeze(0).expand(*covar_rots.shape)
            covar_rots_new = quaternion_raw_multiply(R_quat_expanded, covar_rots)
            transformed_splats_new[k] = covar_rots_new
            
        else:  # Keep other attributes unchanged
            transformed_splats_new[k] = v.contiguous()
    
    
    if return_splats: 
        if return_centroid: 
            return transformed_splats_new, object_centroid
        return transformed_splats_new
    
    # Keep the camera at its original position
    original_world_view_transform = torch.eye(4).to('cuda')
    original_camera_center = torch.tensor([0., 0., 0.], device='cuda')
    # Compute FoV and projection matrix as before
    if cfg.data.category == "hydrants" or cfg.data.category == "teddybears" or cfg.data.category == "motorcycles" or cfg.data.category == "plants" or cfg.data.category == "vases":
        
        FovX = focal2fov(focal_pixels[0], cfg.data.training_resolution)
        FovY = focal2fov(focal_pixels[1], cfg.data.training_resolution)

        projection_matrix = getProjectionMatrix(
            znear=cfg.data.znear, zfar=cfg.data.zfar,
            fovX=FovX, 
            fovY=FovY
        ).transpose(0, 1).to(device)
    else: 
        projection_matrix = getProjectionMatrix(
                znear=cfg.data.znear, zfar=cfg.data.zfar,
                fovX=cfg.data.fov * 2 * np.pi / 360, 
                fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1).to(device)

    # Compute full projection transform
    full_proj_transform = (original_world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    
    
    # print("transformed_splats_new['xyz'].shape", transformed_splats_new["xyz"].shape)   
    # Render image
    pred_image_dict = render_predicted(
        transformed_splats_new, 
        original_world_view_transform,
        full_proj_transform,
        original_camera_center,
        background, cfg, 
        focals_pixels=focal_pixels)

    pred_image = pred_image_dict["render"].unsqueeze(0)

    if return_centroid and return_depth: 
        return pred_image, object_centroid, pred_image_dict["invdepths"].unsqueeze(0)
    if return_depth: 
        return pred_image, pred_image_dict["invdepths"].unsqueeze(0)
    if return_centroid: 
        return pred_image, object_centroid
    return pred_image

def render_with_custom_camera(splats, background, cfg, focal_pixels, rotation, zgt, device = 'cuda', return_splats=False, translation=None, zgt_ood=None, rotate_at_origin=True, return_depth=False, return_rgb=False, return_centroid=False): 
    # Object centroid
    object_centroid = torch.tensor([0., 0., zgt]).to(device)
    if zgt_ood is not None: 
        object_centroid_ood = object_centroid.clone()
        object_centroid_ood[2] = zgt_ood
    # Create the inverse rotation matrix
    R_inv = rotation.T
 
    # Create translation matrices
    T_to_origin = torch.eye(4, device=device)   
    T_to_origin[:3, 3] = -object_centroid

    T_back = torch.eye(4, device=device)
    T_back[:3, 3] = object_centroid
    if zgt_ood is not None: 
        T_back[:3, 3] = object_centroid_ood

    # Create the rotation matrix (4x4)
    R_4x4 = torch.eye(4, device=device)
    R_4x4[:3, :3] = R_inv
    
    R_quat = matrix_to_quaternion(R_inv)

    # print("R_4x4 requires grad: ", R_4x4.requires_grad)
    
    T_translate = torch.eye(4, device=device)
    if translation is not None: 
        T_translate[:3, 3] = translation

    # Compute the full transformation matrix
    if rotate_at_origin: 
        full_transform = T_translate @ T_back @ R_4x4 @ T_to_origin
    else: 
        full_transform = T_translate @ R_4x4
    full_transform = full_transform


    # Apply the transformation to the splats
    transformed_splats_new = {}
    for k, v in splats.items():
        
        if k == 'xyz':  # Transform positions
            homogeneous_coords = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
            transformed_coords = (full_transform @ homogeneous_coords.T).T
            transformed_splats_new[k] = transformed_coords[:, :3].contiguous()
        
        elif k == 'rotation': 
            covar_rots = v
            R_quat_expanded = R_quat.unsqueeze(0).expand(*covar_rots.shape)
            covar_rots_new = quaternion_raw_multiply(R_quat_expanded, covar_rots)
            transformed_splats_new[k] = covar_rots_new
            
        else:  # Keep other attributes unchanged
            transformed_splats_new[k] = v.contiguous()
    
    
    if return_splats and not return_rgb and not return_depth: 
        if return_centroid: 
            return transformed_splats_new, object_centroid
        else: 
            return transformed_splats_new
    what_to_return = [] 
    # Keep the camera at its original position
    original_world_view_transform = torch.eye(4).to('cuda')
    original_camera_center = torch.tensor([0., 0., 0.], device='cuda')
    # Compute FoV and projection matrix as before
    if cfg.data.category == "hydrants" or cfg.data.category == "teddybears" or cfg.data.category == "motorcycles" or cfg.data.category == "plants" or cfg.data.category == "vases":
        
        FovX = focal2fov(focal_pixels[0], cfg.data.training_resolution)
        FovY = focal2fov(focal_pixels[1], cfg.data.training_resolution)

        projection_matrix = getProjectionMatrix(
            znear=cfg.data.znear, zfar=cfg.data.zfar,
            fovX=FovX, 
            fovY=FovY
        ).transpose(0, 1).to(device)
    else: 
        projection_matrix = getProjectionMatrix(
                znear=cfg.data.znear, zfar=cfg.data.zfar,
                fovX=cfg.data.fov * 2 * np.pi / 360, 
                fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1).to(device)

    # Compute full projection transform
    full_proj_transform = (original_world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    
    
    # print("transformed_splats_new['xyz'].shape", transformed_splats_new["xyz"].shape)   
    # Render image
    pred_image_dict = render_predicted(
        transformed_splats_new, 
        original_world_view_transform,
        full_proj_transform,
        original_camera_center,
        background, cfg, 
        focals_pixels=focal_pixels)

    pred_image = pred_image_dict["render"].unsqueeze(0)
    if return_splats: 
        what_to_return.append(transformed_splats_new)
    what_to_return.append(pred_image)
    if return_centroid: 
        what_to_return.append(object_centroid)
    if return_depth: 
        what_to_return.append(pred_image_dict["invdepths"].unsqueeze(0))
    return tuple(what_to_return) if len(what_to_return) > 1 else what_to_return[0]
    



def diffae_cycle_consistency_loss(input_images, fixed_xt, generator, T=12): 
    """ computes the cycle consistency loss for the diffusion autoencoder """
    # Encode the input images
    encoded = generator.encode(input_images)
    # Decode the encoded images
    decoded = generator(fixed_xt, encoded, T=T)
    decoded = decoded.clamp(0, 1) # clamp to [0, 1]
    
    loss = F.mse_loss(input_images, decoded, reduction='none')
    
    loss = loss.view(loss.size(0), -1).mean(dim=1)
    return loss

def get_mse_loss(input_images, gt_images): 
    loss = F.mse_loss(input_images, gt_images, reduction='none')
    loss = loss.view(loss.size(0), -1).mean(dim=1)
    return loss

def get_priored_latents(search_cond, fixed_xt, generator, T=12): 
    # get the prior latents
    # decode the search_cond 
    generator_transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    decoded_images = generator(fixed_xt, search_cond, T=T)
    decoded_images= decoded_images.clamp(0, 1)
    decoded_images_transformed = generator_transform(decoded_images)
    encoded_latents = generator.encode(decoded_images_transformed)
    return encoded_latents

def get_masked_image(mask_unthresh): 

    # Assuming mask_unthresh is already loaded and has shape (3, 128, 128)
    # Convert to grayscale by averaging across the color channels
    mask_gray = mask_unthresh.mean(dim=0)  # Shape will be (128, 128)

    # Flatten the grayscale image for threshold computation
    mask_flat = mask_gray.view(-1)

    # Sort the pixel intensities
    sorted_pixels, _ = torch.sort(mask_flat)

    # Split into two halves
    midpoint = sorted_pixels.numel() // 2
    lower_half = sorted_pixels[:midpoint]
    upper_half = sorted_pixels[midpoint:]

    # Calculate the mean of each half
    mean_lower = lower_half.mean().item()
    mean_upper = upper_half.mean().item()

    # Compute threshold as the midpoint between the two means
    threshold = (mean_lower + mean_upper) / 2

    # Apply the threshold to create the binary mask
    binary_mask = (mask_gray < threshold).float()  # Foreground is 1, background is 0

   
    return binary_mask 


def get_mask(input_images, train_data, gaussian_predictor, cur_num_rotations, cfg): 
    with torch.no_grad(): 
        gaussian_splats_vis = gaussian_predictor(
                input_images, 
                train_data["view_to_world_transforms"][:1, :cfg.data.input_images, ...].repeat(cur_num_rotations, 1, 1, 1),
                train_data["source_cv2wT_quat"][:1, :cfg.data.input_images].repeat(cur_num_rotations, 1, 1),
                None,
            )
        features_dc_zero = torch.zeros_like(gaussian_splats_vis["features_dc"])
        features_rest_zero = torch.zeros_like(gaussian_splats_vis["features_rest"])
        gaussian_splats_vis["features_dc"] = features_dc_zero
        gaussian_splats_vis["features_rest"] = features_rest_zero
        
        background = torch.tensor([1, 1, 1] , dtype=torch.float32, device=device)
        gaussian_splats_vis = {k: v[0] for k, v in gaussian_splats_vis.items()}
        pred_image = render_predicted(
            gaussian_splats_vis, 
            train_data["world_view_transforms"][:, 0],
            train_data["full_proj_transforms"][:, 0],
            train_data["camera_centers"][:, 0],
            
            background, cfg,
            focals_pixels=None
        )
        mask_unthresh = pred_image["render"].detach().cpu()
        mask = get_masked_image(mask_unthresh)
    return mask

def get_orig_diffae_cc_loss(input_images, fixed_xt, orig_generator): 
    generator_transform = transforms.Compose([transforms.Resize(conf_generator.img_size), transforms.CenterCrop(conf_generator.img_size), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    with torch.no_grad(): 
        input_images_generator = generator_transform(input_images)
        encoded_input_images = orig_generator.encode(input_images_generator)
        decoded_images = orig_generator(fixed_xt, encoded_input_images, T=12)
    loss = F.mse_loss(input_images, decoded_images, reduction='none')
    loss = loss.view(loss.size(0), -1).mean(dim=1)  
    return loss


def noise_regularize_stylegan(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss
def get_lr_stylegan(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def latent_noise_stylegan(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def noise_normalize_stylegan_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)
        
        
class Space_Regularizer_StyleGAN:
    def __init__(self, original_G, lpips_net, reg_alpha, reg_l2_lambda, reg_lpips_lambda, wandb = None):
        self.original_G = original_G
        self.morphing_regularizer_alpha = reg_alpha
        self.lpips_loss = lpips_net
        self.reg_l2_lambda = reg_l2_lambda
        self.reg_lpips_lambda = reg_lpips_lambda
        self.wandb = wandb

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = self.morphing_regularizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regularizer_alpha * fixed_w + (1 - self.morphing_regularizer_alpha) * new_w_code

        return result_w

    
    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, w_batch, pt_noises):
        loss = 0.0
        noise_sample = torch.randn(num_of_sampled_latents, 512, device=device)
        w_samples = self.original_G.style(noise_sample)
        # w_samples = self.original_G.style(torch.from_numpy(z_samples).to(device), None,
        #                                     truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples]

        for w_code in territory_indicator_ws:
            # new_img = new_G.synthesis(w_code, noise_mode='none', force_fp32=True)
            new_img, _ = new_G(
                [w_code], 
                input_is_latent=True, noise=pt_noises
            )
            with torch.no_grad():
                # old_img = self.original_G.synthesis(w_code, noise_mode='none', force_fp32=True)
                old_img, _ = self.original_G(
                    [w_code], 
                    input_is_latent=True, noise=pt_noises
                )

            if self.reg_l2_lambda > 0:
                l2_loss_val = F.mse_loss(old_img, new_img)
                loss += l2_loss_val * self.reg_l2_lambda

            if self.reg_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * self.reg_lpips_lambda

        return loss / len(territory_indicator_ws), loss_lpips.detach(), l2_loss_val.detach()

    def space_regularizer_loss(self, new_G, w_batch, latent_ball_num_of_samples,pt_noises):
        ret_val, loss_lpips, l2_loss_val = self.ball_holder_loss_lazy(new_G, latent_ball_num_of_samples, w_batch, pt_noises)
        return ret_val, loss_lpips, l2_loss_val
    

def alpha_blend_with_background(pil_image, background=(255, 255, 255)):
    pil_image = pil_image.convert("RGBA")
    bg = Image.new("RGB", pil_image.size, background)
    bg.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
    return bg
import os
import torch

import torch

def sample_se3_translations(
    num_rotations: int,
    num_latents:   int,
    translation_magnitude_z: float,
    k_x: float,
    k_y: float,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Samples translations exactly like look_at_se3:
      tx ∼ Uniform([−z·k_x, +z·k_x])
      ty ∼ Uniform([−z·k_y, +z·k_y])
      tz ∼ Uniform([−z,      +z     ])
    and returns a (num_rotations * num_latents, 3) tensor.
    """
    # compute per‐axis bounds
    tx_mag = translation_magnitude_z * k_x
    ty_mag = translation_magnitude_z * k_y
    tz_mag = translation_magnitude_z

    total = num_rotations * num_latents

    # draw them
    tx = torch.empty(total, device=device).uniform_(-tx_mag, tx_mag)
    ty = torch.empty(total, device=device).uniform_(-ty_mag, ty_mag)
    tz = torch.empty(total, device=device).uniform_(-tz_mag, tz_mag)

    translations = torch.stack([tx, ty, tz], dim=1)
    translations.requires_grad_(True)
    return translations
    
def foreground_mask(img: torch.Tensor, thresh: float = 0.98) -> torch.Tensor:
    """
    img : (3,H,W) in [0,1]
    Returns a boolean mask (H,W) where at least one channel is < thresh.
    Works for white background; tweak thresh if you change bg colour.
    """
    # background is ~1.0; object pixels are < thresh
    return (img < thresh).any(dim=0)