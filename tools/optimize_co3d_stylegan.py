# imports 
# -------------------------------------------------------------------------------------- # 
from infer3d import config as _cfg
ROOT = _cfg.SPLATTER_REPO_ROOT
from math import e
import sys 
from filelock import FileLock
import time 
import json 
sys.path.append(ROOT)
import csv 
sys.path.append(_cfg.STYLEGAN3_ROOT) # stylegan3
import datetime
import wandb 
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import lpips as lpips_lib
import argparse
from infer3d.utils.optim import *
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import dnnlib
import legacy
from infer3d.utils.general import PILtoTorch
from infer3d.model import GaussianSplatPredictor
from tqdm import tqdm
from hydra import initialize, compose
import random 
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from infer3d.data.co3d import CO3DDataset
# (co3d_fastgs removed: only used in commented-out HQ code)
import imageio
from infer3d.renderer import render_predicted

import hydra

# -------------------------------------------------------------------------------------- # 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = None 
    
class StyleGANCondGenerator(nn.Module): 
    def __init__(self, checkpoint_path, size=128):
        super().__init__()
        # Load StyleGAN3 G_ema from .pkl
        with dnnlib.util.open_url(checkpoint_path) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(device)
        self.G.eval()
        # Provide a compact compat API used elsewhere in this file
        class _Compat:
            def __init__(self, G):
                self.G = G
            def style(self, z, c=None):
                ws = self.G.mapping(z, c)
                return ws[:, 0, :]
            def make_noise(self):
                return []
        self.model = _Compat(self.G)

    def forward(self, cond = None, n_sample=1, truncation = 0.7, noises = None, label= None):
        assert label is not None, "Please provide a label"
        if cond is None: 
            z = torch.randn(n_sample, self.G.z_dim, device=device)
            w = self.model.style(z, label)
        else: 
            w = self.model.style(cond, label)
        return self.forward_latent_w(w)
    
    def forward_latent_w(self, latent_w, noises = None): 
        if latent_w.dim() == 1:
            latent_w = latent_w.unsqueeze(0)
        ws = latent_w.unsqueeze(1).repeat(1, self.G.num_ws, 1)
        img = self.G.synthesis(ws)
        img = img.clamp(-1, 1)
        img = (img + 1) / 2 # normalize to [0, 1]
        return img
        
    def encode(self, img):
        raise NotImplementedError("Not implemented")

def make_poses_relative_to_first(images_and_camera_poses):
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

def get_source_cw2wT(source_cameras_view_to_world):
    from infer3d.utils.general import matrix_to_quaternion
    qs = []
    for c_idx in range(source_cameras_view_to_world.shape[0]):
        qs.append(matrix_to_quaternion(source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
    return torch.stack(qs, dim=0)

def make_data_relative_to(target_data, reference_data, batch_size = 1): 
    comb_data = {}
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
        else: 
            comb_data["world_view_transforms"] = torch.cat([comb_data["world_view_transforms"], data["world_view_transforms_absolute"]], dim=1)
            comb_data["view_to_world_transforms"] = torch.cat([comb_data["view_to_world_transforms"], data["view_to_world_transforms_absolute"]], dim=1)
            comb_data["full_proj_transforms"] = torch.cat([comb_data["full_proj_transforms"], data["full_proj_transforms_absolute"]], dim=1)
            comb_data["camera_centers"] = torch.cat([comb_data["camera_centers"], data["camera_centers_absolute"]], dim=1)
    
    comb_data = make_poses_relative_to_first(comb_data)
    source_cw2wTs = []
    for batch_idx in range(batch_size): 
        cur_source_cw2wT = get_source_cw2wT(comb_data["view_to_world_transforms"][batch_idx])
        source_cw2wTs.append(cur_source_cw2wT)
        
    comb_data["source_cv2wT_quat"] = torch.stack(source_cw2wTs)  
    # return only the relative data
    comb_data["world_view_transforms"] = comb_data["world_view_transforms"][:, num_across_first_dim:]
    comb_data["view_to_world_transforms"] = comb_data["view_to_world_transforms"][:, num_across_first_dim:]
    comb_data["full_proj_transforms"] = comb_data["full_proj_transforms"][:, num_across_first_dim:]
    comb_data["camera_centers"] = comb_data["camera_centers"][:, num_across_first_dim:]
    comb_data["source_cv2wT_quat"] = comb_data["source_cv2wT_quat"][:, num_across_first_dim:]
    comb_data["gt_images"] = target_data["gt_images"]
    assert comb_data["world_view_transforms"].shape[1] == target_data["world_view_transforms_absolute"].shape[1], f"world_view_transforms should have shape {target_data['world_view_transforms_absolute'].shape[1]} cf. {comb_data['world_view_transforms'].shape[1]}"
    
    return comb_data

def create_loop_and_eval(gaussian_splats, test_data_instance, test_data_name, cfg, focals_pixels, save_loop_path=None, save_gt=False, use_absolute_poses=False):
    """Create video loop and compute metrics"""
    from infer3d.utils.loss import ssim as ssim_fn
    
    psnr_all = []
    ssim_all = []
    lpips_all = []
    
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
    
    background = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], dtype=torch.float32, device=device)
    loop_renders = []
    gt_images = []
    
    test_data_instance = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in test_data_instance.items()}
    
    if use_absolute_poses:
        all_world_view_transforms = test_data_instance["world_view_transforms_absolute"]
        all_full_proj_transforms = test_data_instance["full_proj_transforms_absolute"]
        all_camera_centers = test_data_instance["camera_centers_absolute"]
    else:
        all_world_view_transforms = test_data_instance["world_view_transforms"]
        all_full_proj_transforms = test_data_instance["full_proj_transforms"]
        all_camera_centers = test_data_instance["camera_centers"]
    for r_idx in range(all_world_view_transforms.shape[1]):
        world_view_transforms = all_world_view_transforms[0, r_idx].unsqueeze(0)
        full_proj_transforms = all_full_proj_transforms[0, r_idx].unsqueeze(0)
        camera_centers = all_camera_centers[0, r_idx].unsqueeze(0)
        
        if focals_pixels is not None:
            cur_focals_pixels = focals_pixels[0, r_idx]
        else:
            cur_focals_pixels = None

        
        image = render_predicted(gaussian_splats,
                                    world_view_transforms,
                                    full_proj_transforms,  
                                    camera_centers,
                                    background,
                                    cfg,
                                    focals_pixels=cur_focals_pixels)["render"]
        gt_image = test_data_instance["gt_images"][0, r_idx].to(device)

        gt_image = F.interpolate(gt_image.unsqueeze(0), size=(image.shape[-2], image.shape[-1]), mode="bilinear", align_corners=False).squeeze(0)
        
        # Compute metrics
        lpips = lpips_fn(image.unsqueeze(0) * 2 - 1, gt_image.unsqueeze(0) * 2 - 1).item()
        psnr = -10 * torch.log10(torch.mean((image - gt_image) ** 2, dim=[0, 1, 2])).item()
        ssim = ssim_fn(image, gt_image).item()
        
        psnr_all.append(psnr)
        ssim_all.append(ssim)
        lpips_all.append(lpips)
        
        # Store for video
        gt_images.append(torch.clamp(gt_image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    
    scores = {
        "PSNR_novel": sum(psnr_all) / len(psnr_all),
        "SSIM_novel": sum(ssim_all) / len(ssim_all),
        "LPIPS_novel": sum(lpips_all) / len(lpips_all),
    }
    
    if save_loop_path:
        imageio.mimsave(save_loop_path, loop_renders, fps=4, codec='libx264')
        if save_gt:
            imageio.mimsave(save_loop_path.replace(".mp4", "_gt.mp4"), gt_images, fps=4, codec='libx264')
    
    return loop_renders, gt_images, scores

@torch.no_grad()
def eval_baseline(ood_data, ood_image_index, gaussian_predictor, cfg, background, save_video_path=None, use_ood_image=None, ood_rotation=None, zgt=None, ood_centroid=None):
    """Evaluate baseline: just run splatter image on ood_image directly
    
    Args:
        use_ood_image: If provided, use this OOD image instead of the one from ood_data
        ood_rotation: The rotation matrix used to generate the OOD image (needed to transform splats back)
        zgt: The depth value (needed for inverse transformation)
        ood_centroid: The centroid of the OOD object (needed for inverse transformation)
    """
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    
    if use_ood_image is not None:
        # Use the provided OOD image (generated from random rotation)
        # ood_image comes from render_with_custom_camera which has shape [1, C, H, W]
        ood_image = use_ood_image.squeeze(0).to(device)  # Remove batch dim to get [C, H, W]
    else:
        # Use the in-distribution image
        ood_image = ood_data["gt_images"][0, ood_image_index].to(device)

    # Force both image and origin_distances to cfg.data.training_resolution
    target_res = cfg.data.training_resolution
    assert target_res == 128, f"target_res should be 128 cf. {target_res}"
    if ood_image.shape[-2:] != (target_res, target_res):
        ood_image = F.interpolate(
            ood_image.unsqueeze(0),  # [1, C, H, W]
            size=(target_res, target_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    if ood_origin_distances.shape[-2:] != (target_res, target_res):
        ood_origin_distances = F.interpolate(
            ood_origin_distances.unsqueeze(0),  # [1, 1, H, W]
            size=(target_res, target_res),
            mode="nearest",
        ).squeeze(0)
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    
    # Prepare input
    input_images = torch.cat([ood_image.unsqueeze(0).unsqueeze(1), 
                              ood_origin_distances.unsqueeze(0).unsqueeze(1)], dim=2)
    focals_pixels_pred = ood_focals_pixels.unsqueeze(0).unsqueeze(0)
    
    # Run splatter image predictor directly on ood_image
    pred_splats = gaussian_predictor(
        input_images, 
        ood_data["view_to_world_transforms"][:1, ood_image_index:ood_image_index+cfg.data.input_images], 
        ood_data["source_cv2wT_quat"][:1, ood_image_index:ood_image_index+cfg.data.input_images], 
        focals_pixels_pred   
    )
    pred_splats = {k: v[0] for k, v in pred_splats.items()}
    
    # If this was an OOD image from random rotation, we need to undo that rotation
    if use_ood_image is not None and ood_rotation is not None:
        # The OOD image was rendered from rotated splats, so predicted splats are in rotated frame
        # Apply inverse rotation to bring them back to original coordinate frame
        inverse_rotation = ood_rotation.T  # Transpose for inverse rotation
        pred_splats = render_with_custom_camera_align(
            pred_splats, background, cfg, ood_focals_pixels,
            inverse_rotation, zgt, device=device, return_splats=True, translation=None,
            zgt_ood=zgt,
            override_centroid=ood_centroid,
        )
    
    # Evaluate on the same ood_data views (they're already in the right coordinate frame)
    renders, gt_images, scores = create_loop_and_eval(pred_splats, ood_data, "test", cfg, 
                                                        ood_data["focals_pixels"], save_loop_path=save_video_path, save_gt=True, use_absolute_poses=False)
    
    return renders, gt_images, scores

@torch.no_grad()
def eval_pred(best_input_image, best_rotation_matrix, best_translation_matrix, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=None, ood_rotation=None, ood_centroid=None): 
    """Evaluate our optimized method
    
    Args:
        ood_rotation: The rotation matrix used to generate the OOD image (needed to transform splats back)
        ood_centroid: The centroid of the OOD object (needed for inverse transformation)
    """
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    ood_focals_pixels = ood_data["focals_pixels"][0, ood_image_index].to(device)
    
    # Extract only RGB channels from best_input_image (it may have origin_distances concatenated)
    if best_input_image.shape[1] > 3:
        best_input_image = best_input_image[:, :3, :, :]  # Take only RGB channels

    # Force best_input_image and ood_origin_distances to cfg.data.training_resolution
    target_res = cfg.data.training_resolution
    if best_input_image.shape[-2:] != (target_res, target_res):
        best_input_image = F.interpolate(
            best_input_image,
            size=(target_res, target_res),
            mode="bilinear",
            align_corners=False,
        )

    if ood_origin_distances.shape[-2:] != (target_res, target_res):
        ood_origin_distances = F.interpolate(
            ood_origin_distances.unsqueeze(0),  # [1, 1, H, W]
            size=(target_res, target_res),
            mode="nearest",
        ).squeeze(0)
    
    # Prepare input
    input_images = torch.cat([best_input_image.unsqueeze(1).to(device), 
                              ood_origin_distances.unsqueeze(0).unsqueeze(1)], dim=2)
    focals_pixels_pred = ood_focals_pixels.unsqueeze(0).unsqueeze(0)
    
    pred_splats = gaussian_predictor(
        input_images, 
        ood_data["view_to_world_transforms"][:1, ood_image_index:ood_image_index+cfg.data.input_images], 
        ood_data["source_cv2wT_quat"][:1, ood_image_index:ood_image_index+cfg.data.input_images], 
        focals_pixels_pred   
    )
    pred_splats = {k: v[0] for k, v in pred_splats.items()}
    
    # Apply the optimized rotation/translation
    transformed_splats = render_with_custom_camera(pred_splats, 
                                                   background, 
                                                   cfg, 
                                                   ood_focals_pixels,
                                                   best_rotation_matrix.to(device),
                                                   zgt, device = device, return_splats=True, translation=best_translation_matrix.to(device))
    
    # If OOD image was from random rotation, undo that rotation to get back to original frame
    if ood_rotation is not None:
        inverse_rotation = ood_rotation.T  # Transpose for inverse rotation
        transformed_splats = render_with_custom_camera_align(
            transformed_splats, background, cfg, ood_focals_pixels,
            inverse_rotation, zgt, device=device, return_splats=True, translation=None,
            zgt_ood=zgt,
            override_centroid=ood_centroid,
        )
    
    # Evaluate on ood_data views directly (they're already in the right coordinate frame)
    renders, gt_images, scores = create_loop_and_eval(transformed_splats, ood_data, "test", cfg,
                                                        ood_data["focals_pixels"], save_loop_path=save_video_path, save_gt=True, use_absolute_poses=False)
    
    return renders, gt_images, scores


def load_dataset(dataset_name, cfg, use_hq=False):
    if dataset_name == "hydrants": 
        print("loading co3d hydrants")
        # dataset = CO3DFastGSDataset(cfg, "train", use_hq=use_hq, override_training_resolution=1080)
        # val_dataset = CO3DFastGSDataset(cfg, "test", use_hq=use_hq, override_training_resolution=1080)
        dataset = CO3DDataset(cfg, "train", use_hq=use_hq)
        val_dataset = CO3DDataset(cfg, "test", use_hq=use_hq)
    elif dataset_name == "teddybears":
        print("loading co3d teddybears")
        dataset = CO3DDataset(cfg, "train", use_hq=use_hq)
        val_dataset = CO3DDataset(cfg, "test", use_hq=use_hq)
    elif dataset_name == "motorcycles":
        print("loading co3d motorcycles")
        dataset = CO3DDataset(cfg, "train", use_hq=use_hq)
        val_dataset = CO3DDataset(cfg, "test", use_hq=use_hq)
    else: 
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset, val_dataset

def load_models(): 
    gaussian_predictor = GaussianSplatPredictor(cfg)
    gaussian_predictor = gaussian_predictor.to(memory_format=torch.channels_last)
    gaussian_predictor = gaussian_predictor.to(device)
    
    if cfg.opt.pretrained_ckpt is not None: 
        model_path = cfg.opt.pretrained_ckpt
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
        print(f'Loaded splatter image model from {model_path}')

    # Use StyleGAN3 snapshot (.pkl)
    if cfg.data.category == "hydrants":
        generator_ckpt_path = _cfg.STYLEGAN_CKPTS["hydrants"]
    else:
        raise ValueError(f"Dataset {cfg.data.category} not supported")

    generator = StyleGANCondGenerator(generator_ckpt_path)
        
    return gaussian_predictor, generator

def idx_to_label(idx, G, num_samples): 
    label = torch.zeros(num_samples, G.c_dim, device=device)
    label[:, idx] = 1
    return label

def load_test_image(test_img_path, size: int = 128): 
    image = Image.open(test_img_path)
    image = image.convert("RGB")
    image = image.resize((size, size))
    image = np.array(image) / 255.0
    return image

def load_test_checkpoint(test_img_path): 
    checkpoint_path = test_img_path.replace(".png", ".pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint

used_idxes = []
sorted_idxs_by_loss = []
best_loss = float("inf")
log_iters = 0
log_every = 250
lambda_lpips = 0
lambda_mse = 0
batch_size = 32
num_batches = 0
rot_optim = True 
is_rot_optim = True 

is_debugging = False

wandb_run = None
val_dataset = None
dataset = None

def train(cur_cfg, override_example_id = None, test_idx = 0, log_path = "checkpoints-co3d-se3", val_view = None, dataset_name = None, prefix = None, test_img_path = None): 
    # global variables for easier access 
    global used_idxes, sorted_idxs_by_loss, best_loss
    global log_iters, log_every, lambda_lpips, lambda_mse
    global batch_size, num_batches
    global is_debugging
    global cfg
    global rot_optim, is_rot_optim
    global wandb_run
    global dataset, val_dataset
    assert test_img_path is not None, "Please provide a test image path"
    
    torch.set_float32_matmul_precision('high')
    
    # assign config 
    cfg = cur_cfg
    # load 3d model and image generator  
    gaussian_predictor, generator = load_models()
    
    # experiment file path 
    experiment_file_path = f"{log_path}/{override_example_id}-{datetime.datetime.now().strftime('%d-%m-%y-%H-%M-%S')}"

    # makedirs for logging and saving result
    os.makedirs(experiment_file_path, exist_ok=True)
    
    # random seed for reproducibility
    cfg.general.random_seed = cfg.abs.random_seed
    cfg.abs.random_seed = abs(hash(override_example_id)) % 2**32
    g = torch.Generator()
    np.random.seed(cfg.abs.random_seed)
    torch.manual_seed(cfg.abs.random_seed)
    torch.cuda.manual_seed(cfg.abs.random_seed)
    random.seed(cfg.abs.random_seed)
    g.manual_seed(cfg.abs.random_seed)

    
    
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True, generator=g)


    log_every = cfg.abs.log_every
    
    # initialize wandb 
    dict_cfg = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )

    combined_config = {**dict_cfg}
    
    if wandb_run is None: 
        if cfg.wandb.run_name is not None: 
            wandb_run_name = cfg.wandb.run_name
        elif prefix is not None: 
            wandb_run_name = prefix
        else: 
            wandb_run_name = os.path.basename(__file__).split(".")[0] + "-" + dataset_name + "-"
            
        wandb_run = wandb.init(project=cfg.wandb.project, reinit=True, name=wandb_run_name + datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S"),
                                config=combined_config)

    # background for rendering 
    background = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0], dtype=torch.float32).to(device)
    
    gaussian_predictor.eval()

    torch.cuda.empty_cache()

    # ========= Generate OOD image from random SO3 pose =========
    # Get a training sample to generate OOD from

    # get the ood data corresponding to override_example_id
    # val_iter = iter(val_dataloader)
    test_img_checkpoint = load_test_checkpoint(test_img_path)
    if "ood_data" not in test_img_checkpoint:
        ood_seq_index, _ = val_dataset.find_index_for_sequence_prefix(override_example_id)
        ood_data = val_dataset[ood_seq_index]
        print("FOUND OOD DATA FOR OVERRIDE EXAMPLE ID: ", override_example_id)
    else: 
        ood_data = test_img_checkpoint["ood_data"]
        print("FOUND OOD DATA IN CHECKPOINT FOR OVERRIDE EXAMPLE ID: ", override_example_id)
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    if len(ood_data["focals_pixels"].shape) == 2:
        ood_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in ood_data.items()}
    # let's load the checkpoint from the image_path 
    ood_zgt = torch.tensor(test_img_checkpoint["zgt"])
    ood_image_index = test_img_checkpoint["ood_image_index"]
    ood_random_rotation = test_img_checkpoint["ood_random_rotation"].to(device)
    ood_focals_pixels = ood_data["focals_pixels"][:, ood_image_index:ood_image_index+1]
    ood_centroid = test_img_checkpoint["centroid"].to(device)
    
    ood_image = load_test_image(test_img_path, cfg.data.training_resolution)
    ood_image = torch.tensor(np.clip(ood_image, 0, 1))
    ood_image = ood_image.permute(2, 0, 1).float()
    ood_image = ood_image.unsqueeze(0).to(device)

    # at this point we can compare in-distribution to OOD image 
    @torch.no_grad()
    def compare_indist_to_ood(): 
        ood_image_index_here = ood_image_index
        # indist image 
        indist_image = ood_data["gt_images"][:, 0].unsqueeze(0)
        indist_origin_distances = ood_data["origin_distances"][:, 0].unsqueeze(0)
        indist_origin_distances = F.interpolate(indist_origin_distances.squeeze(0), size=(128, 128), mode="bilinear", align_corners=False).unsqueeze(0)
        cur_zgt = indist_origin_distances.max().item()
        print(f"Using cur_zgt: {cur_zgt}")
        indist_image = F.interpolate(indist_image.squeeze(0), size=(128, 128), mode="bilinear", align_corners=False).unsqueeze(0)
        indist_image = torch.cat([indist_image, indist_origin_distances], dim=2)
        indist_view_to_world_transforms = ood_data["view_to_world_transforms"][:, 0].unsqueeze(0)
        indist_source_cv2wT_quats = ood_data["source_cv2wT_quat"][:, 0].unsqueeze(0)
        indist_world_view_transforms = ood_data["world_view_transforms"][:, 0].unsqueeze(0)
        indist_full_proj_transforms = ood_data["full_proj_transforms"][:, 0].unsqueeze(0)
        indist_camera_centers = ood_data["camera_centers"][:, 0].unsqueeze(0)
        # get splats 
        indist_splats = gaussian_predictor(indist_image, indist_view_to_world_transforms, indist_source_cv2wT_quats, ood_focals_pixels)
        indist_splats = {k: v[0].contiguous() for k, v in indist_splats.items()}
        # render from first camera with render_predicted
        indist_render_image = render_predicted(indist_splats, indist_world_view_transforms, indist_full_proj_transforms, indist_camera_centers, background, cfg, focals_pixels=ood_focals_pixels[0, 0])["render"]
        
        # render with custom camera 
        # ood_render_image = render_with_custom_camera(
        #     indist_splats, 
        #     background, 
        #     cfg, 
        #     ood_focals_pixels[0, 0],
        #     ood_random_rotation, 
        #     ood_zgt, 
        #     device=device
        # )
        
        ood_render_image = render_with_custom_camera_align(
            indist_splats, 
            background, 
            cfg, 
            ood_focals_pixels[0, 0],
            ood_random_rotation, 
            ood_zgt, 
            device=device, 
            return_splats=False,
            translation=None,
            zgt_ood=ood_zgt,
            override_centroid=ood_centroid,
        )


        # send to wandb 
        wandb.log({
            "indist_image": wandb.Image(torch.clip(indist_image.squeeze(), 0, 1), caption="Indist Image"),
            "indist_render_image": wandb.Image(torch.clip(indist_render_image.squeeze()[:3], 0, 1), caption="Indist Render Image"),
            "indist_to_ood_image": wandb.Image(torch.clip(ood_render_image.squeeze(), 0, 1), caption="Indist to OOD Image"),
            "ood_image": wandb.Image(torch.clip(ood_image.squeeze(0), 0, 1), caption="OOD Image"),
        })
    
    # compare_indist_to_ood()
    # print("DONE COMPARING INDIST TO OOD")
    # exit()

    zgt = ood_zgt
    
    torch.cuda.empty_cache()

    generator.eval()

    # freeze the generator parameters
    for param in generator.parameters():
        param.requires_grad = False

    num_rotations = cfg.abs.num_rotations
    batch_size = cfg.abs.batch_size
    num_batches = num_rotations // batch_size

    fixed_xT = torch.randn(num_rotations * cfg.abs.num_latents, 3, 128, 128, device=device)
    generator_transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_data_iter = iter(dataloader)
    train_data = next(train_data_iter)
    
    latent_std = 17.4755
    gen_class_idx = 0
    print("Using zgt: ", zgt, "using gen_class_idx: ", gen_class_idx)


    
    # Z space
    init_search_cond = torch.randn(cfg.abs.num_latents, 512, device=device)
    
    with torch.no_grad(): 
        # take to W space 
        label_to_use = idx_to_label(gen_class_idx, generator.G, cfg.abs.num_latents) if generator.G.c_dim > 0 else None
        init_search_cond = generator.model.style(init_search_cond, label_to_use)
        
    assert init_search_cond.shape[0] == cfg.abs.num_latents, f"init_search_cond should have shape {cfg.abs.num_latents} cf. {init_search_cond.shape}"
    search_cond_list = [init_search_cond.clone().detach().requires_grad_(True) for _ in range(num_rotations)]
    search_cond = nn.Parameter(torch.cat(search_cond_list, dim=0))
    
    stylegan_noises_single = generator.model.make_noise()
    stylegan_noises = []
    init_num_images = cfg.abs.num_latents * cfg.abs.num_rotations
    for noise in stylegan_noises_single:
        stylegan_noises.append(noise.repeat(init_num_images, 1, 1, 1).clone().normal_().requires_grad_(True))
    
    assert len(search_cond.shape) == 2 and search_cond.shape[0] == num_rotations * cfg.abs.num_latents, f"search_cond should have shape {num_rotations * cfg.abs.num_latents, 512} cf. {search_cond.shape}"
    
    if cfg.abs.lambda_lpips != 0:
        lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
        print("Using LPIPS loss with lambda_lpips: ", cfg.abs.lambda_lpips)

    num_iterations = cfg.abs.num_iterations

    train_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in train_data.items()}

    best_loss = float('inf')
    best_pred_image = None
    best_input_image = None
    best_search_cond = None 
    best_rotation_matrix = None
    best_translation_matrix = None
    best_rotation_name = ""

    top_k = 10
    assert top_k <= num_rotations, f"top_k should be less than or equal to num_rotations cf. {top_k} and {num_rotations}"
    top_k_pred_images = []
    top_k_input_images = []
    top_k_losses = []
    top_k_rotation_matrices = []
    top_k_translation_matrices = []
    top_k_indexes = []

    random_rotations = get_random_cameras(num_rotations, zgt, device=device)
    random_rotations = torch.stack([random_rotations.clone() for _ in range(cfg.abs.num_latents)], dim=1)
    random_rotations = random_rotations.reshape(-1, 3, 3)
    random_rotations.requires_grad_(True)
    
    random_translations = torch.zeros(num_rotations * cfg.abs.num_latents, 3, device=device)
    random_translations.requires_grad_(True)
    
    optimizer = optim.Adam([ 
        {'params': random_rotations, 'lr': cfg.abs.lr_rotations},
        {'params': random_translations, 'lr': cfg.abs.lr_rotations},
        {'params': search_cond, 'lr': cfg.abs.w_lr},
        {'params': stylegan_noises, 'lr': cfg.abs.w_lr},
    ])

    log_iters = 0
    prev_best_loss = None

    import time
    used_idxes = list(range(num_rotations * cfg.abs.num_latents))
    sorted_idxs_by_loss = []

    # Evaluate baseline once at the start
    print("Evaluating baseline-in-dist (splatter image only with in-dist image)...")
    baseline_indist_video_path = f"{experiment_file_path}/baseline_indist_video.mp4"
    with torch.no_grad():
        baseline_indist_renders, baseline_indist_gt_images, baseline_indist_scores = eval_baseline(
            ood_data, ood_image_index, 
            gaussian_predictor, cfg, background, save_video_path=baseline_indist_video_path, 
            use_ood_image=None, ood_rotation=None, zgt=zgt, ood_centroid=ood_centroid
        )
    print(f"Baseline-in-dist scores: {baseline_indist_scores}")
    
    print("Evaluating baseline-ood (splatter image only with OOD image)...")
    baseline_ood_video_path = f"{experiment_file_path}/baseline_ood_video.mp4"
    with torch.no_grad():
        baseline_ood_renders, baseline_ood_gt_images, baseline_ood_scores = eval_baseline(
            ood_data, ood_image_index, 
            gaussian_predictor, cfg, background, save_video_path=baseline_ood_video_path, 
            use_ood_image=ood_image, ood_rotation=ood_random_rotation, zgt=zgt, ood_centroid=ood_centroid
        )
    
    print("DONE EVALUATING BASELINES")
    # Initialize timing and configuration
    config_steps = [
        {
            "steps": 0,
            "settings": {
                "used_idxes": list(range(num_rotations * cfg.abs.num_latents)),
                "sorted_idxs_by_loss": [],
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 250,
                "lambda_lpips": cfg.abs.lambda_lpips,
                "lambda_mse": cfg.abs.lambda_mse,
                "batch_size": 32,
                "num_batches": lambda: int(len(used_idxes) / batch_size),
            }
        },
        {
            "steps": 3,
            "settings": {
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 250,
                "lambda_lpips": cfg.abs.lambda_lpips,
                "lambda_mse": cfg.abs.lambda_mse,
                "batch_size": 32,
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: int(len(used_idxes) / batch_size),
            }
        },
        {
            "steps": 122,
            "settings": {
                "batch_size": 10,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 250,
                "used_idxes": lambda: sorted_idxs_by_loss[:10] if batch_size < 10 else sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: int(len(used_idxes) / batch_size),
            }
        },
        {
            "steps": 302,
            "settings": {
                "batch_size": 5,
                "lambda_lpips": 2,
                "lambda_mse": 10,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: int(len(used_idxes) / batch_size),
            },
        },
        {
            "steps": 732,
            "settings": {
                "batch_size": 5,
                "lambda_mse": 2,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: int(len(used_idxes) / batch_size),
            },
        },
        {
            "steps": 782,
            "settings": {
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "batch_size": 5,
                "num_batches": lambda: int(len(used_idxes) / batch_size),
            },
        },
        {"steps": 783, "settings": "stop"}
    ]

    # Loop control
    step_idx = 0
    iteration = 0
    start_time = time.time()

    tqdm_iterator = tqdm(total=num_iterations, desc="iterations")

    while step_idx < len(config_steps):
        all_losses_and_idxs = []
        
        elapsed_time = time.time() - start_time

        if iteration >= config_steps[step_idx]["steps"]:
            assert step_idx == 0 or len(sorted_idxs_by_loss) > 0, f"sorted_idxs_by_loss {sorted_idxs_by_loss} should not be empty"     
            settings = config_steps[step_idx]["settings"]
            if settings == "stop":
                # Extract RGB channels from best_input_image (remove origin_distances)
                best_input_rgb = best_input_image.squeeze(0)[:3, :, :]  # [1, 4, H, W] -> [3, H, W]
                
                # Log images one final time before exiting
                dict_to_log = {
                    "Original OOD Image": wandb.Image(log_ood_image, caption="Original OOD Image"),
                    "Input Images Grid": wandb.Image(input_images_grid, caption="Input Images Grid"),
                    "Best Predicted Image": wandb.Image(log_best_pred_image, caption=f"Best pred {best_rotation_name}"),
                    "Top Input Images Grid": wandb.Image(top_input_images_grid, caption="Top Input Images Grid"),
                    "Top Predicted Images Grid": wandb.Image(top_pred_images_grid, caption="Top Predicted Images Grid"),
                    "Predicted Images Grid": wandb.Image(pred_images_grid, caption="Predicted Images Grid"),
                    "best_input_image": wandb.Image(best_input_rgb, caption="Best Input Image"),
                    "best_loss": best_loss.item(), 
                    "current_stage": step_idx+1, 
                }
                if lambda_lpips > 0:
                    dict_to_log["lpips_loss"] = lpips_loss_sum.item()   

                wandb.log(dict_to_log)
                
                # Evaluate our method on top-k
                all_scores = []
                
                best_scores = None 
                avg_scores = None
                for idx in range(top_k):
                    eval_input_image = top_k_input_images[idx].unsqueeze(0)
                    eval_rotation_matrix = top_k_rotation_matrices[idx]
                    eval_translation_matrix = top_k_translation_matrices[idx]
                    ours_video_path = None
                    _, _, cur_scores = eval_pred(eval_input_image, eval_rotation_matrix, eval_translation_matrix, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=ours_video_path, ood_rotation=ood_random_rotation, ood_centroid=ood_centroid)
                    if best_scores is None:
                        best_scores = cur_scores.copy()
                        avg_scores = {key: 0 for key in cur_scores.keys()}
                    else: 
                        if cur_scores["PSNR_novel"] > best_scores["PSNR_novel"]:
                            best_scores = cur_scores.copy()
                        for key, value in cur_scores.items():
                            avg_scores[key] += value
                    all_scores.append(cur_scores)
                
                # Evaluate the known best with video
                ours_best_video_path = f"{experiment_file_path}/ours_best_video.mp4"
                known_best_renders, known_best_gt_images, known_best_scores = eval_pred(best_input_image, best_rotation_matrix, best_translation_matrix, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=ours_best_video_path, ood_rotation=ood_random_rotation, ood_centroid=ood_centroid)
                if known_best_scores["PSNR_novel"] > best_scores["PSNR_novel"]:
                    best_scores = known_best_scores.copy()
                all_scores.append(known_best_scores)
                
                for key in avg_scores.keys():
                    avg_scores[key] /= top_k 
                avg_scores = {key + "_avg": value for key, value in avg_scores.items()}
                best_scores = {key + "_best": value for key, value in best_scores.items()}
                known_best_scores = {key + "_known_best": value for key, value in known_best_scores.items()}
                baseline_indist_scores_wandb = {key + "_baseline_indist": value for key, value in baseline_indist_scores.items()}
                baseline_ood_scores_wandb = {key + "_baseline_ood": value for key, value in baseline_ood_scores.items()}
                merged_scores = {**avg_scores, **best_scores, **known_best_scores, **baseline_indist_scores_wandb, **baseline_ood_scores_wandb}
                
                print(f"Final merged scores (including baselines): {merged_scores}")
                
                # Log videos and comparison to wandb
                baseline_indist_renders_grid = vutils.make_grid(torch.tensor(baseline_indist_renders).permute(0, 3, 1, 2), nrow=3)
                baseline_ood_renders_grid = vutils.make_grid(torch.tensor(baseline_ood_renders).permute(0, 3, 1, 2), nrow=3)
                ours_renders_grid = vutils.make_grid(torch.tensor(known_best_renders).permute(0, 3, 1, 2), nrow=3)
                gt_grid = vutils.make_grid(torch.tensor(known_best_gt_images).permute(0, 3, 1, 2), nrow=3)
                
                wandb.log({
                    "Baseline In-Dist Renders Grid": wandb.Image(baseline_indist_renders_grid, caption="Baseline In-Dist Renders"),
                    "Baseline OOD Renders Grid": wandb.Image(baseline_ood_renders_grid, caption="Baseline OOD Renders"),
                    "Ours Best Renders Grid": wandb.Image(ours_renders_grid, caption="Ours Best Renders"),
                    "GT Images Grid": wandb.Image(gt_grid, caption="Ground Truth"),
                    "Baseline In-Dist Video": wandb.Video(baseline_indist_video_path, fps=4, format="mp4"),
                    "Baseline OOD Video": wandb.Video(baseline_ood_video_path, fps=4, format="mp4"),
                    "Ours Best Video": wandb.Video(ours_best_video_path, fps=4, format="mp4"),
                    **merged_scores
                })
                
                # save topk everything and best everything to experiment_file_path using torch.save
                torch.save({
                    "top_k_pred_images": top_k_pred_images,
                    "top_k_input_images": top_k_input_images,
                    "top_k_losses": top_k_losses,
                    "top_k_rotation_matrices": top_k_rotation_matrices,
                    "top_k_translation_matrices": top_k_translation_matrices,
                    "zgt": zgt,
                    "best_pred_image": best_pred_image,
                    "best_input_image": best_input_image,
                    "best_loss": best_loss,
                    "best_rotation_matrix": best_rotation_matrix,
                    "best_translation_matrix": best_translation_matrix,
                    "best_rotation_name": best_rotation_name, 
                    "best_search_cond": best_search_cond, 
                    "ood_data": ood_data, 
                    "ood_image_index": ood_image_index,
                    "ood_image": ood_image,
                    "ood_random_rotation": ood_random_rotation,
                    "ood_centroid": torch.tensor([0., 0., float(zgt)]),  # centroid = [0,0,zgt]; needed by eval pipeline
                    "example_id": override_example_id,
                    "scores": merged_scores,
                    "all_scores": all_scores,
                    "baseline_indist_scores": baseline_indist_scores,
                    "baseline_ood_scores": baseline_ood_scores
                }, os.path.join(experiment_file_path, f"topk_best_everything_latest.pth"))

                print("Reached the end of timed conditions. Exiting.")
                return merged_scores
                break
        
            print(f"Applying configuration step {step_idx} at {iteration} steps and {elapsed_time} seconds")
            for key, value in settings.items():
                if callable(value):
                    globals()[key] = value()
                else:
                    globals()[key] = value

            step_idx += 1
            assert int(len(used_idxes) / batch_size) == num_batches, f"num_batches {num_batches} should be equal to {int(len(used_idxes) / batch_size)}"
            
        if is_rot_optim and not rot_optim: 
            optimizer = optim.Adam([ 
                {'params': search_cond, 'lr': cfg.abs.w_lr},
                {'params': stylegan_noises, 'lr': cfg.abs.w_lr},
            ])
            is_rot_optim = False
            
        for batch_idx in range(num_batches):
            
            optimizer.zero_grad()
            cur_idxs = used_idxes[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            
            cur_search_cond = search_cond[cur_idxs]
            cur_random_rotations = random_rotations[cur_idxs]
            cur_random_translations = random_translations[cur_idxs]
            cur_fixed_xT = fixed_xT[cur_idxs]
            cur_stylegan_noises = [noise[cur_idxs] for noise in stylegan_noises]
            
            # add noise to latent to increase exploration 
            end_steps = config_steps[-1]["steps"]
            t = iteration / end_steps
            
            noise_strength = latent_std * cfg.abs.w_noise * max(0, 1 - t / cfg.abs.w_noise_ramp) ** 2
            cur_latent_in = latent_noise_stylegan(cur_search_cond, noise_strength)
            
            search_cond_lr = get_lr_stylegan(t, cfg.abs.w_lr)
            
            optimizer.param_groups[2]['lr'] = search_cond_lr
            
            # convert to SO3
            if iteration > 0 and cfg.abs.orthogonalize_rotations:
                cur_random_rotations = symmetric_orthogonalization(cur_random_rotations.view(-1, 9))
            
            cur_num_rotations = cur_search_cond.shape[0]

            input_images = generator.forward_latent_w(cur_latent_in, noises=cur_stylegan_noises)
            
            input_images = input_images.clamp(0, 1)
            input_images = input_images.unsqueeze(1).to(device)

            # Use OOD image's origin distances to match the scale of the target OOD image
            origin_distances = ood_data["origin_distances"][:1, ood_image_index:ood_image_index+1, ...].repeat(cur_num_rotations, 1, 1, 1, 1)
            input_images = torch.cat([input_images, origin_distances], dim=2)
            cur_focals_pixels = ood_data["focals_pixels"][:1, :cfg.data.input_images].repeat(cur_num_rotations, 1, 1)

            gaussian_splats_vis = gaussian_predictor(
                input_images, 
                train_data["view_to_world_transforms"][:1, :cfg.data.input_images, ...].repeat(cur_num_rotations, 1, 1, 1),
                train_data["source_cv2wT_quat"][:1, :cfg.data.input_images].repeat(cur_num_rotations, 1, 1), 
                cur_focals_pixels,
            )

            pred_images = []
            

            for i in range(cur_num_rotations):
                transformed_splats = {k: v[i] for k, v in gaussian_splats_vis.items()}
                # IMPORTANT: render the candidate using the same pivot convention as the OOD image generation.
                # The OOD image in this script is produced via centroid-aware transforms
                # (see `render_with_custom_camera_align` usage), while `render_with_custom_camera`
                # assumes pivot at (0,0,zgt). That mismatch can make the optimization inherently fail.
                pred_image = render_with_custom_camera_align(
                    transformed_splats,
                    background,
                    cfg,
                    ood_data["focals_pixels"][0, 0] if cur_focals_pixels is not None else None,
                    cur_random_rotations[i],
                    zgt,
                    device=device,
                    translation=cur_random_translations[i],
                    zgt_ood=zgt,
                    override_centroid=ood_centroid,
                )
                
                pred_images.append(pred_image)
            
            pred_images = torch.cat(pred_images, dim=0)
            ood_images = ood_image.expand(cur_num_rotations, -1, -1, -1)
            mse_losses = get_mse_loss(pred_images, ood_images)
            mse_losses = mse_losses.view(-1)
            
            losses = [
                lambda_mse * mse_losses[i]
                for i in range(cur_num_rotations)
            ]
            
            if lambda_lpips > 0:
                lpips_loss_values = lpips_fn(pred_images * 2 - 1, ood_images * 2 - 1)
                lpips_loss_values = lpips_loss_values.view(-1)
                losses = [losses[i] + lambda_lpips * lpips_loss_values[i] for i in range(cur_num_rotations)]
            else: 
                lpips_loss_values = torch.zeros(cur_num_rotations)

            noise_regulate_loss = noise_regularize_stylegan(cur_stylegan_noises)
            
            total_loss = sum(losses) + cfg.abs.w_noise_regularize * noise_regulate_loss 
            
            all_losses_and_idxs.extend([(cur_idxs[i], losses[i]) for i in range(len(cur_idxs))])
            
            total_loss.backward()
            random_translations.grad[:, 2] = 0  # Zero z gradient so only x and y are optimized
            optimizer.step() 
            
            noise_normalize_stylegan_(stylegan_noises)
            
            # decision_losses = torch.tensor([l_lpips.item() + 4 * l_mse.item() for l_mse, l_lpips in zip(mse_losses, lpips_loss_values)])
            # Use the same per-sample objective you're actually optimizing to rank "best"/top-k.
            # (Previously: lpips + 4*mse, which can select a different candidate than SGD is minimizing.)
            if lambda_lpips > 0:
                decision_losses = (lambda_mse * mse_losses + lambda_lpips * lpips_loss_values).detach()
            else:
                decision_losses = (lambda_mse * mse_losses).detach()
            cur_best_loss_idx = torch.argmin(decision_losses)

            if decision_losses[cur_best_loss_idx] < best_loss:
                best_pred_image = pred_images[cur_best_loss_idx].detach().cpu()
                best_input_image = input_images[cur_best_loss_idx].detach().cpu()
                best_loss = decision_losses[cur_best_loss_idx]
                best_rotation_matrix = cur_random_rotations[cur_best_loss_idx].detach().cpu()
                best_translation_matrix = cur_random_translations[cur_best_loss_idx].detach().cpu()
                best_search_cond = cur_search_cond[cur_best_loss_idx].detach().cpu()
                best_rotation_name = f"Rotation {cur_best_loss_idx + batch_idx * batch_size + 1}"
            
            # Update top K images list
            for i in range(cur_num_rotations):
                cur_index = i + batch_idx * batch_size
                if len(top_k_losses) < top_k:
                    top_k_losses.append(decision_losses[i].item())
                    top_k_pred_images.append(pred_images[i].detach().cpu())
                    top_k_input_images.append(input_images[i, 0].cpu().detach())
                    top_k_rotation_matrices.append(cur_random_rotations[i].detach().cpu())
                    top_k_translation_matrices.append(cur_random_translations[i].detach().cpu())
                    top_k_indexes.append(cur_index)
                else:
                    if cur_index in top_k_indexes: 
                        replace_idx = top_k_indexes.index(cur_index)
                    else: 
                        replace_idx = next((idx for idx, loss in enumerate(top_k_losses) if loss > decision_losses[i].item()), None)

                    if replace_idx is not None: 
                        top_k_losses[replace_idx] = decision_losses[i].item()
                        top_k_pred_images[replace_idx] = pred_images[i].detach().cpu()
                        top_k_input_images[replace_idx] = input_images[i, 0].cpu().detach()
                        top_k_rotation_matrices[replace_idx] = cur_random_rotations[i].detach().cpu()
                        top_k_translation_matrices[replace_idx] = cur_random_translations[i].detach().cpu()
                        top_k_indexes[replace_idx] = cur_index
            
            if log_iters % log_every == 0: 
                print("total_loss: ", total_loss.item(), "\nbest_loss", best_loss)
                mse_loss_sum = mse_losses.sum()
                print("mse_loss: ", mse_loss_sum.item())
                if lambda_lpips > 0:
                    lpips_loss_sum = lpips_loss_values.sum()
                    print("lpips_loss_sum: ", lpips_loss_sum.item())
                print("\n\n")
                
                num_to_log = min(cfg.abs.num_to_log, cur_num_rotations)
                
                log_top_pred_images = [(np.clip(img.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8) for img in top_k_pred_images]
                log_top_input_images = [(np.clip(img.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8) for img in top_k_input_images]

                log_pred_images = [pred_images[i].cpu().detach().numpy().transpose(1, 2, 0) for i in range(num_to_log)]
                log_pred_images = [(np.clip(img, 0, 1) * 255).astype(np.uint8) for img in log_pred_images]

                log_best_pred_image = (np.clip(best_pred_image.cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                log_ood_image = (np.clip(ood_image[0].cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)

                log_input_images = [input_images[i, 0].cpu().detach().numpy().transpose(1, 2, 0) for i in range(num_to_log)]
                log_input_images = [(np.clip(img, 0, 1) * 255).astype(np.uint8) for img in log_input_images]

                # Create grids for input and predicted images
                input_images_grid = vutils.make_grid(torch.tensor(np.array(log_input_images)).permute(0, 3, 1, 2), nrow=4)
                pred_images_grid = vutils.make_grid(torch.tensor(np.array(log_pred_images)).permute(0, 3, 1, 2), nrow=4)

                top_input_images_grid = vutils.make_grid(torch.tensor(np.array(log_top_input_images)).permute(0, 3, 1, 2), nrow=4)
                top_pred_images_grid = vutils.make_grid(torch.tensor(np.array(log_top_pred_images)).permute(0, 3, 1, 2), nrow=4)
                
                try:
                    # Extract RGB channels from best_input_image (remove origin_distances)
                    best_input_rgb = best_input_image.squeeze(0)[:3, :, :]  # [1, 4, H, W] -> [3, H, W]
                    
                    dict_to_log = {
                        "Original OOD Image": wandb.Image(log_ood_image, caption="Original OOD Image"),
                        "Input Images Grid": wandb.Image(input_images_grid, caption="Input Images Grid"),
                        "Best Predicted Image": wandb.Image(log_best_pred_image, caption=f"Best pred {best_rotation_name}"),
                        "Top Input Images Grid": wandb.Image(top_input_images_grid, caption="Top Input Images Grid"),
                        "Top Predicted Images Grid": wandb.Image(top_pred_images_grid, caption="Top Predicted Images Grid"),
                        "Predicted Images Grid": wandb.Image(pred_images_grid, caption="Predicted Images Grid"),
                        "best_input_image": wandb.Image(best_input_rgb, caption="Best Input Image"),
                        "best_loss": best_loss.item(), 
                        "mse_loss": mse_loss_sum.item(),
                        "lpips_loss": lpips_loss_sum.item() if lambda_lpips > 0 else -1,
                        "current_stage": step_idx+1, 
                    }
                    if lambda_lpips > 0:
                        dict_to_log["lpips_loss"] = lpips_loss_sum.item()   
                    
                    wandb.log(dict_to_log)
                    
                except Exception as e:
                    print(f"Warning: Wandb logging failed with error: {e}")
                    
            sorted_idxs_by_loss = [idx for idx, _ in sorted(all_losses_and_idxs, key=lambda x: x[1])]
            log_iters += 1
        iteration+= 1
        tqdm_iterator.update(1)


@hydra.main(version_base=None, config_path=_cfg.CONFIGS_DIR, config_name="abs_config")  
def main(cfg: DictConfig): 
    import pandas as pd
    global dataset, val_dataset
    
    quality = "good"
    split = "ood"
    reevaluate = True
    
    dataset_name = cfg.data.category
    log_path = os.path.join(_cfg.RUNS_ROOT, cfg.general.prefix)
    os.makedirs(log_path, exist_ok=True)  # Ensure log directory exists
    results_save_path = f"{log_path}/co3d_se3_results.csv"
    test_imgs_csv_path = cfg.general.test_imgs_csv_path 
    use_hq = cfg.general.use_hq
    assert test_imgs_csv_path is not None, "test_imgs_csv_path must be set in the config file"

    test_imgs_csv = pd.read_csv(test_imgs_csv_path)
    test_imgs_csv = test_imgs_csv[(test_imgs_csv["quality"] == quality) & (test_imgs_csv["split"] == split)]

    test_imgs_list = test_imgs_csv["path"].tolist()
    test_imgs_list = [img_path for img_path in test_imgs_list if os.path.exists(img_path)]
    parent_n = lambda path, n: path if n <=0 else parent_n(os.path.dirname(path), n - 1)

    test_ids_to_use = [os.path.basename(parent_n(img_path, 2)) for img_path in test_imgs_list]

    print("length of test_ids_to_use", len(test_ids_to_use))
    assert len(test_ids_to_use) > 0, "test_ids_to_use must be greater than 0"
    print("length of test_imgs_list", len(test_imgs_list))
    assert len(test_imgs_list) > 0, "test_imgs_list must be greater than 0"
    # Handle single example case
    if cfg.general.override_example_id is not None: 
        print("using custom override example id")
        train(cfg, cfg.general.override_example_id, test_idx=cfg.general.test_idx, log_path=log_path, dataset_name=dataset_name, prefix=cfg.general.prefix)
        return 
    
    processed_example_ids = set()
    
    # Check if results file exists and load processed example_ids if it does
    if os.path.exists(results_save_path):
        with open(results_save_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_example_ids.add(row["example_id"])
    
    # Load test dataset to get example IDs
    # Get list of example IDs from the dataset
    
    if cfg.general.maxsamples is not None and isinstance(cfg.general.maxsamples, int):
        print("Using maxsamples:", cfg.general.maxsamples)
        test_imgs_list = test_imgs_list[:cfg.general.maxsamples]
    
    size_per_split = len(test_imgs_list) // cfg.general.total_splits
    remainder = len(test_imgs_list) % cfg.general.total_splits
    cur_size = size_per_split
    if cfg.general.split == cfg.general.total_splits - 1: 
        cur_size += remainder
    
    cur_test_imgs_list = test_imgs_list[cfg.general.split * size_per_split: (cfg.general.split * size_per_split) + cur_size]
    assert cur_test_imgs_list, (
        f"shard {cfg.general.split}/{cfg.general.total_splits} of {len(test_imgs_list)} "
        "test images is empty -- lower general.total_splits (it silently no-ops otherwise)")
    
    lock_path = results_save_path + ".lock"


    # loading datasets 
    # will load dataset 1080p if test_img_path is 1080. you have to set use_hq yourself though. 
    dataset, val_dataset = load_dataset(dataset_name, cfg, use_hq = use_hq)
    
    for test_img_path in cur_test_imgs_list: 
        override_example_id = os.path.basename(parent_n(test_img_path, 2))
        ood_test_id, override_example_id = val_dataset.find_index_for_sequence_prefix(override_example_id.replace("hydrant_", ""))
        try: 
            if override_example_id in processed_example_ids:
                print(f"Skipping already processed example id: {override_example_id}")
                continue
                # raise Exception(f"Skipping already processed example id: {override_example_id}")
            
            print(f"Running optimization for example id: {override_example_id}")
            start_time = time.time()
            results = train(cfg, override_example_id, test_idx=0, log_path=log_path, dataset_name=dataset_name, prefix=cfg.general.prefix, test_img_path=test_img_path)
            results["example_id"] = override_example_id
            
            with FileLock(lock_path):
                # Check if file needs header before opening
                needs_header = not os.path.exists(results_save_path) or os.path.getsize(results_save_path) == 0
                
                with open(results_save_path, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=results.keys())
                    
                    if needs_header:
                        writer.writeheader()
                    
                    writer.writerow(results)
            
            print(f"Result for {override_example_id} appended to {results_save_path}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Finished example id {override_example_id} in {elapsed_time:.2f} seconds")
        except Exception as e: 
            print(f"Error for example id {override_example_id}: {e}")
            raise e
            continue
        
if __name__ == "__main__": 
    main()

