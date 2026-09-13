# imports 
# -------------------------------------------------------------------------------------- # 
from infer3d import config as _cfg
from math import e
import math
import sys 
from filelock import FileLock
import time 
import json 
import csv 
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
# add diffae to sys.path
DIFFAE_ROOT = _cfg.DIFFAE_ROOT
sys.path.append(_cfg.DIFFAE_ROOT)
from templates import *
import hydra

# -------------------------------------------------------------------------------------- # 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = None 
    
class DiffAEGenerator(nn.Module): 
    def __init__(self, conf, checkpoint_path=None):
        super().__init__()
        self.conf = conf
        model = LitModel(conf)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        else: 
            state = torch.load(f'{DIFFAE_ROOT}/checkpoints/{conf.name}/last.ckpt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'], strict=False)
        model.ema_model.eval()
        model.ema_model.to(device)
        # model.model.eval()
        # model.model.to(device)
        self.model = model

    def forward(self, latent_code, cond=None, T=12):
        if cond is None: 
            cond = torch.randn(1, 512).to(device)
        xT = latent_code
        return self.model.render(xT, cond, T=T)
    
    def encode(self, img):
        return self.model.encode(img)

    def encode_stochastic(self, img, cond, T=250):
        return self.model.encode_stochastic(img, cond, T=T)



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
def eval_baseline(ood_data, ood_image_index, gaussian_predictor, cfg, background, save_video_path=None, use_ood_image=None, ood_rotation=None, zgt=None, ood_centroid=None, eval_indist=False):
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
    if use_ood_image is not None and ood_rotation is not None and not eval_indist:
        # The OOD image was rendered from rotated splats, so predicted splats are in rotated frame
        # Apply inverse rotation to bring them back to original coordinate frame
        inverse_rotation = ood_rotation.T  # Transpose for inverse rotation
        pred_splats = render_with_custom_camera(
            pred_splats, background, cfg, ood_focals_pixels,
            inverse_rotation, zgt, device=device, return_splats=True, translation=None,
            zgt_ood=zgt,
            # override_centroid=ood_centroid,
        )
    
    # Evaluate on the same ood_data views (they're already in the right coordinate frame)
    renders, gt_images, scores = create_loop_and_eval(pred_splats, ood_data, "test", cfg, 
                                                        ood_data["focals_pixels"], save_loop_path=save_video_path, save_gt=True, use_absolute_poses=False)
    
    return renders, gt_images, scores

@torch.no_grad()
def eval_pred(best_input_image, best_rotation_matrix, best_translation_matrix, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=None, ood_rotation=None, ood_centroid=None, eval_indist=False): 
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
        ood_data["view_to_world_transforms"][:1, 0:cfg.data.input_images], 
        ood_data["source_cv2wT_quat"][:1, 0:cfg.data.input_images], 
        focals_pixels_pred   
    )
    pred_splats = {k: v[0] for k, v in pred_splats.items()}
    
    # Apply the optimized rotation/translation
    transformed_splats = render_with_custom_camera_align(pred_splats, 
                                                   background, 
                                                   cfg, 
                                                   ood_focals_pixels,
                                                   best_rotation_matrix.to(device),
                                                   zgt, device = device, return_splats=True, translation=best_translation_matrix.to(device), zgt_ood=zgt)
    
    # If OOD image was from random rotation, undo that rotation to get back to original frame
    if ood_rotation is not None and not eval_indist:
        inverse_rotation = ood_rotation.T  # Transpose for inverse rotation
        transformed_splats = render_with_custom_camera(
            transformed_splats, background, cfg, ood_focals_pixels,
            inverse_rotation, zgt, device=device, return_splats=True, translation=None,
            zgt_ood=zgt,
            # override_centroid=ood_centroid,
        )
        
    # we also invert going from ood_image_index to 0 (which is the first image in the sequence) and has identity camera 
    # V2W is transposed in the dataset, so V2W[3, :3] is the translation and V2W[:3, :3] is R.T
    # We want to apply V2W.T to column vectors.
    # render_with_custom_camera expects `rotation` such that its transpose is applied.
    # So we pass rotation = V2W[:3, :3] and translation = V2W[3, :3]
    V2W = ood_data["view_to_world_transforms"][0, ood_image_index]
    second_inverse_rotation = V2W[:3, :3]
    second_inverse_translation = V2W[3, :3]
    
    transformed_splats = render_with_custom_camera(
        transformed_splats, background, cfg, ood_focals_pixels,
        second_inverse_rotation, zgt, device=device, return_splats=True, 
        translation=second_inverse_translation,
        zgt_ood=zgt,
        rotate_at_origin=False
    )
    
    # Evaluate on ood_data views directly (they're already in the right coordinate frame)
    renders, gt_images, scores = create_loop_and_eval(transformed_splats, ood_data, "test", cfg,
                                                        ood_data["focals_pixels"], save_loop_path=save_video_path, save_gt=True, use_absolute_poses=False)
    
    return renders, gt_images, scores


def load_dataset(dataset_name, cfg, use_hq=False):
    if dataset_name == "hydrants" or dataset_name == "vases": 
        print(f"loading co3d {dataset_name}")
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
    conf_generator = None 
    if cfg.data.category == "hydrants":
        conf_generator = co3d_hydrants_autoenc_128()
    elif cfg.data.category == "vases":
        conf_generator = co3d_vases_autoenc_128()
    assert conf_generator is not None, f"conf_generator is not set because dataset{cfg.data.category} is not supported"
    generator = DiffAEGenerator(conf_generator)
    generator.eval().to(device)
    return gaussian_predictor, generator


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
lambda_depth = 0
batch_size = 32
num_batches = 0
rot_optim = True 
is_rot_optim = True 

is_debugging = False

wandb_run = None
val_dataset = None
dataset = None

def train(cur_cfg, override_example_id = None, test_idx = 0, log_path = "checkpoints-co3d-se3", val_view = None, dataset_name = None, prefix = None, test_img_path = None, run_indist = False): 
    # global variables for easier access 
    global used_idxes, sorted_idxs_by_loss, best_loss
    global log_iters, log_every, lambda_lpips, lambda_mse, lambda_depth
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
    # Allow force_seed override for sensitivity-to-initialization experiments.
    # When force_seed >= 0, all objects in the run share the same initial proposals,
    # so variance across seeds isolates initialization sensitivity.
    _force_seed = getattr(cfg.general, 'force_seed', -1)
    if _force_seed is not None and int(_force_seed) >= 0:
        cfg.abs.random_seed = int(_force_seed)
    else:
        cfg.abs.random_seed = abs(hash(override_example_id)) % 2**32
    g = torch.Generator()
    np.random.seed(cfg.abs.random_seed)
    torch.manual_seed(cfg.abs.random_seed)
    torch.cuda.manual_seed(cfg.abs.random_seed)
    random.seed(cfg.abs.random_seed)
    g.manual_seed(cfg.abs.random_seed)

    
    
    dataloader = DataLoader(dataset,
                            batch_size=cfg.abs.num_latents,
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
        if run_indist:
            wandb_run_name += "indist-"
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
        raise ValueError("OOD data found in checkpoint")
        ood_data = test_img_checkpoint["ood_data"]
        print("FOUND OOD DATA IN CHECKPOINT FOR OVERRIDE EXAMPLE ID: ", override_example_id)
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    if len(ood_data["focals_pixels"].shape) == 2:
        ood_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in ood_data.items()}
    # let's load the checkpoint from the image_path 
    ood_zgt = torch.tensor(test_img_checkpoint["zgt"])
    ood_image_index = test_img_checkpoint["ood_image_index"]
    ood_depth_image = test_img_checkpoint["depth_image"].to(device)
    ood_depth_image = F.interpolate(ood_depth_image, size=(cfg.data.training_resolution, cfg.data.training_resolution), mode="bilinear", align_corners=False)
    ood_indist_render_image = test_img_checkpoint["indist_render_image"].to(device)
    ood_random_rotation = test_img_checkpoint["ood_random_rotation"].to(device)
    ood_focals_pixels = ood_data["focals_pixels"][:, ood_image_index:ood_image_index+1]
    ood_centroid = test_img_checkpoint["centroid"].to(device)
    
    
    if run_indist: 
        ood_image = ood_data["gt_images"][:1, ood_image_index].to(device)
        ood_image = F.interpolate(ood_image, size=(cfg.data.training_resolution, cfg.data.training_resolution), mode="bilinear", align_corners=False)
        ood_random_rotation = torch.eye(3).to(device)
        ood_depth_image = test_img_checkpoint["indist_depth_image"].to(device)
        ood_depth_image = F.interpolate(ood_depth_image, size=(cfg.data.training_resolution, cfg.data.training_resolution), mode="bilinear", align_corners=False)

    else: 
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
        
        ood_render_image, ood_render_depth_image = render_with_custom_camera_align(
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
            return_depth=True,
        )


        # send to wandb 
        wandb.log({
            "indist_image": wandb.Image(torch.clip(indist_image.squeeze(), 0, 1), caption="Indist Image"),
            "indist_render_image": wandb.Image(torch.clip(indist_render_image.squeeze()[:3], 0, 1), caption="Indist Render Image"),
            "indist_to_ood_image": wandb.Image(torch.clip(ood_render_image.squeeze(), 0, 1), caption="Indist to OOD Image"),
            "indist_to_ood_depth_image": wandb.Image(torch.clip(ood_render_depth_image.squeeze(0), 0, 1), caption="Indist to OOD Depth Image"),
            "ood_indist_render_image": wandb.Image(torch.clip(ood_indist_render_image.squeeze()[:3], 0, 1), caption="OOD Indist Render Image"),
            "ood_depth_image": wandb.Image(torch.clip(ood_depth_image.squeeze(0), 0, 1), caption="OOD Depth Image"),
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

    # Total latents: train-image latents + 1 OOD-encoded latent
    total_latents = cfg.abs.num_latents + 1
    # Total rotations: random rotations + 1 identity rotation
    total_rotations = num_rotations + 1

    # Total candidates: train latents with all rotations + OOD latent with identity only
    total_candidates = total_rotations * cfg.abs.num_latents + 1

    fixed_xT = torch.randn(total_candidates, 3, 128, 128, device=device)
    generator_transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    train_data_iter = iter(dataloader)
    train_data = next(train_data_iter)
    
    print("Using zgt: ", zgt)

    # Encode first training view to get DiffAE latent proposals (like side_top_so3_search_steps_a100.py)
    train_image = generator_transform(train_data['gt_images'][:, 0]).to(device)
    assert train_image.shape[1] == 3, f"Train image should have 3 channels cf. {train_image.shape}"
    
    with torch.no_grad():
        init_search_cond_train = generator.encode(train_image)  # [num_latents, 512]
    assert init_search_cond_train.shape[0] == cfg.abs.num_latents, f"init_search_cond_train should have shape {cfg.abs.num_latents} cf. {init_search_cond_train.shape}"
    
    # Encode the OOD image to get an extra latent proposal
    ood_image_for_encoder = generator_transform(ood_image).to(device)  # ood_image is [1, 3, H, W]
    with torch.no_grad():
        init_search_cond_ood = generator.encode(ood_image_for_encoder)  # [1, 512]
    assert init_search_cond_ood.shape[0] == 1, f"init_search_cond_ood should have shape 1 cf. {init_search_cond_ood.shape}"
    
    # Build partial Cartesian product:
    # - Train latents with ALL rotations (random + identity)
    # - OOD latent with ONLY identity rotation
    train_latents_repeated = [init_search_cond_train.clone().detach().requires_grad_(True)
                              for _ in range(total_rotations)]
    train_latents_all_rots = torch.cat(train_latents_repeated, dim=0)  # [total_rotations * num_latents, 512]

    # OOD latent with ONLY identity rotation
    ood_latent_identity = init_search_cond_ood.clone().detach().requires_grad_(True)  # [1, 512]

    # Combine them
    search_cond = nn.Parameter(torch.cat([train_latents_all_rots, ood_latent_identity], dim=0))
    # search_cond = nn.Parameter(torch.cat([train_latents_all_rots, train_latents_all_rots[-1:]], dim=0))
    # Shape: [total_rotations * num_latents + 1, 512] = [total_candidates, 512]
    
    # w_avg for regularization (mean of only train latents)
    w_avg = init_search_cond_train.mean(dim=0).detach()

    assert len(search_cond.shape) == 2 and search_cond.shape[0] == total_candidates, f"search_cond should have shape {total_candidates, 512} cf. {search_cond.shape}"
    
    # Always built: the stage schedule below can raise lambda_lpips even when the
    # configured value is 0, so a conditionally-created lpips_fn crashes mid-run.
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
    print("Using LPIPS loss with lambda_lpips: ", cfg.abs.lambda_lpips)

    num_iterations = cfg.abs.num_iterations

    train_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in train_data.items()}

    best_loss = float('inf')
    best_pred_image = None
    best_pred_depth_image = None
    best_input_image = None
    best_search_cond = None 
    best_rotation_matrix = None
    best_translation_matrix = None
    best_rotation_name = ""

    top_k = min(10, total_candidates)
    assert top_k <= total_candidates, f"top_k should be less than or equal to total_candidates cf. {top_k} and {total_candidates}"
    top_k_pred_images = []
    top_k_pred_depth_images = []
    top_k_input_images = []
    top_k_losses = []
    top_k_rotation_matrices = []
    top_k_translation_matrices = []
    top_k_indexes = []

    # Random rotations for cfg.abs.num_rotations, plus identity rotation appended
    random_rotations_base = get_random_cameras(num_rotations, zgt, device=device)  # [num_rotations, 3, 3]
    identity_rotation = torch.eye(3, device=device).unsqueeze(0)  # [1, 3, 3]
    all_rotations = torch.cat([random_rotations_base, identity_rotation], dim=0)  # [total_rotations, 3, 3]

    # Build partial Cartesian product for rotations:
    # - Expand ALL rotations across TRAIN latents only
    train_rotations = torch.stack([all_rotations.clone() for _ in range(cfg.abs.num_latents)], dim=1)
    train_rotations = train_rotations.reshape(-1, 3, 3)  # [total_rotations * num_latents, 3, 3]

    # Identity rotation for OOD latent
    ood_rotation = identity_rotation.clone()  # [1, 3, 3]

    # Combine them
    random_rotations = torch.cat([train_rotations, ood_rotation], dim=0)  # [total_candidates, 3, 3]
    random_rotations.requires_grad_(True)
    
    # Translations: zeros for all candidates
    random_translations = torch.zeros(total_candidates, 3, device=device)
    random_translations.requires_grad_(True)
    
    # Define pinned candidate index: (OOD latent, identity rotation)
    # OOD latent with identity rotation is now the LAST index in the flattened array
    pinned_idx = total_rotations * cfg.abs.num_latents  # Last index
    assert pinned_idx < total_candidates, f"pinned_idx {pinned_idx} out of range for {total_candidates} candidates"
    print(f"Pinned candidate index (OOD latent + identity rotation): {pinned_idx}")
    
    optimizer = optim.Adam([ 
        {'params': random_rotations, 'lr': cfg.abs.lr_rotations},
        {'params': random_translations, 'lr': cfg.abs.lr_rotations},
        {'params': search_cond, 'lr': cfg.abs.w_lr},
    ])

    log_iters = 0
    prev_best_loss = None

    import time
    used_idxes = list(range(total_candidates))
    sorted_idxs_by_loss = []
    
    # Variables to store pinned candidate's images for logging
    pinned_input_image = None
    pinned_pred_image = None
    pinned_pred_depth_image = None
    pinned_decision_loss = None

    # Evaluate baseline once at the start
    print("Evaluating baseline-in-dist (splatter image only with in-dist image)...")
    baseline_indist_video_path = f"{experiment_file_path}/baseline_indist_video.mp4"
    with torch.no_grad():
        baseline_indist_renders, baseline_indist_gt_images, baseline_indist_scores = eval_baseline(
            ood_data, ood_image_index, 
            gaussian_predictor, cfg, background, save_video_path=baseline_indist_video_path, 
            use_ood_image=None, ood_rotation=None, zgt=zgt, ood_centroid=ood_centroid, eval_indist=run_indist
        )
    print(f"Baseline-in-dist scores: {baseline_indist_scores}")
    
    print("Evaluating baseline-ood (splatter image only with OOD image)...")
    baseline_ood_video_path = f"{experiment_file_path}/baseline_ood_video.mp4"
    with torch.no_grad():
        baseline_ood_renders, baseline_ood_gt_images, baseline_ood_scores = eval_baseline(
            ood_data, ood_image_index, 
            gaussian_predictor, cfg, background, save_video_path=baseline_ood_video_path, 
            use_ood_image=ood_image, ood_rotation=ood_random_rotation, zgt=zgt, ood_centroid=ood_centroid, eval_indist=run_indist
        )
    
    print("DONE EVALUATING BASELINES")
    
    # Helper to deduplicate while preserving order
    def dedup_keep_order(lst):
        seen = set()
        result = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result
    
    # Initialize timing and configuration
    # Force-keep pinned_idx in used_idxes for iterations < 302, then let it compete naturally
    config_steps = [
        {
            "steps": 0,
            "settings": {
                "used_idxes": list(range(total_candidates)),
                "sorted_idxs_by_loss": [],
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 250,
                "lambda_lpips": cfg.abs.lambda_lpips,
                "lambda_mse": cfg.abs.lambda_mse,
                "lambda_depth": cfg.abs.lambda_depth,
                "batch_size": batch_size,
                "num_batches": lambda: math.ceil(len(used_idxes) / batch_size),
            }
        },
        {
            # After the initial selection pass (iteration 0), restrict optimization
            # to the best `batch_size` candidates, but force-keep pinned_idx.
            "steps": 1,
            "settings": {
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 250,
                "lambda_lpips": cfg.abs.lambda_lpips,
                "lambda_mse": cfg.abs.lambda_mse,
                "batch_size": batch_size,
                # Force-keep pinned_idx in used_idxes
                "used_idxes": lambda: dedup_keep_order([pinned_idx] + sorted_idxs_by_loss[:32]),
                "num_batches": lambda: math.ceil(len(used_idxes) / batch_size),
            }
        },
        {
            "steps": 122,
            "settings": {
                "batch_size": batch_size,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 250,
                # Force-keep pinned_idx in used_idxes
                "used_idxes": lambda: dedup_keep_order([pinned_idx] + (sorted_idxs_by_loss[:10] if batch_size < 10 else sorted_idxs_by_loss[:batch_size])),
                "num_batches": lambda: math.ceil(len(used_idxes) / batch_size),
            }
        },
        {
            # At steps 302, stop force-keeping pinned_idx - let it compete naturally
            "steps": 302,
            "settings": {
                "batch_size": batch_size,
                # 2 as in the paper runs, but respect an explicit lambda_lpips=0
                # (Table 8's L_MSE-only ablation) instead of silently overriding it.
                "lambda_lpips": 0 if cfg.abs.lambda_lpips == 0 else 2,
                "lambda_mse": 10,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: math.ceil(len(used_idxes) / batch_size),
            },
        },
        {
            "steps": 732,
            "settings": {
                "batch_size": batch_size,
                "lambda_mse": 2,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: math.ceil(len(used_idxes) / batch_size),
            },
        },
        {
            "steps": 782,
            "settings": {
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "batch_size": batch_size,
                "num_batches": lambda: math.ceil(len(used_idxes) / batch_size),
            },
        },
        {"steps": 783, "settings": "stop"}
    ]

    # Loop control
    step_idx = 0
    iteration = 0
    start_time = time.time()
    loss_trajectory = []  # {iteration, elapsed_time_s, best_loss} logged every iter for OOD-at-step-T analysis
    torch.cuda.reset_peak_memory_stats()

    tqdm_iterator = tqdm(total=num_iterations, desc="iterations")

    # ---- Step-0 OOD detection timing ----
    # Time just: encode(ood_image) -> decode -> 3DGS -> render -> loss
    # No optimization, no random proposals. This is the "feedforward only" OOD detection cost.
    with torch.no_grad():
        torch.cuda.synchronize()
        _t0_ood = time.time()
        _ood_cond_fresh = generator.encode(ood_image_for_encoder)
        _ood_decoded = generator(fixed_xT[pinned_idx:pinned_idx+1], _ood_cond_fresh, T=12).clamp(0, 1).unsqueeze(1)
        _origin_dists = ood_data["origin_distances"][:1, ood_image_index:ood_image_index+1, ...].repeat(1, 1, 1, 1, 1)
        _ood_input_with_dist = torch.cat([_ood_decoded, _origin_dists], dim=2)
        _focals_p = ood_data["focals_pixels"][:1, ood_image_index:ood_image_index+cfg.data.input_images]
        _splats_p = gaussian_predictor(
            _ood_input_with_dist,
            train_data["view_to_world_transforms"][:1, :cfg.data.input_images, ...],
            train_data["source_cv2wT_quat"][:1, :cfg.data.input_images],
            _focals_p,
        )
        _pred_p, _pred_depth_p = render_with_custom_camera_align(
            {k: v[0] for k, v in _splats_p.items()},
            background, cfg, _focals_p[0, 0],
            torch.eye(3, device=device),
            zgt, device=device,
            translation=torch.zeros(3, device=device),
            zgt_ood=zgt,
            return_depth=True,
        )
        _mse_p = get_mse_loss(_pred_p, ood_image).item()
        _depth_p = get_mse_loss(_pred_depth_p, ood_depth_image).item()
        _lpips_p = lpips_fn(_pred_p * 2 - 1, ood_image * 2 - 1).item() if cfg.abs.lambda_lpips > 0 else 0.0
        torch.cuda.synchronize()
        step0_pinned_eval_time_ms = (time.time() - _t0_ood) * 1000
        step0_pinned_loss = cfg.abs.lambda_mse * _mse_p + cfg.abs.lambda_lpips * _lpips_p + cfg.abs.lambda_depth * _depth_p
    print(f"Step-0 OOD detection: time={step0_pinned_eval_time_ms:.1f}ms, loss={step0_pinned_loss:.4f}")

    optim_start_time = None  # set at start of iteration 1 (first real gradient step)

    while step_idx < len(config_steps):
        all_losses_and_idxs = []
        
        elapsed_time = time.time() - start_time

        if iteration >= config_steps[step_idx]["steps"]:
            assert step_idx == 0 or len(sorted_idxs_by_loss) > 0, f"sorted_idxs_by_loss {sorted_idxs_by_loss} should not be empty"     
            settings = config_steps[step_idx]["settings"]
            if settings == "stop":
                # Capture optimization end time before any evaluation or logging
                optim_end_time = time.time()
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
                    _, _, cur_scores = eval_pred(eval_input_image, eval_rotation_matrix, eval_translation_matrix, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=ours_video_path, ood_rotation=ood_random_rotation, ood_centroid=ood_centroid, eval_indist=run_indist)
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
                known_best_renders, known_best_gt_images, known_best_scores = eval_pred(best_input_image, best_rotation_matrix, best_translation_matrix, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=ours_best_video_path, ood_rotation=ood_random_rotation, ood_centroid=ood_centroid, eval_indist=run_indist)
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
                # exit() # TODO: remove this
                # Save loss trajectory + VRAM stats for compute-cost rebuttal experiments
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
                trajectory_data = {
                    "example_id": override_example_id,
                    "run_indist": run_indist,
                    "batch_size": batch_size,
                    "num_latents": cfg.abs.num_latents,
                    "peak_vram_gb": peak_vram_gb,
                    "total_time_s": time.time() - start_time,
                    "total_optim_time_s": (optim_end_time - optim_start_time) if optim_start_time is not None else None,
                    "step0_pinned_eval_time_ms": step0_pinned_eval_time_ms,
                    "step0_pinned_loss": step0_pinned_loss,
                    "trajectory": loss_trajectory,
                }
                with open(os.path.join(experiment_file_path, "loss_trajectory.json"), "w") as f:
                    json.dump(trajectory_data, f)

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
                    "ood_image_index": ood_image_index,
                    "ood_image": ood_image,
                    "ood_random_rotation": ood_random_rotation,  # ADD THIS!
                    "ood_centroid": ood_centroid,
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
            assert math.ceil(len(used_idxes) / batch_size) == num_batches, f"num_batches {num_batches} should be equal to {math.ceil(len(used_idxes) / batch_size)}"
            
        if is_rot_optim and not rot_optim: 
            optimizer = optim.Adam([ 
                {'params': search_cond, 'lr': cfg.abs.w_lr},
            ])
            is_rot_optim = False
            
        if iteration == 1 and optim_start_time is None:
            optim_start_time = time.time()

        for batch_idx in range(num_batches):

            optimizer.zero_grad()
            cur_idxs = used_idxes[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            if len(cur_idxs) == 0:
                continue
            
            cur_search_cond = search_cond[cur_idxs]
            cur_random_rotations = random_rotations[cur_idxs]
            cur_random_translations = random_translations[cur_idxs]
            cur_fixed_xT = fixed_xT[cur_idxs]
            
            # Iteration 0 is a selection-only pass: compute per-candidate loss for ranking,
            # but do NOT take any gradient steps (even though evaluation is mini-batched).
            selection_only = (iteration == 0)
            cur_latent_in = cur_search_cond
            
            # convert to SO3
            if iteration > 0 and cfg.abs.orthogonalize_rotations:
                cur_random_rotations = symmetric_orthogonalization(cur_random_rotations.view(-1, 9))
                
            # Force pinned candidate to strictly remain at identity rotation and zero translation
            pinned_local_idx = cur_idxs.index(pinned_idx) if pinned_idx in cur_idxs else None
            if pinned_local_idx is not None:
                cur_random_rotations[pinned_local_idx] = torch.eye(3, device=device)
                cur_random_translations[pinned_local_idx] = torch.zeros(3, device=device)

            cur_num_rotations = cur_search_cond.shape[0]

            if selection_only:
                # print("hajaba mankuna")
                with torch.no_grad():
                    # DiffAE forward: T=12 is the default DDIM steps (tune as needed)
                    input_images = generator(cur_fixed_xT, cur_latent_in, T=12)
                    input_images = input_images.clamp(0, 1)
                    input_images = input_images.unsqueeze(1).to(device)

                    # Use OOD image's origin distances to match the scale of the target OOD image
                    origin_distances = ood_data["origin_distances"][:1, ood_image_index:ood_image_index+1, ...].repeat(cur_num_rotations, 1, 1, 1, 1)
                    input_images = torch.cat([input_images, origin_distances], dim=2)
                    cur_focals_pixels = ood_data["focals_pixels"][:1, ood_image_index:ood_image_index+cfg.data.input_images].repeat(cur_num_rotations, 1, 1)
                    
                    gaussian_splats_vis = gaussian_predictor(
                        input_images, 
                        train_data["view_to_world_transforms"][:1, :cfg.data.input_images, ...].repeat(cur_num_rotations, 1, 1, 1),
                        train_data["source_cv2wT_quat"][:1, :cfg.data.input_images].repeat(cur_num_rotations, 1, 1), 
                        cur_focals_pixels,
                    )
            else:
                # DiffAE forward: T=12 is the default DDIM steps (tune as needed)
                input_images = generator(cur_fixed_xT, cur_latent_in, T=12)
                # print("hakuna matata")
                input_images = input_images.clamp(0, 1)
                input_images = input_images.unsqueeze(1).to(device)

                # Use OOD image's origin distances to match the scale of the target OOD image
                origin_distances = ood_data["origin_distances"][:1, ood_image_index:ood_image_index+1, ...].repeat(cur_num_rotations, 1, 1, 1, 1)
                input_images = torch.cat([input_images, origin_distances], dim=2)
                cur_focals_pixels = ood_data["focals_pixels"][:1, ood_image_index:ood_image_index+cfg.data.input_images].repeat(cur_num_rotations, 1, 1)

                gaussian_splats_vis = gaussian_predictor(
                    input_images, 
                    train_data["view_to_world_transforms"][:1, :cfg.data.input_images, ...].repeat(cur_num_rotations, 1, 1, 1),
                    train_data["source_cv2wT_quat"][:1, :cfg.data.input_images].repeat(cur_num_rotations, 1, 1), 
                    cur_focals_pixels,
                )

            pred_images = []
            pred_depth_images = []
            

            for i in range(cur_num_rotations):
                transformed_splats = {k: v[i] for k, v in gaussian_splats_vis.items()}
                # IMPORTANT: render the candidate using the same pivot convention as the OOD image generation.
                # The OOD image in this script is produced via centroid-aware transforms
                # (see `render_with_custom_camera_align` usage), while `render_with_custom_camera`
                # assumes pivot at (0,0,zgt). That mismatch can make the optimization inherently fail.
                pred_image, pred_depth_image = render_with_custom_camera_align(
                    transformed_splats,
                    background,
                    cfg,
                    # ood_data["focals_pixels"][0, 0] if cur_focals_pixels is not None else None,
                    cur_focals_pixels[0,0],
                    cur_random_rotations[i],
                    zgt,
                    device=device,
                    translation=cur_random_translations[i],
                    zgt_ood=zgt,
                    # override_centroid=ood_centroid,
                    return_depth=True,
                )
                
                pred_images.append(pred_image)
                pred_depth_images.append(pred_depth_image)
            pred_images = torch.cat(pred_images, dim=0)
            pred_depth_images = torch.cat(pred_depth_images, dim=0)
            ood_images = ood_image.expand(cur_num_rotations, -1, -1, -1)
            ood_depth_images = ood_depth_image.expand(cur_num_rotations, -1, -1, -1)
            depth_mse_losses = get_mse_loss(pred_depth_images, ood_depth_images)
            depth_mse_losses = depth_mse_losses.view(-1)

            mse_losses = get_mse_loss(pred_images, ood_images)
            mse_losses = mse_losses.view(-1)

            losses = [
                lambda_mse * mse_losses[i] + lambda_depth * depth_mse_losses[i]
                for i in range(cur_num_rotations)
            ]
            
            if lambda_lpips > 0:
                lpips_loss_values = lpips_fn(pred_images * 2 - 1, ood_images * 2 - 1)
                lpips_loss_values = lpips_loss_values.view(-1)
                losses = [losses[i] + lambda_lpips * lpips_loss_values[i] for i in range(cur_num_rotations)]
            else: 
                lpips_loss_values = torch.zeros(cur_num_rotations)

            all_losses_and_idxs.extend([(cur_idxs[i], losses[i]) for i in range(len(cur_idxs))])

            # Per-candidate w-regularization so ranking/top-k uses the same objective as optimization.
            # Shape: [cur_num_rotations]
            if selection_only:
                # No gradients in iteration 0 (selection pass).
                w_reg_losses = cfg.abs.w_reg * (cur_search_cond.detach() - w_avg).norm(dim=1)
            else:
                w_reg_losses = cfg.abs.w_reg * (cur_search_cond - w_avg).norm(dim=1)

            if not selection_only:
                w_reg_loss = w_reg_losses.sum()
                total_loss = sum(losses) + w_reg_loss

                total_loss.backward()
                # Note: previously z-grad was disabled; keep current behavior (commented) unless explicitly changed.
                # random_translations.grad[:, 2] = 0  # Zero z gradient so only x and y are optimized
                
                # Prevent the pinned latent from being optimized by Adam
                # if pinned_idx in cur_idxs and search_cond.grad is not None:
                #     search_cond.grad[pinned_idx] = 0.0
                    
                optimizer.step()
            
            # decision_losses = torch.tensor([l_lpips.item() + 4 * l_mse.item() for l_mse, l_lpips in zip(mse_losses, lpips_loss_values)])
            # Use the same per-sample objective you're actually optimizing to rank "best"/top-k.
            # (Previously: lpips + 4*mse, which can select a different candidate than SGD is minimizing.)
            if lambda_lpips > 0:
                decision_losses = (lambda_mse * mse_losses + lambda_lpips * lpips_loss_values + lambda_depth * depth_mse_losses + cfg.abs.w_reg* 0 * w_reg_losses.detach()).detach()
            else:
                decision_losses = (lambda_mse * mse_losses + lambda_depth * depth_mse_losses + cfg.abs.w_reg* 0 * w_reg_losses.detach()).detach()
            cur_best_loss_idx = torch.argmin(decision_losses)

            # Capture pinned candidate's images if it's in this batch
            if pinned_idx in cur_idxs:
                pinned_local_idx = cur_idxs.index(pinned_idx)
                pinned_input_image = input_images[pinned_local_idx, 0].detach().cpu()  # [4, H, W] (RGB + origin_distances)
                pinned_pred_image = pred_images[pinned_local_idx].detach().cpu()
                pinned_pred_depth_image = pred_depth_images[pinned_local_idx].detach().cpu()
                pinned_decision_loss = decision_losses[pinned_local_idx].item()

            if decision_losses[cur_best_loss_idx] < best_loss:
                best_pred_image = pred_images[cur_best_loss_idx].detach().cpu()
                best_pred_depth_image = pred_depth_images[cur_best_loss_idx].detach().cpu()
                best_input_image = input_images[cur_best_loss_idx].detach().cpu()
                best_loss = decision_losses[cur_best_loss_idx]
                best_rotation_matrix = cur_random_rotations[cur_best_loss_idx].detach().cpu()
                best_translation_matrix = cur_random_translations[cur_best_loss_idx].detach().cpu()
                best_search_cond = cur_search_cond[cur_best_loss_idx].detach().cpu()
                best_rotation_name = f"Rotation {cur_best_loss_idx + batch_idx * batch_size + 1}"
            
            # Update top K images list
            for i in range(cur_num_rotations):
                cur_index = cur_idxs[i] # FIX: Use global candidate index, not loop counter!
                if len(top_k_losses) < top_k:
                    top_k_losses.append(decision_losses[i].item())
                    top_k_pred_images.append(pred_images[i].detach().cpu())
                    top_k_pred_depth_images.append(pred_depth_images[i].detach().cpu())
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
                        top_k_pred_depth_images[replace_idx] = pred_depth_images[i].detach().cpu()
                        top_k_input_images[replace_idx] = input_images[i, 0].cpu().detach()
                        top_k_rotation_matrices[replace_idx] = cur_random_rotations[i].detach().cpu()
                        top_k_translation_matrices[replace_idx] = cur_random_translations[i].detach().cpu()
                        top_k_indexes[replace_idx] = cur_index
            
            if log_iters > 0 and log_iters % log_every == 0:
                if selection_only:
                    print("selection_only(iteration 0): no optimizer.step()\n", "best_loss", best_loss)
                else:
                    print("total_loss: ", total_loss.item(), "\nbest_loss", best_loss)
                mse_loss_sum = mse_losses.sum()
                print("mse_loss: ", mse_loss_sum.item())
                if lambda_lpips > 0:
                    lpips_loss_sum = lpips_loss_values.sum()
                    print("lpips_loss_sum: ", lpips_loss_sum.item())
                print("\n\n")
                
                num_to_log = min(cfg.abs.num_to_log, cur_num_rotations)
                
                log_top_pred_images = [(np.clip(img.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8) for img in top_k_pred_images]
                log_top_pred_depth_images = [(np.clip(img.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8) for img in top_k_pred_depth_images]
                log_top_input_images = [(np.clip(img.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8) for img in top_k_input_images]

                log_pred_images = [pred_images[i].cpu().detach().numpy().transpose(1, 2, 0) for i in range(num_to_log)]
                log_pred_images = [(np.clip(img, 0, 1) * 255).astype(np.uint8) for img in log_pred_images]
                log_pred_depth_images = [pred_depth_images[i].cpu().detach().numpy().transpose(1, 2, 0) for i in range(num_to_log)]
                log_pred_depth_images = [(np.clip(img, 0, 1) * 255).astype(np.uint8) for img in log_pred_depth_images]

                log_best_pred_image = (np.clip(best_pred_image.cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                log_best_pred_depth_image = (np.clip(best_pred_depth_image.cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                log_ood_image = (np.clip(ood_image[0].cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                log_ood_depth_image = (np.clip(ood_depth_image[0].cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                log_input_images = [input_images[i, 0].cpu().detach().numpy().transpose(1, 2, 0) for i in range(num_to_log)]
                log_input_images = [(np.clip(img, 0, 1) * 255).astype(np.uint8) for img in log_input_images]

                # Create grids for input and predicted images
                input_images_grid = vutils.make_grid(torch.tensor(np.array(log_input_images)).permute(0, 3, 1, 2), nrow=4)
                pred_images_grid = vutils.make_grid(torch.tensor(np.array(log_pred_images)).permute(0, 3, 1, 2), nrow=4)
                pred_depth_images_grid = vutils.make_grid(torch.tensor(np.array(log_pred_depth_images)).permute(0, 3, 1, 2), nrow=4)
                top_input_images_grid = vutils.make_grid(torch.tensor(np.array(log_top_input_images)).permute(0, 3, 1, 2), nrow=4)
                top_pred_images_grid = vutils.make_grid(torch.tensor(np.array(log_top_pred_images)).permute(0, 3, 1, 2), nrow=4)
                top_pred_depth_images_grid = vutils.make_grid(torch.tensor(np.array(log_top_pred_depth_images)).permute(0, 3, 1, 2), nrow=4)

                try:
                    # Extract RGB channels from best_input_image (remove origin_distances)
                    best_input_rgb = best_input_image.squeeze(0)[:3, :, :]  # [1, 4, H, W] -> [3, H, W]
                    
                    dict_to_log = {
                        "Original OOD Image": wandb.Image(log_ood_image, caption="Original OOD Image"),
                        "Input Images Grid": wandb.Image(input_images_grid, caption="Input Images Grid"),
                        "Best Predicted Image": wandb.Image(log_best_pred_image, caption=f"Best pred {best_rotation_name}"),
                        "Top Input Images Grid": wandb.Image(top_input_images_grid, caption="Top Input Images Grid"),
                        "Top Predicted Images Grid": wandb.Image(top_pred_images_grid, caption="Top Predicted Images Grid"),
                        "Top Predicted Depth Images Grid": wandb.Image(top_pred_depth_images_grid, caption="Top Predicted Depth Images Grid"),
                        "Predicted Images Grid": wandb.Image(pred_images_grid, caption="Predicted Images Grid"),
                        "Predicted Depth Images Grid": wandb.Image(pred_depth_images_grid, caption="Predicted Depth Images Grid"),
                        "OOD Depth Image": wandb.Image(log_ood_depth_image, caption="OOD Depth Image"),
                        "best_input_image": wandb.Image(best_input_rgb, caption="Best Input Image"),
                        "best_pred_depth_image": wandb.Image(log_best_pred_depth_image, caption="Best Predicted Depth Image"),
                        "best_loss": best_loss.item(), 
                        "mse_loss": mse_loss_sum.item(),
                        "lpips_loss": lpips_loss_sum.item() if lambda_lpips > 0 else -1,
                        "w_reg_loss": w_reg_loss.item(),
                        "current_stage": step_idx+1, 
                    }
                    if lambda_lpips > 0:
                        dict_to_log["lpips_loss"] = lpips_loss_sum.item()   
                    
                    # Log pinned candidate (OOD latent + identity rotation) if available
                    if pinned_input_image is not None and pinned_pred_image is not None:
                        # Extract RGB channels from pinned_input_image
                        pinned_input_rgb = pinned_input_image[:3, :, :]  # [4, H, W] -> [3, H, W]
                        log_pinned_input = (np.clip(pinned_input_rgb.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                        log_pinned_pred = (np.clip(pinned_pred_image.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                        log_pinned_pred_depth = (np.clip(pinned_pred_depth_image.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                        
                        dict_to_log["Pinned (OOD+Identity) Input Image"] = wandb.Image(log_pinned_input, caption="OOD-encoded latent input (should match OOD if in-dist)")
                        dict_to_log["Pinned (OOD+Identity) Pred Image"] = wandb.Image(log_pinned_pred, caption="OOD-encoded latent @ identity rotation")
                        dict_to_log["Pinned (OOD+Identity) Pred Depth"] = wandb.Image(log_pinned_pred_depth, caption="OOD-encoded latent depth @ identity")
                        dict_to_log["pinned_decision_loss"] = pinned_decision_loss if pinned_decision_loss is not None else -1
                    
                    wandb.log(dict_to_log)
                    
                except Exception as e:
                    print(f"Warning: Wandb logging failed with error: {e}")
            log_iters += 1

        # Compute global ranking once per iteration (over all candidate batches),
        # independent of wandb/logging cadence.
        sorted_idxs_by_loss = [idx for idx, _ in sorted(all_losses_and_idxs, key=lambda x: float(x[1]))]

        loss_trajectory.append({
            "iteration": iteration,
            "elapsed_time_s": time.time() - start_time,
            "optim_elapsed_time_s": (time.time() - optim_start_time) if optim_start_time is not None else None,
            "best_loss": best_loss.item(),
            "pinned_loss": pinned_decision_loss,  # loss of directly-encoded OOD candidate (no optimization)
        })

        # Note: used_idxes is controlled by config_steps, not hardcoded here.
        # The config steps already handle forcing pinned_idx for early iterations.
        iteration += 1
        tqdm_iterator.update(1)


@hydra.main(version_base=None, config_path=_cfg.CONFIGS_DIR, config_name="abs_config")  
def main(cfg: DictConfig): 
    import pandas as pd
    global dataset, val_dataset
    
    quality = ["good","med"]
    split = "ood"
    reevaluate = True
    
    dataset_name = cfg.data.category
    log_path = os.path.join(_cfg.RUNS_ROOT, cfg.general.prefix)
    os.makedirs(log_path, exist_ok=True)  # Ensure log directory exists
    results_save_path = f"{log_path}/co3d_se3_results.csv"
    test_imgs_csv_path = cfg.general.test_imgs_csv_path 
    use_hq = cfg.general.use_hq
    run_indist = cfg.general.run_indist
    assert test_imgs_csv_path is not None, "test_imgs_csv_path must be set in the config file"

    test_imgs_csv = pd.read_csv(test_imgs_csv_path)
    test_imgs_csv = test_imgs_csv[(test_imgs_csv["quality"].isin(quality)) & (test_imgs_csv["split"] == split)]

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
        train(cfg, cfg.general.override_example_id, test_idx=cfg.general.test_idx, log_path=log_path, dataset_name=dataset_name, prefix=cfg.general.prefix, run_indist=run_indist)
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
        ood_test_id, override_example_id = val_dataset.find_index_for_sequence_prefix(override_example_id.replace(f"{cfg.data.category[:-1]}_", ""))
        try: 
            if override_example_id in processed_example_ids:
                print(f"Skipping already processed example id: {override_example_id}")
                continue
                # raise Exception(f"Skipping already processed example id: {override_example_id}")
            
            print(f"Running optimization for example id: {override_example_id}")
            start_time = time.time()
            results = train(cfg, override_example_id, test_idx=0, log_path=log_path, dataset_name=dataset_name, prefix=cfg.general.prefix, test_img_path=test_img_path, run_indist=run_indist)
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

