# imports 
# -------------------------------------------------------------------------------------- # 
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

ROOT=f"{_EXT}/splatter-image"
# ROOT="/data/scratch/akiruga/splatter-image"
import sys 
from filelock import FileLock
import time 
import json 
sys.path.append(ROOT)
import csv 
# sys.path.append(f'{ROOT}/experiments/stylegan2-pytorch') # stylegan (not present on this cluster)
sys.path.append(f'{_EXT}/stylegan3') # stylegan3
# from model import Generator  # legacy stylegan2-pytorch, unused
import datetime
import wandb
# REBUTTAL: neutralize wandb media logging (pure logging; zero effect on numerics).
# Avoids a wandb.Image 4-dim permute crash on torch 2.9 that would abort the run
# before the checkpoint save / score return.
wandb.Image = lambda *a, **k: None
wandb.Video = lambda *a, **k: None
wandb.log = lambda *a, **k: None
wandb.init = lambda *a, **k: None 
import os
import numpy as np
import torch
import torchvision.utils as vutils
import lpips as lpips_lib
import argparse
from utils.abs_utils import *
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import dnnlib
import legacy
from utils.general_utils import PILtoTorch
from scene.gaussian_predictor import GaussianSplatPredictor
from tqdm import tqdm
from hydra import initialize, compose
import random 
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from splatter_image_datasets.shapenet_nmr import ShapenetNMR
import imageio
from gaussian_renderer import render_predicted

import hydra

# -------------------------------------------------------------------------------------- # 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = None 
# airplane, bench, cabinet, car, chair, display, lamp, loudspeaker, rifle, sofa, table
class_name_to_idx = {
    "airplane": 0,
    "bench": 1,
    "cabinet": 2,
    "car": 3,
    "chair": 4,
    "display": 5,
    "lamp": 6,
    "loudspeaker": 7,
    "rifle": 8,
    "sofa": 9,
    "table": 10,
}

class_to_object_ids_list_path = f"{ROOT}/rendering_scripts/vincent-renderer/object_ids_per_category_nmr.json"
with open(class_to_object_ids_list_path, "r") as f:
    class_to_object_ids_list = json.load(f)
# create object id to class mapping
object_id_to_class = {object_id.strip(): class_name for class_name, object_ids in class_to_object_ids_list.items() for object_id in object_ids}
    
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
                # c = None if self.G.c_dim == 0 else torch.zeros(z.shape[0], self.G.c_dim, device=z.device)
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
    from utils.general_utils import matrix_to_quaternion
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

def create_loop_and_eval(gaussian_splats, test_data_instance, test_data_name, cfg, save_loop_path=None, save_gt=False):
    """Create video loop and compute metrics"""
    from utils.loss_utils import ssim as ssim_fn
    
    psnr_all = []
    ssim_all = []
    lpips_all = []
    
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device)
    
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    loop_renders = []
    gt_images = []
    
    test_data_instance = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in test_data_instance.items()}
    
    for r_idx in range(test_data_instance["world_view_transforms"].shape[1]):
        world_view_transforms = test_data_instance["world_view_transforms"][0, r_idx].unsqueeze(0)
        full_proj_transforms = test_data_instance["full_proj_transforms"][0, r_idx].unsqueeze(0)
        camera_centers = test_data_instance["camera_centers"][0, r_idx].unsqueeze(0)

        image = render_predicted(gaussian_splats,
                                    world_view_transforms,
                                    full_proj_transforms,  
                                    camera_centers,
                                    background,
                                    cfg,
                                    focals_pixels=None)["render"]
        
        gt_image = test_data_instance["gt_images"][0, r_idx].to(device)
        
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

def eval_baseline(ood_data, ood_index, eval_data, cur_focals_pixels, gaussian_predictor, cfg, background, save_video_path=None, reference_data_override=None):
    """Evaluate baseline: just run splatter image on ood_image directly.

    If reference_data_override is provided, eval cameras are made relative to THAT
    reference (the canonical side-top input camera) instead of the OOD input camera.
    This is the *naive feedforward* baseline that ignores the OOD input pose, so the
    prediction is mis-oriented and metrics degrade -- the "baseline at OOD pose"."""
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    ood_image = ood_data["gt_images"][0, ood_index].to(device)

    # Run splatter image predictor directly on ood_image
    pred_splats = gaussian_predictor(
        ood_image.unsqueeze(0).unsqueeze(1).to(device),
        ood_data["view_to_world_transforms"][:1, ood_index:ood_index+cfg.data.input_images],
        ood_data["source_cv2wT_quat"][:1, ood_index:ood_index+cfg.data.input_images],
        cur_focals_pixels
    )
    pred_splats = {k: v[0] for k, v in pred_splats.items()}

    reference_data = reference_data_override if reference_data_override is not None else ood_data
    reference_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in reference_data.items()}
    eval_data_relative = make_data_relative_to(target_data=eval_data, reference_data=reference_data)
    renders, gt_images, scores = create_loop_and_eval(pred_splats, eval_data_relative, "test", cfg, save_loop_path=save_video_path, save_gt=True)

    return renders, gt_images, scores

def eval_pred(best_input_image, best_rotation_matrix, eval_data, cur_focals_pixels, zgt, background, ood_data, ood_index, gaussian_predictor, cfg, save_video_path=None): 
    """Evaluate our optimized method"""
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in ood_data.items()}
    ood_image = ood_data["gt_images"][0, ood_index].to(device)
    
    pred_splats = gaussian_predictor(
        best_input_image.unsqueeze(1).to(device), 
        ood_data["view_to_world_transforms"][:1, ood_index:ood_index+cfg.data.input_images], 
        ood_data["source_cv2wT_quat"][:1, ood_index:ood_index+cfg.data.input_images], 
        cur_focals_pixels   
    )
    pred_splats = {k: v[0] for k, v in pred_splats.items()}
    transformed_splats = render_with_custom_camera(pred_splats, 
                                                   background, 
                                                   cfg, 
                                                   cur_focals_pixels,
                                                   best_rotation_matrix.to(device),
                                                   zgt, device = device, return_splats=True)
    eval_data_relative = make_data_relative_to(target_data=eval_data, reference_data=ood_data)
    renders, gt_images, scores = create_loop_and_eval(transformed_splats, eval_data_relative, "test", cfg, save_loop_path=save_video_path, save_gt=True)
    
    return renders, gt_images, scores


def load_dataset(dataset_name, val_view, test_idx, train_example_ids, override_example_id, cfg):
    
    
    if dataset_name == "shapenet_nmr":
        print("loading shapenet nmr")
        print("cfg-chairs", cfg)
        # GAP-6 PROBE: load the SAME object's canonical side-top view via override.
        # (The openmind-era side-top intrins JSON references objects not present on
        #  this cluster; using the current object's side-top is a valid canonical
        #  reference and avoids that broken path.)
        dataset = ShapenetNMR(cfg, "train", override_example_ids=[override_example_id], override_overall_view="side-top")
        override_example_ids = [override_example_id] if override_example_id is not None else None
        assert override_example_ids is not None, "Please provide an example id to override"
        val_dataset = ShapenetNMR(cfg, "test", override_overall_view=val_view, override_example_ids=override_example_ids, deterministic_test_idxs = [test_idx])

        eval_data = ShapenetNMR(cfg, "test", override_overall_view="test", override_example_ids=override_example_ids, deterministic_test_idxs = [0])
    
    elif dataset_name in class_name_to_idx:
        print("loading NMR for class", dataset_name)
        print("cfg-chairs", cfg)
        dataset = ShapenetNMR(cfg, "train", override_example_ids=train_example_ids, override_overall_view="side-top", particular_class_list=[dataset_name])
        override_example_ids = [override_example_id] if override_example_id is not None else None
        assert override_example_ids is not None, "Please provide an example id to override"
        val_dataset = ShapenetNMR(cfg, "test", override_overall_view=val_view, override_example_ids=override_example_ids, deterministic_test_idxs = [test_idx], particular_class_list=[dataset_name])

        eval_data = ShapenetNMR(cfg, "test", override_overall_view="test", override_example_ids=override_example_ids, deterministic_test_idxs = [0], particular_class_list=[dataset_name])
    
    return dataset, val_dataset, eval_data

def load_models(): 
        
    gaussian_predictor = GaussianSplatPredictor(cfg)
    gaussian_predictor = gaussian_predictor.to(memory_format=torch.channels_last)
    gaussian_predictor = gaussian_predictor.to(device)
    
    model_path = f"{ROOT}/experiments_out/2025-08-06/12-11-04/model_latest.pth"
    if cfg.opt.pretrained_ckpt is not None: 
        model_path = cfg.opt.pretrained_ckpt

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    gaussian_predictor.load_state_dict(checkpoint["model_state_dict"])
    print('side-top shapenet nmr all classes splatter image model loaded')

    # Use StyleGAN3 snapshot (.pkl)
    generator_ckpt_path = f"{_EXT}/stylegan3/training-runs/00004-stylegan2-nmr128-gpus8-batch256-gamma10/network-snapshot-020889.pkl"
    

    generator = StyleGANCondGenerator(generator_ckpt_path)
        
    return gaussian_predictor, generator

def idx_to_label(idx, G, num_samples): 
    label = torch.zeros(num_samples, G.c_dim, device=device)
    label[:, idx] = 1
    return label

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

def train(cur_cfg, override_example_id = None, test_idx = 0, log_path = "checkpoints-side-top-so3-last-try", val_view = None, dataset_name = None, prefix = None): 
    # global variables for easier access 
    global used_idxes, sorted_idxs_by_loss, best_loss
    global log_iters, log_every, lambda_lpips, lambda_mse
    global batch_size, num_batches
    global is_debugging
    global cfg
    global rot_optim, is_rot_optim
    global wandb_run
    
    assert val_view is not None, "Please provide a val_view"
    torch.set_float32_matmul_precision('high') # not sure why but found this in splatter image code 
    
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
    cfg.abs.random_seed =  abs(hash(override_example_id)) % 2**32
    g = torch.Generator()
    np.random.seed(cfg.abs.random_seed)
    torch.manual_seed(cfg.abs.random_seed)
    torch.cuda.manual_seed(cfg.abs.random_seed)
    random.seed(cfg.abs.random_seed)
    g.manual_seed(cfg.abs.random_seed)

    # load distinct example ids 
    train_example_ids = None

    dataset, val_dataset, eval_data = load_dataset(dataset_name, val_view, test_idx, train_example_ids, override_example_id, cfg)
    
    dataloader = DataLoader(dataset,
                            batch_size=cfg.abs.num_latents,
                            shuffle=True, generator=g)

    val_dataloader = DataLoader(val_dataset, 
                                    batch_size=1,
                                    shuffle=True, generator=g)
    
    eval_dataloader = DataLoader(eval_data,
                            batch_size=1,
                            shuffle=True, generator=g)
    
    eval_iter = iter(eval_dataloader)
    eval_data = next(eval_iter)
    eval_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in eval_data.items()}

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

    if not cfg.abs.custom_ood_image and not cfg.abs.random_ood_image:
        val_iter = iter(val_dataloader)
        ood_data = next(val_iter)
        
        ood_image_index = 0
        
        print("ood_image_index", ood_image_index)
        
        ood_image = torch.tensor(np.clip(ood_data["gt_images"][0, ood_image_index].cpu().numpy(), 0, 1)).unsqueeze(0).to(device)
        ood_focals_pixels = ood_data.get("focals_pixels", [None])[0]
        if ood_focals_pixels is not None:
            ood_focals_pixels = ood_focals_pixels[ood_image_index].unsqueeze(0).to(device)
            
    elif cfg.abs.custom_ood_image and not cfg.abs.random_ood_image: 
        from utils.app_utils import remove_background, resize_foreground, set_white_background, resize_foreground_w_size_output, generate_mask_from_alpha
        if cfg.abs.custom_img_remove_background:
            import rembg 
            rembg_session = rembg.new_session()
        
            ood_image = Image.open(cfg.abs.custom_ood_image)
            if ood_image.mode == "RGBA":
                ood_image = alpha_blend_with_background(ood_image)
                ood_image = ood_image.convert('RGB')
            
            ood_image = remove_background(ood_image, rembg_session)
            foreground_ratio = 0.65
            ood_image = resize_foreground(ood_image, foreground_ratio)
            ood_mask = generate_mask_from_alpha(ood_image)
            ood_mask = ood_mask.to(device)
            ood_image = set_white_background(ood_image)
        
        else: 
            ood_image = Image.open(cfg.abs.custom_ood_image)
            ood_image = resize_foreground_w_size_output(ood_image, ratio=0.65, output_size=128)
            ood_mask = generate_mask_from_alpha(ood_image)
            ood_mask = ood_mask.to(device)
            ood_image = set_white_background(ood_image)
            
        ood_image = PILtoTorch(ood_image, (cfg.data.training_resolution, cfg.data.training_resolution)).clamp(0.0, 1.0)[:3, :, :]
        ood_image = ood_image.unsqueeze(0).to(device)
        assert ood_image.shape[1] == 3, f"OOD image should have 3 channels cf. {ood_image.shape}"
    
    elif cfg.abs.random_ood_image and not cfg.abs.custom_ood_image: 
        
        ood_data = next(iter(val_dataloader))
        
        ood_image_index = random.randint(0, ood_data["gt_images"].shape[1] - 1)
        
        ood_input_images = ood_data["gt_images"][0, ood_image_index].unsqueeze(0).unsqueeze(0).to(device)
        ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].unsqueeze(0).unsqueeze(0).to(device)
        ood_focals_pixels = ood_data.get("focals_pixels", [None])[0]
        if ood_focals_pixels is not None:
            ood_focals_pixels = ood_focals_pixels[ood_image_index].unsqueeze(0).unsqueeze(0).to(device)
        ood_view_to_world_transforms = ood_data["view_to_world_transforms"][0, 0].unsqueeze(0).unsqueeze(0).to(device)
        ood_source_cv2wT_quats = ood_data["source_cv2wT_quat"][0, 0].unsqueeze(0).unsqueeze(0).to(device)
        ood_zgt = ood_origin_distances.max().item()
    else: 
        raise ValueError("Please provide either a custom ood image as a path, set random_ood_image to True to use a rendered image or set both to False to use a real image")

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
    train_data = next(train_data_iter) # b, ...
    
    if dataset_name == "shapenet_cars_modified": 
        zgt = 1.3 # fixed sphere radius as per SRN paper
        latent_std = 17.4755
        gen_class_idx = 3
    elif dataset_name == "shapenet_chairs_modified": 
        zgt = 1.3
        latent_std = 18.68
        gen_class_idx = 4
    elif dataset_name == "shapenet_nmr": 
        zgt = 1.3
        latent_std = 17.4755
        current_category = object_id_to_class[override_example_id.strip()]
        gen_class_idx = class_name_to_idx[current_category]
        assert gen_class_idx is not None, f"gen_class_idx should be not None cf. {gen_class_idx}"
    print("Using zgt: ", zgt, "using gen_class_idx: ", gen_class_idx)
 
    
    train_image = generator_transform(train_data['gt_images'][:, 0]).to(device)
    assert train_image.shape[1] == 3, f"Train image should have 3 channels cf. {train_image.shape}"

    train_focals_pixels = train_data.get("focals_pixels", [None])[0]
    
    if train_focals_pixels is not None:
        train_focals_pixels = train_focals_pixels[0].unsqueeze(0).to(device)
    
    # Z space
    init_search_cond = torch.randn(cfg.abs.num_latents, 512, device=device) # b, 512
    
    
    with torch.no_grad(): 
        # take to W space 
        init_search_cond = generator.model.style(init_search_cond, idx_to_label(gen_class_idx, generator.G, cfg.abs.num_latents))
        
    assert init_search_cond.shape[0] == cfg.abs.num_latents, f"init_search_cond should have shape {cfg.abs.num_latents} cf. {init_search_cond.shape}"
    search_cond_list = [init_search_cond.clone().detach().requires_grad_(True) for _ in range(num_rotations)] # num_rotations, b, 512
    search_cond = nn.Parameter(torch.cat(search_cond_list, dim=0)) # num_rotations * b, 512
    
    stylegan_noises_single = generator.model.make_noise()
    stylegan_noises = []
    init_num_images = cfg.abs.num_latents * cfg.abs.num_rotations
    for noise in stylegan_noises_single:
        stylegan_noises.append(noise.repeat(init_num_images, 1, 1, 1).clone().normal_().requires_grad_(True))
    
    assert len(search_cond.shape) == 2 and search_cond.shape[0] == num_rotations * cfg.abs.num_latents, f"search_cond should have shape {num_rotations * cfg.abs.num_latents, 512} cf. {search_cond.shape}"
    # jumble them up so we can log different latents
    # search_cond = search_cond[torch.randperm(search_cond.size(0))]
    
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
    best_rotation_name = ""

    top_k = 10 # Define the value of K
    assert top_k <= num_rotations, f"top_k should be less than or equal to num_rotations cf. {top_k} and {num_rotations}"
    top_k_pred_images = []
    top_k_input_images = []
    top_k_losses = []
    top_k_rotation_matrices = []
    top_k_indexes = []


    random_rotations = get_random_cameras(num_rotations, zgt, device = device) # num_rotations, 3, 3
    # we want num_rotations * num_latents, 3, 3
    random_rotations = torch.stack([random_rotations.clone() for _ in range(cfg.abs.num_latents)], dim=1)
    random_rotations = random_rotations.reshape(-1, 3, 3)
    random_rotations.requires_grad_(True)
    optimizer = optim.Adam([ 
        {'params': random_rotations, 'lr': cfg.abs.lr_rotations},
        {'params': search_cond, 'lr': cfg.abs.w_lr},
        {'params': stylegan_noises, 'lr': cfg.abs.w_lr},
    ])

    log_iters = 0

    if cfg.abs.random_ood_image: 
        
        print("using random_seed: ", cfg.abs.random_seed, cfg.general.random_seed)
        
        np.random.seed(cfg.abs.random_seed + 1)
        
        ood_random_rotations = get_random_cameras(1, ood_zgt, device = device)
        
        cur_ood_input_images = torch.cat([ood_input_images, ood_origin_distances], dim = 2)
        
        gaussian_splats_ood = gaussian_predictor( 
            cur_ood_input_images, 
            ood_view_to_world_transforms, 
            ood_source_cv2wT_quats, 
            ood_focals_pixels 
        )
        
        print("gaussian_splats_ood['xyz'].shape", gaussian_splats_ood["xyz"].shape)
        
        # render with random camera
        gaussian_splats_ood = {k: v[0] for k, v in gaussian_splats_ood.items()} 
        ood_render_image = render_with_custom_camera(gaussian_splats_ood, background, cfg, ood_focals_pixels[0, 0], ood_random_rotations[0], ood_zgt, device = device)
        ood_image = torch.tensor(np.clip(ood_render_image.cpu().detach().numpy(), 0, 1)).to(device)
        
        del gaussian_splats_ood
        del ood_render_image
        

    prev_best_loss = None

    import time
    used_idxes = list(range(num_rotations * cfg.abs.num_latents))
    sorted_idxs_by_loss = []

    # Evaluate baseline once at the start
    print("Evaluating baseline (splatter image only)...")
    baseline_video_path = f"{experiment_file_path}/baseline_video.mp4"
    with torch.no_grad():
        baseline_renders, baseline_gt_images, baseline_scores = eval_baseline(
            ood_data, ood_image_index, eval_data, ood_focals_pixels, 
            gaussian_predictor, cfg, background, save_video_path=baseline_video_path
        )
    print(f"Baseline scores: {baseline_scores}")

    # ---- GAP-6 PROBE: also compute the NAIVE baseline rendered at the OOD pose ----
    # Naive = splatter prediction from the SO(3) input, but eval cameras made relative
    # to the canonical side-top input camera (train_data) instead of the so3 input.
    # This exposes the mis-orientation the feedforward model cannot correct.
    with torch.no_grad():
        _, _, baseline_naive_scores = eval_baseline(
            ood_data, ood_image_index, eval_data, ood_focals_pixels,
            gaussian_predictor, cfg, background, save_video_path=None,
            reference_data_override=train_data,
        )
    print(f"Baseline NAIVE (at OOD pose) scores: {baseline_naive_scores}")
    # REBUTTAL: keep the naive numbers as extra columns and PROCEED to the full
    # optimization (the probe's early-return is removed).
    baseline_scores = {**baseline_scores,
                       **{k + "_naive": v for k, v in baseline_naive_scores.items()}}

    # Initialize timing and configuration
    config_steps = [
        {
            "steps": 0,  # Initial configuration
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
            "steps": 3,  # Initial configuration
            # "steps": 1,  # Initial configuration
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
            # "steps": 61,  # After 5 minutes
            "steps": 122,  # After 5 minutes
            # "steps": 5,  # After 5 minutes
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
            # "steps": 151,  # After another 5 minutes
            "steps": 302,  # After another 5 minutes
            # "steps": 10,  # After another 5 minutes
            "settings": {
                "batch_size": 5,
                "lambda_lpips": 2,
                # "lambda_lpips": 10,
                "lambda_mse": 10,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: int(len(used_idxes) / batch_size),
                
            },
            
        },
        {
            # "steps": 151,  # After another 5 minutes
            "steps": 732,  # After another 5 minutes
            # "steps": 15,  # After another 5 minutes
            "settings": {
                "batch_size": 5,
                # "lambda_lpips": 0.0,
                # "lambda_lpips": 10.0,
                # "lambda_mse": 10,
                "lambda_mse": 2,
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "num_batches": lambda: int(len(used_idxes) / batch_size),
                
            },
            
        },
        {
            # "steps": 151,  # After another 5 minutes
            "steps": 782,  # After another 5 minutes
            # "steps": 20,  # After another 5 minutes
            "settings": {
                "used_idxes": lambda: sorted_idxs_by_loss[:batch_size],
                "best_loss": float('inf'),
                "log_iters": 0,
                "log_every": 100,
                "batch_size": 5,
                "num_batches": lambda: int(len(used_idxes) / batch_size),
            },
            
        },
        # End condition at 900 seconds (15 minutes)
        # {"steps": 366, "settings": "stop"}
        # {"steps": 20, "settings": "stop"}
        {"steps": 783, "settings": "stop"}
        # {"steps": 25, "settings": "stop"}
        # {"steps": 10000, "settings": "stop"}
    ]


    # Loop control
    step_idx = 0
    iteration = 0
    start_time = time.time()

    tqdm_iterator = tqdm(total=num_iterations, desc="iterations")

    while step_idx < len(config_steps):
        # used to track the losses and indexes of the current step
        all_losses_and_idxs = []
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Check if current step needs to be applied
        if iteration >= config_steps[step_idx]["steps"]:
            assert step_idx == 0 or len(sorted_idxs_by_loss) > 0, f"sorted_idxs_by_loss {sorted_idxs_by_loss} should not be empty"     
            settings = config_steps[step_idx]["settings"]
            if settings == "stop":
                # Log images one final time before exiting
                dict_to_log = {
                    "Original OOD Image": wandb.Image(log_ood_image, caption="Original OOD Image"),
                    "Input Images Grid": wandb.Image(input_images_grid, caption="Input Images Grid"),
                    "Best Predicted Image": wandb.Image(log_best_pred_image, caption=f"Best pred {best_rotation_name}"),
                    "Top Input Images Grid": wandb.Image(top_input_images_grid, caption="Top Input Images Grid"),
                    "Top Predicted Images Grid": wandb.Image(top_pred_images_grid, caption="Top Predicted Images Grid"),
                    "Predicted Images Grid": wandb.Image(pred_images_grid, caption="Predicted Images Grid"),
                    "best_input_image": wandb.Image(best_input_image, caption="Best Input Image"),
                    "best_loss": best_loss.item(), 
                    "current_stage": step_idx+1, 
                }
                if lambda_lpips > 0:
                    dict_to_log["lpips_loss"] = lpips_loss_sum.item()   

                wandb.log(dict_to_log)  # Log the final images and data
                
                # Evaluate our method on top-k
                all_scores = []
                
                best_scores = None 
                avg_scores = None
                for idx in range(top_k):
                    eval_input_image = top_k_input_images[idx].unsqueeze(0)
                    eval_rotation_matrix = top_k_rotation_matrices[idx]
                    ours_video_path = None  # Don't save individual videos for top-k
                    _, _, cur_scores = eval_pred(eval_input_image, eval_rotation_matrix, eval_data, ood_focals_pixels, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=ours_video_path)
                    if best_scores is None:
                        best_scores = cur_scores.copy()
                        avg_scores = {key: 0 for key in cur_scores.keys()}
                    else: 
                        # maximize psnr 
                        if cur_scores["PSNR_novel"] > best_scores["PSNR_novel"]:
                            best_scores = cur_scores.copy()
                        # average all scores
                        for key, value in cur_scores.items():
                            avg_scores[key] += value
                    all_scores.append(cur_scores)
                
                # Evaluate the known best with video
                ours_best_video_path = f"{experiment_file_path}/ours_best_video.mp4"
                known_best_renders, known_best_gt_images, known_best_scores = eval_pred(best_input_image, best_rotation_matrix, eval_data, ood_focals_pixels, zgt, background, ood_data, ood_image_index, gaussian_predictor, cfg, save_video_path=ours_best_video_path)
                if known_best_scores["PSNR_novel"] > best_scores["PSNR_novel"]:
                    best_scores = known_best_scores.copy()
                all_scores.append(known_best_scores)
                
                for key in avg_scores.keys():
                    avg_scores[key] /= top_k 
                avg_scores = {key + "_avg": value for key, value in avg_scores.items()}
                best_scores = {key + "_best": value for key, value in best_scores.items()}
                known_best_scores = {key + "_known_best": value for key, value in known_best_scores.items()}
                baseline_scores_wandb = {key + "_baseline": value for key, value in baseline_scores.items()}
                merged_scores = {**avg_scores, **best_scores, **known_best_scores, **baseline_scores_wandb}
                
                print(f"Final merged scores (including baseline): {merged_scores}")
                
                # Log videos and comparison to wandb
                baseline_renders_grid = vutils.make_grid(torch.tensor(baseline_renders).permute(0, 3, 1, 2), nrow=3)
                ours_renders_grid = vutils.make_grid(torch.tensor(known_best_renders).permute(0, 3, 1, 2), nrow=3)
                gt_grid = vutils.make_grid(torch.tensor(known_best_gt_images).permute(0, 3, 1, 2), nrow=3)
                
                wandb.log({
                    "Baseline Renders Grid": wandb.Image(baseline_renders_grid, caption="Baseline Renders"),
                    "Ours Best Renders Grid": wandb.Image(ours_renders_grid, caption="Ours Best Renders"),
                    "GT Images Grid": wandb.Image(gt_grid, caption="Ground Truth"),
                    "Baseline Video": wandb.Video(baseline_video_path, fps=4, format="mp4"),
                    "Ours Best Video": wandb.Video(ours_best_video_path, fps=4, format="mp4"),
                    **merged_scores
                })
                
                # save topk everything and best everything to experiment_file_path using torch.save
                torch.save({
                    "top_k_pred_images": top_k_pred_images,
                    "top_k_input_images": top_k_input_images,
                    "top_k_losses": top_k_losses,
                    "top_k_rotation_matrices": top_k_rotation_matrices,
                    "zgt": zgt,
                    "best_pred_image": best_pred_image,
                    "best_input_image": best_input_image,
                    "best_loss": best_loss,
                    "best_rotation_matrix": best_rotation_matrix,
                    "best_rotation_name": best_rotation_name, 
                    "best_search_cond": best_search_cond, 
                    "ood_data": ood_data, 
                    "ood_image_index": ood_image_index,
                    "example_id": override_example_id,
                    "scores": merged_scores,  # This already includes baseline scores with "_baseline" suffix
                    "all_scores": all_scores,
                    "baseline_scores": baseline_scores  # Also save raw baseline scores separately
                }, os.path.join(experiment_file_path, f"topk_best_everything_latest.pth"))

                print("Reached the end of timed conditions. Exiting.")
                return merged_scores
                break
        
            print(f"Applying configuration step {step_idx} at {iteration} steps and {elapsed_time} seconds")
            # Apply each setting in this configuration step
            for key, value in settings.items():
                if callable(value):
                    globals()[key] = value()  # Execute lambda with existing value if needed
                else:
                    globals()[key] = value  # Set directly

            # Move to the next configuration step
            step_idx += 1
            assert int(len(used_idxes) / batch_size) == num_batches, f"num_batches {num_batches} should be equal to {int(len(used_idxes) / batch_size)}"
            
        if is_rot_optim and not rot_optim: 
            optimizer = optim.Adam([ 
                # {'params': random_rotations, 'lr': cfg.abs.lr_rotations},
                {'params': search_cond, 'lr': cfg.abs.w_lr},
                {'params': stylegan_noises, 'lr': cfg.abs.w_lr},
            ])
            is_rot_optim = False
            
        for batch_idx in range(num_batches):
            
            optimizer.zero_grad()
            cur_idxs = used_idxes[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            
            cur_search_cond = search_cond[cur_idxs]
            cur_random_rotations = random_rotations[cur_idxs]
            cur_fixed_xT = fixed_xT[cur_idxs]
            cur_stylegan_noises = [noise[cur_idxs] for noise in stylegan_noises]
            
            
            # add noise to latent to increase exploration 
            end_steps = config_steps[-1]["steps"]
            t = iteration / end_steps
            
            noise_strength = latent_std * cfg.abs.w_noise * max(0, 1 - t / cfg.abs.w_noise_ramp) ** 2
            cur_latent_in = latent_noise_stylegan(cur_search_cond, noise_strength)
            
            search_cond_lr = get_lr_stylegan(t, cfg.abs.w_lr)
            
            optimizer.param_groups[1]['lr'] = search_cond_lr
            
            # convert to SO3
            if iteration > 0 and cfg.abs.orthogonalize_rotations:
                cur_random_rotations = symmetric_orthogonalization(cur_random_rotations.view(-1, 9))
            
            cur_num_rotations = cur_search_cond.shape[0]

            input_images = generator.forward_latent_w(cur_latent_in, noises=cur_stylegan_noises)
            
            input_images = input_images.clamp(0, 1)
            input_images = input_images.unsqueeze(1).to(device) # B, 1, C, H, W

            if cfg.data.category == "hydrants" or cfg.data.category == "teddybears":
                origin_distances = train_data["origin_distances"][:, :cfg.data.input_images, ...].repeat(cur_num_rotations, 1, 1, 1, 1)

                input_images = torch.cat([input_images, origin_distances], dim  = 2)
                cur_focals_pixels = train_data["focals_pixels"][:1, :cfg.data.input_images].repeat(cur_num_rotations, 1, 1)
            else: 
                cur_focals_pixels = None

            gaussian_splats_vis = gaussian_predictor(
                input_images, 
                train_data["view_to_world_transforms"][:1, :cfg.data.input_images, ...].repeat(cur_num_rotations, 1, 1, 1),
                train_data["source_cv2wT_quat"][:1, :cfg.data.input_images].repeat(cur_num_rotations, 1, 1), 
                cur_focals_pixels,
            )

            pred_images = []


            for i in range(cur_num_rotations):
                transformed_splats = {k: v[i] for k, v in gaussian_splats_vis.items()}
                pred_image = render_with_custom_camera(
                    transformed_splats,
                    background,
                    cfg,
                    train_data["focals_pixels"][0, 0] if cur_focals_pixels is not None else None,
                    cur_random_rotations[i],
                    zgt,
                    device
                )
                
                if cfg.abs.use_mask: 
                    pred_image = pred_image * ood_mask
                
                pred_images.append(pred_image)
            

            pred_images = torch.cat(pred_images, dim=0)
            ood_images = ood_image.expand(cur_num_rotations, -1, -1, -1)
            
            mse_losses = get_mse_loss(pred_images, ood_images)
            mse_losses = mse_losses.view(-1)
            
            
            # Combine the losses for each image
            losses = [
                lambda_mse * mse_losses[i]
                for i in range(cur_num_rotations)
            ]
            
            # Compute LPIPS loss for the whole batch if enabled
            if lambda_lpips > 0:
                lpips_loss_values = lpips_fn(pred_images * 2 - 1, ood_images * 2 - 1)
                lpips_loss_values = lpips_loss_values.view(-1)  # Make sure it's a 1D tensor with individual losses
                losses = [losses[i] + lambda_lpips * lpips_loss_values[i] for i in range(cur_num_rotations)]
            else: 
                lpips_loss_values = torch.zeros(cur_num_rotations)

            # compute noise regularization loss 
            noise_regulate_loss = noise_regularize_stylegan(cur_stylegan_noises)
            
            total_loss = sum(losses) + cfg.abs.w_noise_regularize * noise_regulate_loss 
            
            all_losses_and_idxs.extend([(cur_idxs[i], losses[i]) for i in range(len(cur_idxs))])
            
            
            total_loss.backward()
            optimizer.step() 
            
            # normalize the stylegan_noises 
            noise_normalize_stylegan_(stylegan_noises)

            # cur_best_loss_idx = torch.argmin(torch.tensor([l.item() for l in losses]))
            
            decision_losses = torch.tensor([l_lpips.item() + 4 * l_mse.item() for l_mse, l_lpips in zip(mse_losses, lpips_loss_values)])
            cur_best_loss_idx = torch.argmin(decision_losses)

            if decision_losses[cur_best_loss_idx] < best_loss:
                best_pred_image = pred_images[cur_best_loss_idx].detach().cpu()
                best_input_image = input_images[cur_best_loss_idx].detach().cpu()
                # best_loss = losses[cur_best_loss_idx]
                best_loss = decision_losses[cur_best_loss_idx]
                best_rotation_matrix = cur_random_rotations[cur_best_loss_idx].detach().cpu()
                best_search_cond = cur_search_cond[cur_best_loss_idx].detach().cpu()
                # need to acount for batch index
                best_rotation_name = f"Rotation {cur_best_loss_idx + batch_idx * batch_size + 1}"
            
            # Update top K images list
            for i in range(cur_num_rotations):
                cur_index = i + batch_idx * batch_size
                if len(top_k_losses) < top_k:
                    top_k_losses.append(decision_losses[i].item())
                    top_k_pred_images.append(pred_images[i].detach().cpu())
                    top_k_input_images.append(input_images[i, 0].cpu().detach())
                    top_k_rotation_matrices.append(cur_random_rotations[i].detach().cpu())
                    top_k_indexes.append(cur_index)
                else:
                    # if cur_index is already in top_k_indexes
                    if cur_index in top_k_indexes: 
                        replace_idx = top_k_indexes.index(cur_index)
                    else: 
                        replace_idx = next((idx for idx, loss in enumerate(top_k_losses) if loss > decision_losses[i].item()), None)

                    if replace_idx is not None: 
                        top_k_losses[replace_idx] = decision_losses[i].item()
                        top_k_pred_images[replace_idx] = pred_images[i].detach().cpu()
                        top_k_input_images[replace_idx] = input_images[i, 0].cpu().detach()
                        top_k_rotation_matrices[replace_idx] = cur_random_rotations[i].detach().cpu()
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

                # if iteration % log_every == 0:
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
                    dict_to_log = {
                        "Original OOD Image": wandb.Image(log_ood_image, caption="Original OOD Image"),
                        "Input Images Grid": wandb.Image(input_images_grid, caption="Input Images Grid"),
                        "Best Predicted Image": wandb.Image(log_best_pred_image, caption=f"Best pred {best_rotation_name}"),
                        "Top Input Images Grid": wandb.Image(top_input_images_grid, caption="Top Input Images Grid"),
                        "Top Predicted Images Grid": wandb.Image(top_pred_images_grid, caption="Top Predicted Images Grid"),
                        "Predicted Images Grid": wandb.Image(pred_images_grid, caption="Predicted Images Grid"),
                        "best_input_image": wandb.Image(best_input_image, caption="Best Input Image"),
                        "best_loss": best_loss.item(), 
                        "mse_loss": mse_loss_sum.item(),
                        "lpips_loss": lpips_loss_sum.item() if lambda_lpips > 0 else -1,
                        "current_stage": step_idx+1, 
                        
                    }
                    if lambda_lpips > 0:
                        dict_to_log["lpips_loss"] = lpips_loss_sum.item()   
                    
                    if cfg.abs.use_mask: 
                        log_ood_mask = (np.clip(ood_mask.cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                        dict_to_log["OOD Mask"] = wandb.Image(log_ood_mask, caption="OOD Mask")
                        
                    # Log images to wandb
                    wandb.log(dict_to_log)
                    
                except Exception as e:
                    # Plot grid for input images
                    fig_input, axes_input = plt.subplots(2, 4, figsize=(20, 10))
                    
                    axes_input[0, 0].imshow(log_ood_image)
                    axes_input[0, 0].set_title("Original OOD Image")
                    axes_input[0, 0].axis("off")
                    
                    for idx, ax in enumerate(axes_input.flatten()[1:num_to_log + 1]):
                        ax.imshow(log_input_images[idx])
                        # Rotation {cur_best_loss_idx + batch_idx * batch_size + 1}
                        ax.set_title(f"Input Image {idx + batch_idx * batch_size + 1}")
                        ax.axis("off")
                    log_best_input_image = (np.clip(best_input_image[0].cpu().detach().numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                    axes_input.flatten()[-1].imshow(log_best_input_image)
                    axes_input.flatten()[-1].set_title(f"Best input {best_rotation_name}")
                    axes_input.flatten()[-1].axis("off")

                    plt.tight_layout()
                    plt.show()
                    
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    axes[0, 0].imshow(log_ood_image)
                    axes[0, 0].set_title("Original OOD Image")
                    axes[0, 0].axis("off")

                    for idx, ax in enumerate(axes.flatten()[1:num_to_log + 1]):
                        ax.imshow(log_pred_images[idx])
                        ax.set_title(f"Rotation {idx + 1}")
                        ax.axis("off")

                    axes.flatten()[-1].imshow(log_best_pred_image)
                    axes.flatten()[-1].set_title(f"Best pred {best_rotation_name}")
                    axes.flatten()[-1].axis("off")

                    plt.show()

                # check stopping criterion
                if prev_best_loss is None: 
                    prev_best_loss = best_loss
                    
            
            sorted_idxs_by_loss = [idx for idx, _ in sorted(all_losses_and_idxs, key=lambda x: x[1])]
            log_iters += 1 # independent log_iters counter 
        iteration+= 1
        tqdm_iterator.update(1)

     


@hydra.main(version_base=None, config_path='../../configs', config_name="abs_config")  
def main(cfg:  DictConfig): 
    
    ood_view = "so3"
    
    dataset_name = cfg.data.category
    log_path = f"{ROOT}/{cfg.general.prefix}"
    results_save_path = f"{log_path}/task_a_ours_so3.csv"
    # Initialize processed_example_ids set
    if all([cfg.general.override_example_id, cfg.general.ood_view]): 
        print("using custom override example id")
        val_view = cfg.general.ood_view if cfg.general.ood_view !="ood" else ood_view
        train(cfg, cfg.general.override_example_id, test_idx = cfg.general.test_idx, log_path=log_path, val_view=val_view, dataset_name=dataset_name, prefix=cfg.general.prefix)
        return 
    
    processed_example_ids = set()
    
    # Check if results file exists and load processed example_ids if it does
    if os.path.exists(results_save_path):
        with open(results_save_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_example_ids.add(row["example_id"])
            
    with open(cfg.general.data_example_ids_path, "r") as f: 
        data_example_ids_with_views = json.load(f)
    test_ids_to_use = data_example_ids_with_views
    # check if not null and is integer
    if cfg.general.maxsamples is not None and isinstance(cfg.general.maxsamples, int):
        print("Using maxsamples:", cfg.general.maxsamples)
        test_ids_to_use = test_ids_to_use[:cfg.general.maxsamples]
    size_per_split = len(test_ids_to_use) // cfg.general.total_splits
    remainder = len(test_ids_to_use) % cfg.general.total_splits
    cur_size = size_per_split
    if cfg.general.split == cfg.general.total_splits - 1: 
        cur_size += remainder
    
    cur_example_ids = test_ids_to_use[cfg.general.split * size_per_split: (cfg.general.split * size_per_split) + cur_size]
    
    lock_path = results_save_path + ".lock"
    
    for override_example_id, view, test_img_idx in cur_example_ids: 
        try: 
            # Skip if example ID is already processed
            if override_example_id in processed_example_ids:
                print(f"Skipping already processed example id: {override_example_id}")
                continue
            
            print(f"Running optimization for example id: {override_example_id}")
            start_time = time.time()
            val_view = view if view !="ood" else ood_view
            results = train(cfg, override_example_id, test_idx = test_img_idx, log_path=log_path, val_view = val_view, dataset_name=dataset_name, prefix=cfg.general.prefix)
            results["example_id"] = override_example_id
            # Append the result directly to the CSV file with locking to prevent race conditions
            with FileLock(lock_path), open(results_save_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                
                # Write header if file is empty
                if os.path.getsize(results_save_path) == 0:
                    writer.writeheader()
                
                writer.writerow(results)
            
            # Add the example_id to the processed set
            print(f"Result for {override_example_id} appended to {results_save_path}")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Finished example id {override_example_id} in {elapsed_time:.2f} seconds")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error for example id {override_example_id}: {e}")
            continue
        
if __name__ == "__main__": 
    main()

