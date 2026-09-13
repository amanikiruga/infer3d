"""CATSplat model loading and input preparation for the RE10K experiment.

Vendored so this repository is self-contained: the upstream CATSplat release does
not ship these helpers. Requires CATSPLAT_ROOT on sys.path for its `datasets` and
`evaluation` modules (eval_table3.py arranges that).
"""
"""
CATSplat RE10K OOD Data Generator

Generate GT or rendered images for RE10K dataset:
- By default (render_catsplat=False): saves GT images from linearly sampled views
- With render_catsplat=True: uses first image as input to CATSplat,
                             then renders at linearly sampled views using GT poses
- With render_input_images_path: uses modified images (e.g. fisheye) as input to CATSplat
                                  instead of GT images, while keeping GT poses/intrinsics

Usage examples:
    # Save GT images (5 sequences, 24 views each)
    python experiments/diffae_catsplat_re10k/generate_ood_data_re10k.py \
        +general.num_images=5 +general.num_views=24 \
        +general.save_path=/path/to/output/re10k_gt \
        general.random_seed=42

    # Render with CATSplat (5 sequences, 24 views each)
    python experiments/diffae_catsplat_re10k/generate_ood_data_re10k.py \
        +general.num_images=5 +general.num_views=24 \
        +general.save_path=/path/to/output/re10k_render \
        general.random_seed=42 \
        +general.render_catsplat=true \
        run.checkpoint=/path/to/catsplat_checkpoint.pth

    # Render with modified input images (e.g. fisheye)
    python experiments/diffae_catsplat_re10k/generate_ood_data_re10k.py \
        +general.num_images=5 +general.num_views=24 \
        +general.save_path=/path/to/output/re10k_fisheye_render \
        general.random_seed=42 \
        +general.render_catsplat=true \
        +general.render_input_images_path=/path/to/re10k_fisheye_gt \
        run.checkpoint=/path/to/catsplat_checkpoint.pth
"""

import sys
import os
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra
from PIL import Image
import torchvision.transforms as T

# Add CATSplat root to path
CATSPLAT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(CATSPLAT_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_catsplat_model(cfg, checkpoint_path):
    """Load pretrained CATSplat model."""
    from models.model import GaussianPredictor

    model = GaussianPredictor(cfg)
    model = model.to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        model.load_model(checkpoint_path, device='cuda', ckpt_ids=0)
        print(f"Loaded CATSplat from {checkpoint_path}")
    else:
        print(f"WARNING: No checkpoint provided or checkpoint not found at {checkpoint_path}")

    model.set_eval()
    return model


def load_re10k_dataset(cfg, split="test"):
    """Load RE10K dataset using CATSplat's built-in loader."""
    from datasets.util import create_datasets

    dataset, _ = create_datasets(cfg, split=split)
    return dataset


def sample_frame_indices(total_frames, num_views):
    """Sample frame indices evenly using linspace."""
    return np.linspace(0, total_frames - 1, num_views, dtype=int)


def prepare_catsplat_inputs(source_data, target_data=None):
    """
    Prepare inputs dict for CATSplat model.

    Args:
        source_data: Source frame data (dict with keys like ("color", 0, 0), etc.)
        target_data: Optional target frame data (dict with keys like ("color", 1, 0), etc.)

    Returns:
        inputs: Dict ready for CATSplat forward pass
    """
    inputs = {}

    # Copy source frame data (frame 0)
    for key, value in source_data.items():
        if isinstance(key, tuple) and key[0] in ["color", "color_aug", "K_tgt", "K_src", "inv_K_src", "T_c2w", "T_w2c", "llava_feat", "depth_sparse", "unidepth", "scale_colmap"]:
            inputs[key] = value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value

    # Add target frame data if provided (frame 1)
    if target_data is not None:
        for key, value in target_data.items():
            if isinstance(key, tuple) and key[0] in ["color", "K_tgt", "T_c2w", "T_w2c"]:
                # Remap to frame 1
                new_key = (key[0], 1, *key[2:]) if len(key) > 2 else (key[0], 1)
                inputs[new_key] = value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        inputs["target_frame_ids"] = [1]
    else:
        inputs["target_frame_ids"] = []

    return inputs
