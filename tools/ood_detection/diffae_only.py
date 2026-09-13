"""
diffae_only_co3d.py
-------------------
Like diffae_co3d_encoder_depth.py but WITHOUT SplatterImage.
For each test image: DiffAE encode -> decode -> MSE vs input image.
Times only the encode-decode block (not model/data loading).

Saves per-example JSON: {prefix}/{example_id}-diffae_only_result.json
  {example_id, mse_loss, time_ms, run_indist}

Also appends one row to {prefix}/diffae_only_results.csv.
"""

import sys
from infer3d import config as _cfg
ROOT = _cfg.SPLATTER_REPO_ROOT
DIFFAE_ROOT = _cfg.DIFFAE_ROOT
sys.path.append(_cfg.STYLEGAN3_ROOT)
sys.path.append(DIFFAE_ROOT)

import os
import csv
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from filelock import FileLock
from omegaconf import DictConfig
import hydra
import lpips as lpips_lib

from infer3d.data.co3d import CO3DDataset
from templates import *  # co3d_hydrants_autoenc_128, etc.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = None

# ── DiffAE wrapper (identical to full script) ─────────────────────────────────

class DiffAEGenerator(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        model = LitModel(conf)
        state = torch.load(f'{DIFFAE_ROOT}/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)
        model.ema_model.eval()
        model.ema_model.to(device)
        self.model = model

    def forward(self, latent_code, cond, T=12):
        return self.model.render(latent_code, cond, T=T)

    def encode(self, img):
        return self.model.encode(img)


generator_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_generator(cfg):
    if cfg.data.category == "hydrants":
        conf = co3d_hydrants_autoenc_128()
    elif cfg.data.category == "vases":
        conf = co3d_vases_autoenc_128()
    else:
        raise ValueError(f"Unsupported category: {cfg.data.category}")
    gen = DiffAEGenerator(conf)
    gen.eval().to(device)
    return gen


# ── Image loading (identical to full script) ──────────────────────────────────

def load_test_image(test_img_path, size=128):
    image = Image.open(test_img_path).convert("RGB").resize((size, size))
    return np.array(image) / 255.0


def load_test_checkpoint(test_img_path):
    checkpoint_path = test_img_path.replace(".png", ".pth")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


# ── Core timing function ───────────────────────────────────────────────────────

def run_diffae_timed(generator, lpips_fn, image_01, fixed_xT):
    img_enc = generator_transform(image_01).to(device)  # normalise to [-1, 1]

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        cond = generator.encode(img_enc)                        # encode
        decoded = generator(fixed_xT, cond, T=12).clamp(-1, 1) # decode
    torch.cuda.synchronize()
    time_ms = (time.time() - t0) * 1000

    decoded_01 = (decoded + 1) / 2  # [-1,1] -> [0,1]
    if decoded_01.shape[-2:] != image_01.shape[-2:]:
        decoded_01 = F.interpolate(decoded_01, size=image_01.shape[-2:], mode="bilinear", align_corners=False)
    image_01_dev = image_01.to(device)
    mse = F.mse_loss(decoded_01, image_01_dev).item()
    lpips_val = lpips_fn(decoded_01 * 2 - 1, image_01_dev * 2 - 1).item()  # both in [-1,1]
    return mse, lpips_val, time_ms


# ── Per-example entry point ────────────────────────────────────────────────────

def process_one(generator, lpips_fn, fixed_xT, val_dataset, test_img_path, run_indist, log_path, resolution=128):
    parent_n = lambda path, n: path if n <= 0 else parent_n(os.path.dirname(path), n - 1)
    raw_id = os.path.basename(parent_n(test_img_path, 2))
    _, example_id = val_dataset.find_index_for_sequence_prefix(
        raw_id.replace(f"{cfg.data.category[:-1]}_", "")
    )

    result_path = os.path.join(log_path, f"{example_id}-diffae_only_result.json")
    if os.path.exists(result_path):
        print(f"Skipping already done: {example_id}")
        with open(result_path) as f:
            return json.load(f)

    # ── Load input image ──────────────────────────────────────────────────────
    test_checkpoint = load_test_checkpoint(test_img_path)
    ood_image_index = test_checkpoint["ood_image_index"]

    ood_seq_idx, _ = val_dataset.find_index_for_sequence_prefix(example_id)
    ood_data = val_dataset[ood_seq_idx]
    if len(ood_data["focals_pixels"].shape) == 2:
        ood_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in ood_data.items()}

    if run_indist:
        # In-distribution: use the regular CO3D image at that index
        image_np = ood_data["gt_images"][0, ood_image_index].permute(1, 2, 0).numpy()
        image_np = np.clip(image_np, 0, 1)
    else:
        # OOD: use the pre-rendered OOD image from the checkpoint PNG
        image_np = load_test_image(test_img_path, size=resolution)  # [H, W, 3]

    image_01 = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    image_01 = F.interpolate(image_01, size=(resolution, resolution), mode="bilinear", align_corners=False)

    # Warm-up (first call can include CUDA kernel launch overhead)
    with torch.no_grad():
        _dummy = generator.encode(generator_transform(image_01).to(device))

    mse, lpips_val, time_ms = run_diffae_timed(generator, lpips_fn, image_01, fixed_xT)
    combined = mse + 0.5 * lpips_val  # same weights as diffae_abs config

    result = {
        "example_id": example_id,
        "mse_loss": mse,
        "lpips_loss": lpips_val,
        "combined_loss": combined,
        "time_ms": time_ms,
        "run_indist": run_indist,
    }

    with open(result_path, "w") as f:
        json.dump(result, f)

    print(f"  {example_id} | mse={mse:.6f} | lpips={lpips_val:.6f} | combined={combined:.6f} | time={time_ms:.1f}ms | {'ID' if run_indist else 'OOD'}")
    return result


# ── Hydra main ────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path=_cfg.CONFIGS_DIR, config_name="abs_config")
def main(cur_cfg: DictConfig):
    import pandas as pd
    global cfg
    cfg = cur_cfg

    log_path = os.path.join(_cfg.RUNS_ROOT, cfg.general.prefix)
    os.makedirs(log_path, exist_ok=True)
    results_save_path = os.path.join(log_path, "diffae_only_results.csv")

    test_imgs_csv_path = cfg.general.test_imgs_csv_path
    run_indist = cfg.general.run_indist
    resolution = cfg.data.training_resolution

    # Load DiffAE (no SplatterImage)
    generator = load_generator(cfg)
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device).eval()

    # Fixed noise vector for reproducible decoding
    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)
    fixed_xT = torch.randn(1, 3, 128, 128, generator=rng).to(device)

    # Load val/test dataset (needed to map sequence ID -> data)
    _, val_dataset = _load_val_dataset(cfg)

    # Load and filter CSV
    df = pd.read_csv(test_imgs_csv_path)
    df = df[(df["quality"].isin(["good", "med"])) & (df["split"] == "ood")]
    test_imgs_list = [p for p in df["path"].tolist() if os.path.exists(p)]
    assert len(test_imgs_list) > 0, "No test images found"

    if cfg.general.maxsamples is not None:
        test_imgs_list = test_imgs_list[:int(cfg.general.maxsamples)]

    # Split across GPUs if requested
    size_per_split = len(test_imgs_list) // cfg.general.total_splits
    remainder = len(test_imgs_list) % cfg.general.total_splits
    cur_size = size_per_split + (remainder if cfg.general.split == cfg.general.total_splits - 1 else 0)
    cur_list = test_imgs_list[cfg.general.split * size_per_split:
                              cfg.general.split * size_per_split + cur_size]

    print(f"Processing {len(cur_list)} images (split {cfg.general.split}/{cfg.general.total_splits}), run_indist={run_indist}")

    lock_path = results_save_path + ".lock"

    for test_img_path in cur_list:
        try:
            result = process_one(generator, lpips_fn, fixed_xT, val_dataset, test_img_path,
                                  run_indist, log_path, resolution)
            with FileLock(lock_path):
                needs_header = not os.path.exists(results_save_path) or os.path.getsize(results_save_path) == 0
                with open(results_save_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["example_id", "mse_loss", "lpips_loss", "combined_loss", "time_ms", "run_indist"])
                    if needs_header:
                        writer.writeheader()
                    writer.writerow(result)
        except Exception as e:
            print(f"ERROR on {test_img_path}: {e}")
            import traceback; traceback.print_exc()

    print(f"Done. Results in {results_save_path}")


def _load_val_dataset(cfg):
    """Load only val/test dataset (no training dataset needed)."""
    dataset = CO3DDataset(cfg, "train")
    val_dataset = CO3DDataset(cfg, "test")
    return dataset, val_dataset


if __name__ == "__main__":
    main()
