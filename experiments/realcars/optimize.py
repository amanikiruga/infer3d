"""Infer3D search against a single real photograph (RealCars).

The prior (DiffAE) and the lifter (Splatter Image) are both trained only on synthetic
SRN Cars, so pixel losses against a real photo are not usable directly. Supervision is
instead: foreground RGB, a depth-aware background term, scale-and-shift aligned
monocular depth, and DINOv2 patch-feature cosine similarity -- the last is what lets the
search ignore real-world texture the synthetic prior cannot represent.

Every `save_every` iterations this dumps a PNG grid (input vs best vs top-k) and appends
to a JSON loss trajectory, so a run can be inspected while it is going.

Usage (one scene; run.sh loops over all 20):
  CUDA_VISIBLE_DEVICES=0 python -u experiments/realcars/optimize.py \
    abs=diffae_abs +dataset=cars general.split=0 general.total_splits=1 \
    general.prefix=realcars_opt \
    opt.pretrained_ckpt=$LIFTER_DIR/srn_cars.pth \
    +real_ood.image_path=$REALCARS_ROOT/HQ339/<scene>/frame_00000.jpg \
    +real_ood.out_subdir=00 \
    +real_ood.num_iterations=783 +real_ood.save_every=50 \
    +real_ood.narrow_s1_end=122 +real_ood.narrow_s2_end=302 \
    +real_ood.lambda_rgb=1.0 +real_ood.lambda_bg=0.5 +real_ood.lambda_depth=0.5 \
    +real_ood.lambda_dino=0.5 +real_ood.lambda_lpips=0.5 \
    +real_ood.w_reg=0.025 +real_ood.rank_w_reg_mult=1.0

Writes $INFER3D_RUNS_ROOT/realcars_opt/<out_subdir>/final.pth plus PNG panels and a
JSON loss trajectory.

This script intentionally exits after one image so we can iterate fast; pass a
*comma-separated* `+real_ood.image_path=a,b,c` to run multiple sequentially.
"""

import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from infer3d import config as _cfg

DIFFAE_ROOT = _cfg.DIFFAE_ROOT
sys.path.append(DIFFAE_ROOT)          # we import the DiffAE model definitions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import hydra
import imageio
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from infer3d.utils.optim import (
    get_mse_loss, get_random_cameras, render_with_custom_camera_align,
    symmetric_orthogonalization,
)
from infer3d.model import GaussianSplatPredictor
from templates import srn_cars_train_autoenc

# DiffAE LitModel
from experiment import LitModel  # type: ignore  (lives in DIFFAE_ROOT)

from real_ood_inputs import build_inputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Models ───────────────────────────────────────────────────────────────────

class DiffAEGenerator(nn.Module):
    def __init__(self, conf, checkpoint_path=None):
        super().__init__()
        self.conf = conf
        model = LitModel(conf)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu")
        else:
            state = torch.load(f"{DIFFAE_ROOT}/checkpoints/{conf.name}/last.ckpt",
                               map_location="cpu")
        model.load_state_dict(state["state_dict"], strict=False)
        model.ema_model.eval().to(device)
        self.model = model

    def forward(self, latent_code, cond=None, T=12):
        if cond is None:
            cond = torch.randn(1, 512, device=device)
        return self.model.render(latent_code, cond, T=T)

    def encode(self, img):
        return self.model.encode(img)


def load_models(cfg):
    gp = GaussianSplatPredictor(cfg).to(memory_format=torch.channels_last).to(device)
    if cfg.opt.pretrained_ckpt is not None:
        ckpt = torch.load(cfg.opt.pretrained_ckpt, map_location=device, weights_only=False)
        gp.load_state_dict(ckpt["model_state_dict"])
        print(f"[real_ood] loaded splatter image ckpt: {cfg.opt.pretrained_ckpt}")
    gp.eval()
    gen = DiffAEGenerator(srn_cars_train_autoenc()).eval().to(device)
    for p in gen.parameters():
        p.requires_grad = False
    return gp, gen


def load_train_dataset(cfg):
    from infer3d.data.srn import SRNDataset
    return SRNDataset(cfg, "train", data_category="cars")


# ── Real-image losses ────────────────────────────────────────────────────────

def masked_mse(pred, target, mask):
    """pred,target: [B,C,H,W]; mask: [1,1,H,W] or [B,1,H,W]. Per-batch mean
    over masked pixels."""
    B = pred.shape[0]
    if mask.shape[0] == 1 and B > 1:
        mask = mask.expand(B, -1, -1, -1)
    diff2 = (pred - target) ** 2                     # [B, C, H, W]
    diff2 = diff2.mean(dim=1, keepdim=True)          # [B, 1, H, W]
    weight = mask.clamp(0, 1)
    num = (diff2 * weight).sum(dim=(1, 2, 3))
    den = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return num / den                                  # [B]


def masked_mse_white(pred, mask_safe_bg):
    """Encourage pred to be white in safe-bg pixels."""
    B = pred.shape[0]
    target = torch.ones_like(pred)
    return masked_mse(pred, target, mask_safe_bg)


def fit_scale_shift_torch(x, y, w, n_iters: int = 30):
    """Differentiable-ish scale+shift fit: min_{s,t} sum_i w_i*(s*x_i + t - y_i)^2.
    Closed-form per-batch element. x,y,w: [B,1,H,W]. Returns s,t: [B,1,1,1] each."""
    B = x.shape[0]
    w = w.clamp_min(0)
    Wsum = w.sum(dim=(1, 2, 3))                       # [B]
    if (Wsum < 10).any():
        # Degenerate; just use no-scale-no-shift.
        s = torch.ones(B, 1, 1, 1, device=x.device)
        t = torch.zeros(B, 1, 1, 1, device=x.device)
        return s, t
    mx = (w * x).sum(dim=(1, 2, 3)) / Wsum
    my = (w * y).sum(dim=(1, 2, 3)) / Wsum
    vxx = (w * (x - mx[:, None, None, None]) ** 2).sum(dim=(1, 2, 3)) / Wsum
    vxy = (w * (x - mx[:, None, None, None]) * (y - my[:, None, None, None])).sum(dim=(1, 2, 3)) / Wsum
    s = (vxy / vxx.clamp_min(1e-8)).view(B, 1, 1, 1)
    t = (my - s.view(B) * mx).view(B, 1, 1, 1)
    return s, t


def aligned_inv_depth_loss(gs_invdepth, da3_invdepth, mask):
    """Mask-only scale+shift align GS inv-depth onto DA3 inv-depth, then MSE.
    Both are [B,1,H,W]; mask [1,1,H,W] or [B,...]."""
    s, t = fit_scale_shift_torch(gs_invdepth.detach(), da3_invdepth, mask)
    aligned = s * gs_invdepth + t
    return masked_mse(aligned, da3_invdepth, mask), s.squeeze(), t.squeeze()


# ── DINOv2 cosine loss (lazy load) ───────────────────────────────────────────

DINO_INPUT_SIZE = 518          # 518/14 = 37 → 37×37 patch grid (matches inputs cache)

_DINO_PROC = None
_DINO_MODEL = None
_DINO_NORM_MEAN = None
_DINO_NORM_STD = None


def _ensure_dino():
    global _DINO_PROC, _DINO_MODEL, _DINO_NORM_MEAN, _DINO_NORM_STD
    if _DINO_MODEL is None:
        from transformers import AutoImageProcessor, AutoModel
        print("[real_ood] loading DINOv2 …")
        _DINO_PROC = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        _DINO_MODEL = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        for p in _DINO_MODEL.parameters():
            p.requires_grad = False
        # normalisation as in BitImageProcessor
        _DINO_NORM_MEAN = torch.tensor(_DINO_PROC.image_mean, device=device).view(1, 3, 1, 1)
        _DINO_NORM_STD = torch.tensor(_DINO_PROC.image_std, device=device).view(1, 3, 1, 1)
    return _DINO_MODEL


@torch.no_grad()
def dino_features_no_grad(rgb_01: torch.Tensor):
    """rgb_01: [B,3,H,W] in [0,1]. Returns patch tokens [B, S, C]."""
    _ensure_dino()
    x = F.interpolate(rgb_01, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
                      mode="bilinear", align_corners=False)
    x = (x - _DINO_NORM_MEAN) / _DINO_NORM_STD
    return _DINO_MODEL(pixel_values=x).last_hidden_state[:, 1:, :]


def dino_features_with_grad(rgb_01: torch.Tensor):
    """Same but keeps graph for backprop into renders."""
    _ensure_dino()
    x = F.interpolate(rgb_01, size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
                      mode="bilinear", align_corners=False)
    x = (x - _DINO_NORM_MEAN) / _DINO_NORM_STD
    return _DINO_MODEL(pixel_values=x).last_hidden_state[:, 1:, :]


def dino_cosine_loss(render_01, ood_dino_grid):
    """render_01: [B,3,H,W]; ood_dino_grid: [1, C, h, w] (cached). Returns [B]."""
    feat = dino_features_with_grad(render_01)         # [B, S, C]
    B, S, C = feat.shape
    h = w = int(round(S ** 0.5))
    pred_grid = feat.reshape(B, h, w, C).permute(0, 3, 1, 2)
    target_grid = ood_dino_grid.to(device)            # [1, C, h, w]
    pred_n = F.normalize(pred_grid, dim=1)
    targ_n = F.normalize(target_grid, dim=1)
    cos = (pred_n * targ_n).sum(dim=1)                # [B, h, w]
    return (1.0 - cos).mean(dim=(1, 2))               # [B]


# ── LPIPS loss (lazy load) ───────────────────────────────────────────────────

_LPIPS_FN = None


def _ensure_lpips():
    global _LPIPS_FN
    if _LPIPS_FN is None:
        import lpips as lpips_lib
        print("[real_ood] loading LPIPS (vgg) …")
        _LPIPS_FN = lpips_lib.LPIPS(net='vgg').to(device).eval()
        for p in _LPIPS_FN.parameters():
            p.requires_grad = False
    return _LPIPS_FN


def lpips_loss(pred_01, target_01):
    """No mask, matches heavy. pred_01,target_01: [B,3,H,W] in [0,1]. Returns [B]."""
    fn = _ensure_lpips()
    return fn(pred_01 * 2 - 1, target_01 * 2 - 1).view(-1)


# ── Visualisation helpers ────────────────────────────────────────────────────

def _depth_vis(d_np, lo=None, hi=None):
    if lo is None or hi is None:
        m = np.isfinite(d_np) & (d_np > 0)
        if not m.any():
            return np.zeros((*d_np.shape, 3), dtype=np.uint8)
        lo, hi = np.percentile(d_np[m], [2, 98])
    n = np.clip((d_np - lo) / max(hi - lo, 1e-6), 0, 1)
    return (cm.plasma(n)[..., :3] * 255).astype(np.uint8)


def _to_uint8_chw(t01):
    return (t01.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def _title(img, text, h=20):
    bar = np.full((h, img.shape[1], 3), 25, dtype=np.uint8)
    cv2.putText(bar, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1)
    return np.concatenate([bar, img], axis=0)


def save_progress_grid(out_path, *,
                       ood_rgb_t, ood_mask_t, ood_depth_t, ood_bgsafe_t,
                       best_render_t, best_invdepth_t, best_aligned_invdepth_t,
                       top_k_render_ts, top_k_input_ts, top_k_losses,
                       iteration, best_loss, breakdown):
    """A 2-row composite. Saves PNG."""
    ood_rgb = _to_uint8_chw(ood_rgb_t[0])
    ood_mask_rgb = (np.repeat((ood_mask_t[0, 0].cpu().numpy() * 255).astype(np.uint8)[..., None], 3, -1))
    bg_rgb = (np.repeat((ood_bgsafe_t[0, 0].cpu().numpy() * 255).astype(np.uint8)[..., None], 3, -1))
    ood_depth_vis = _depth_vis(1.0 / ood_depth_t[0, 0].cpu().numpy().clip(min=1e-6))
    best_render = _to_uint8_chw(best_render_t)
    best_inv = _depth_vis(best_invdepth_t[0, 0].cpu().numpy())
    best_aligned = _depth_vis(best_aligned_invdepth_t[0, 0].cpu().numpy())

    row1 = np.concatenate([
        _title(ood_rgb,        "OOD rgb_128"),
        _title(ood_mask_rgb,   "mask"),
        _title(bg_rgb,         "bg_safe"),
        _title(ood_depth_vis,  "OOD inv-depth (DA3)"),
        _title(best_render,    f"best render  loss={best_loss:.4f}"),
        _title(best_inv,       "GS inv-depth (raw)"),
        _title(best_aligned,   "GS inv-depth (aligned)"),
    ], axis=1)

    if len(top_k_render_ts) > 0:
        ncol = min(5, len(top_k_render_ts))
        renders = torch.stack([t.cpu() for t in top_k_render_ts[:ncol]])
        inputs = torch.stack([t.cpu() for t in top_k_input_ts[:ncol]])
        renders_grid = vutils.make_grid(renders.clamp(0, 1), nrow=ncol).permute(1, 2, 0).numpy()
        renders_grid = (renders_grid * 255).clip(0, 255).astype(np.uint8)
        inputs_grid = vutils.make_grid(inputs.clamp(0, 1), nrow=ncol).permute(1, 2, 0).numpy()
        inputs_grid = (inputs_grid * 255).clip(0, 255).astype(np.uint8)

        # match width to row1
        def _resize_w(im, w):
            h_new = int(round(im.shape[0] * w / im.shape[1]))
            return cv2.resize(im, (w, h_new), interpolation=cv2.INTER_AREA)
        target_w = row1.shape[1]
        ig = _resize_w(_title(inputs_grid, "top-k DiffAE inputs"), target_w)
        rg = _resize_w(_title(renders_grid, f"top-k renders  losses={[f'{x:.3f}' for x in top_k_losses[:ncol]]}"), target_w)
        row2 = np.concatenate([ig, rg], axis=0)

        max_w = max(row1.shape[1], row2.shape[1])
        def _padw(r, w):
            if r.shape[1] >= w: return r
            return np.concatenate([r, np.zeros((r.shape[0], w - r.shape[1], 3), dtype=np.uint8)], axis=1)
        grid = np.concatenate([_padw(row1, max_w), _padw(row2, max_w)], axis=0)
    else:
        grid = row1

    # bottom strip with the loss breakdown
    bk_h = 22
    bk = np.full((bk_h, grid.shape[1], 3), 15, dtype=np.uint8)
    text = f"iter={iteration}  " + "  ".join([f"{k}={v:.4f}" for k, v in breakdown.items()])
    cv2.putText(bk, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    grid = np.concatenate([grid, bk], axis=0)
    imageio.imwrite(out_path, grid)


def _tokens_to_grid(tok):
    """[1, S, C] -> [1, C, h, w] with h=w=sqrt(S)."""
    S = tok.shape[1]; C = tok.shape[2]
    h = w = int(round(S ** 0.5))
    return tok.reshape(1, h, w, C).permute(0, 3, 1, 2).contiguous()


def _joint_dino_pca(grids, fg_mask_t=None):
    """grids: list of [1, C, h, w] tensors (any device). fg_mask_t: optional
    [1,1,H,W] float in [0,1] — the SAM FG mask. If provided, PCA is fit on
    FG-only tokens (downsampled mask, threshold 0.5) and BG patches rendered
    black; FG-only min-max keeps colour budget on parts (DINOv2-paper style).
    Returns a list of [h, w, 3] uint8 arrays in matched colour space."""
    Cdim = grids[0].shape[1]
    flat, sizes, per_grid_fg = [], [], []
    for g in grids:
        h, w = g.shape[2], g.shape[3]
        flat.append(g.detach().to("cpu").reshape(Cdim, -1).numpy())
        sizes.append((h, w))
        if fg_mask_t is not None:
            m = F.interpolate(fg_mask_t.float().to("cpu"), size=(h, w),
                              mode="bilinear", align_corners=False)[0, 0].numpy()
            per_grid_fg.append(m > 0.5)
        else:
            per_grid_fg.append(np.ones((h, w), dtype=bool))

    F_full = np.concatenate(flat, axis=1)                  # [C, N]
    fg_concat = np.concatenate([f.flatten() for f in per_grid_fg], axis=0)

    # Fit PCA on FG tokens only (paper-style); fall back to all if mask too small.
    if fg_concat.sum() >= 16:
        F_fit = F_full[:, fg_concat]
    else:
        F_fit = F_full
    fit_mean = F_fit.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(F_fit - fit_mean, full_matrices=False)
    pcs = U[:, :3].T @ (F_full - fit_mean)                 # [3, N]

    # Normalize using FG-only range so all colour budget is on parts.
    ref = pcs[:, fg_concat] if fg_concat.sum() >= 16 else pcs
    pmin = ref.min(axis=1, keepdims=True)
    pmax = ref.max(axis=1, keepdims=True)
    pcs = np.clip((pcs - pmin) / (pmax - pmin + 1e-6), 0, 1)

    out, cur = [], 0
    for (h, w), fg_m in zip(sizes, per_grid_fg):
        n = h * w
        rgb = pcs[:, cur:cur + n].reshape(3, h, w).transpose(1, 2, 0)
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        if fg_mask_t is not None:
            rgb = rgb * fg_m[..., None]                    # BG → black
        out.append(rgb)
        cur += n
    return out


def save_wandb_panel(out_path, *,
                     ood_rgb_t, ood_depth_t, ood_dino_grid,
                     best_render_t, best_input_t, best_aligned_invdepth_t,
                     top_k_render_ts, top_k_input_ts, top_k_invdepth_ts,
                     top_k_losses,
                     da3_inv, mask,
                     iteration, best_loss):
    """Composite: row-GT (image, inv-depth, DINO) + row-best (input, render,
    inv-depth, DINO) + 4 top-k rows (input, render, inv-depth, DINO).
    Shared inv-depth percentiles taken from GT; DINO PCA basis fit jointly."""
    K = len(top_k_render_ts)

    # 1. depth scale from GT inv-depth (this is what aligned preds target)
    gt_inv_np = (1.0 / ood_depth_t[0, 0].clamp_min(1e-6)).cpu().numpy()
    finite = np.isfinite(gt_inv_np) & (gt_inv_np > 0)
    if finite.any():
        lo, hi = np.percentile(gt_inv_np[finite], [2, 98])
    else:
        lo, hi = 0.0, 1.0

    # 2. align top-k inv-depths to GT (closed-form scale+shift on mask)
    top_k_aligned = []
    for invd in top_k_invdepth_ts:
        x = invd.unsqueeze(0).to(device) if invd.dim() == 3 else invd.to(device)
        s, t = fit_scale_shift_torch(x.detach(), da3_inv, mask)
        top_k_aligned.append((s * x + t)[0, 0].detach().cpu().numpy())

    # 3. DINO features for best + top-k renders, then joint PCA with GT
    with torch.no_grad():
        best_tok = dino_features_no_grad(best_render_t.to(device).unsqueeze(0))
        best_grid = _tokens_to_grid(best_tok)
        topk_grids = [_tokens_to_grid(dino_features_no_grad(r.to(device).unsqueeze(0)))
                      for r in top_k_render_ts]
    pca_inputs = [ood_dino_grid, best_grid] + topk_grids
    pca_rgbs = _joint_dino_pca(pca_inputs, fg_mask_t=mask)
    gt_dino_rgb = pca_rgbs[0]
    best_dino_rgb = pca_rgbs[1]
    topk_dino_rgbs = pca_rgbs[2:]

    # 4. assemble cells (resize DINO/depth grids up to 128 for legibility)
    def _up128(im):
        if im.shape[:2] == (128, 128):
            return im
        return cv2.resize(im, (128, 128), interpolation=cv2.INTER_NEAREST)

    ood_rgb = _to_uint8_chw(ood_rgb_t[0])
    gt_dep_rgb = _depth_vis(gt_inv_np, lo=lo, hi=hi)
    row_gt = np.concatenate([
        _title(ood_rgb,                "GT image"),
        _title(gt_dep_rgb,             f"GT inv-depth  scale=[{lo:.3f},{hi:.3f}]"),
        _title(_up128(gt_dino_rgb),    "GT DINO PCA"),
    ], axis=1)

    best_input_rgb = _to_uint8_chw(best_input_t)
    best_render_rgb = _to_uint8_chw(best_render_t)
    best_dep_rgb = _depth_vis(best_aligned_invdepth_t[0, 0].cpu().numpy(), lo=lo, hi=hi)
    row_best = np.concatenate([
        _title(best_input_rgb,         "best DiffAE input"),
        _title(best_render_rgb,        f"best render  loss={best_loss:.4f}"),
        _title(best_dep_rgb,           "best inv-depth (aligned, shared)"),
        _title(_up128(best_dino_rgb),  "best DINO PCA"),
    ], axis=1)

    if K > 0:
        cells_in, cells_re, cells_de, cells_di = [], [], [], []
        for i in range(K):
            cells_in.append(_to_uint8_chw(top_k_input_ts[i]))
            cells_re.append(_to_uint8_chw(top_k_render_ts[i]))
            cells_de.append(_depth_vis(top_k_aligned[i], lo=lo, hi=hi))
            cells_di.append(_up128(topk_dino_rgbs[i]))
        topk_in_row = _title(np.concatenate(cells_in, axis=1), "top-k DiffAE inputs")
        topk_re_row = _title(np.concatenate(cells_re, axis=1),
                             f"top-k renders  losses={[f'{x:.3f}' for x in top_k_losses[:K]]}")
        topk_de_row = _title(np.concatenate(cells_de, axis=1), "top-k inv-depth (aligned, shared)")
        topk_di_row = _title(np.concatenate(cells_di, axis=1), "top-k DINO PCA")
        rows = [row_gt, row_best, topk_in_row, topk_re_row, topk_de_row, topk_di_row]
    else:
        rows = [row_gt, row_best]

    max_w = max(r.shape[1] for r in rows)
    def _padw(r):
        if r.shape[1] >= max_w: return r
        return np.concatenate([r, np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)], axis=1)
    grid = np.concatenate([_padw(r) for r in rows], axis=0)

    bk = np.full((22, grid.shape[1], 3), 15, dtype=np.uint8)
    cv2.putText(bk, f"iter={iteration}  best_loss={best_loss:.4f}  K={K}",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    grid = np.concatenate([grid, bk], axis=0)
    imageio.imwrite(out_path, grid)


def save_loss_curves(out_path, trajectory):
    """Plot loss vs iteration."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if not trajectory:
        plt.close(fig); return
    keys = [k for k in trajectory[0].keys() if k not in ("iteration", "elapsed_s")]
    for k in keys:
        ys = [d.get(k, np.nan) for d in trajectory]
        ax.plot([d["iteration"] for d in trajectory], ys, label=k, linewidth=1)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.set_yscale("log")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# ── Lambda schedule ──────────────────────────────────────────────────────────

def _build_lambda_schedule(stages_cfg, defaults, num_iterations):
    """Return a list of (end_iter_exclusive, {rgb, bg, depth, dino}) stages.
    `defaults` is a dict with all 4 keys. If `stages_cfg` is empty/None, a
    single stage covering [0, num_iterations) with `defaults` is returned
    (= today's behaviour). Within a user-provided stage, missing keys fall
    back to `defaults`."""
    if stages_cfg is None or len(stages_cfg) == 0:
        return [(num_iterations, dict(defaults))]
    if hasattr(stages_cfg, "_content") or isinstance(stages_cfg, (list,)) and stages_cfg and hasattr(stages_cfg[0], "_content"):
        stages_cfg = OmegaConf.to_container(stages_cfg, resolve=True)
    out = []
    for s in stages_cfg:
        out.append((int(s["end"]), {
            "rgb":   float(s.get("rgb",   defaults["rgb"])),
            "bg":    float(s.get("bg",    defaults["bg"])),
            "depth": float(s.get("depth", defaults["depth"])),
            "dino":  float(s.get("dino",  defaults["dino"])),
            "lpips": float(s.get("lpips", defaults["lpips"])),
        }))
    out.sort(key=lambda x: x[0])
    return out


def _current_lambdas(iteration, schedule):
    for end, lams in schedule:
        if iteration < end:
            return lams
    return schedule[-1][1]


# ── Search loop ──────────────────────────────────────────────────────────────

def run_one(cfg, image_path: str, out_dir: Path, train_dataset,
            gaussian_predictor, generator,
            num_iterations: int = 120,
            save_every: int = 20,
            num_latents: int = None,
            num_rotations: int = None,
            batch_size: int = None,
            lambda_rgb: float = 1.0,
            lambda_bg: float = 0.5,
            lambda_depth: float = 0.5,
            lambda_dino: float = 0.5,
            lambda_lpips: float = 0.0,
            lambda_dino_warmup_iters: int = 0,
            fast_T_optim: int = 6,
            fast_T_selection: int = 6,
            seed: int = 0,
            wandb_run=None,
            lambda_schedule=None,
            narrow_s1_end: int = 4,
            narrow_s2_end: int = None,
            w_reg_override: float = None,
            rank_w_reg_mult: float = 1.0):
    """Search for the best (latent, rotation) explaining this OOD image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[real_ood] === {image_path} ===")
    print(f"[real_ood] out_dir = {out_dir}")

    # Lambda schedule (stage-based; default = single stage = constants)
    default_lambdas = {"rgb": lambda_rgb, "bg": lambda_bg,
                       "depth": lambda_depth, "dino": lambda_dino,
                       "lpips": lambda_lpips}
    if lambda_schedule is None:
        lambda_schedule = [(num_iterations, dict(default_lambdas))]
    print(f"[real_ood] lambda schedule ({len(lambda_schedule)} stage(s)):")
    for end, lams in lambda_schedule:
        print(f"           up to iter {end}: {lams}")

    # Narrowing schedule: hardcoded thresholds → configurable knobs
    if narrow_s2_end is None or narrow_s2_end <= 0:
        narrow_s2_end = num_iterations // 3
    print(f"[real_ood] narrowing: iter 0=all, [1,{narrow_s1_end})=pin+top32, "
          f"[{narrow_s1_end},{narrow_s2_end})=pin+top-max(b,12), "
          f"[{narrow_s2_end},{num_iterations})=pin+top-b")

    # 1. Inputs (cached)
    cache_dir = out_dir / "inputs_cache"
    bundle = build_inputs(image_path, str(cache_dir))
    rgb_white = bundle["rgb_128"].to(device)              # [1,3,128,128]
    mask = bundle["mask_128"].to(device)                  # [1,1,128,128]
    bg_safe = bundle["bg_safe_128"].to(device)            # [1,1,128,128]
    da3_depth = bundle["da3_depth_128"].to(device)        # [1,1,128,128] metric m
    da3_inv = 1.0 / da3_depth.clamp_min(1e-6)             # inv-depth ↑ = closer
    dino_target_grid = bundle["dino_feat_grid"].to(device)

    # 2. Seeding
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)

    # 3. Training-image latent proposals
    num_latents = num_latents or int(cfg.abs.num_latents)
    num_rotations = num_rotations or int(cfg.abs.num_rotations)
    batch_size = batch_size or int(cfg.abs.batch_size)
    print(f"[real_ood] num_latents={num_latents}  num_rotations={num_rotations}  batch_size={batch_size}")

    dataloader = DataLoader(train_dataset, batch_size=num_latents, shuffle=True, generator=g)
    train_data = next(iter(dataloader))
    train_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in train_data.items()}

    # The DiffAE encoder transform expects a 128x128 RGB normalised to mean .5 / std .5.
    gen_transform = transforms.Compose([
        transforms.Resize(128), transforms.CenterCrop(128),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_image_for_enc = gen_transform(train_data["gt_images"][:, 0]).to(device)
    with torch.no_grad():
        init_train = generator.encode(train_image_for_enc)            # [num_latents, 512]
    ood_for_enc = gen_transform(rgb_white).to(device)
    with torch.no_grad():
        init_ood = generator.encode(ood_for_enc)                      # [1, 512]

    total_rotations = num_rotations + 1  # random + identity
    total_candidates = total_rotations * num_latents + 1              # +1 for OOD@identity
    print(f"[real_ood] total_candidates={total_candidates}  (rotations*latents + ood-at-identity)")

    fixed_xT = torch.randn(total_candidates, 3, 128, 128, device=device)

    train_latents_repeated = [init_train.clone().detach() for _ in range(total_rotations)]
    train_latents_all = torch.cat(train_latents_repeated, dim=0)      # [tr*nl, 512]
    search_cond = nn.Parameter(torch.cat([train_latents_all, init_ood.clone().detach()], dim=0))
    pinned_idx = total_rotations * num_latents                        # last index = OOD+identity
    w_avg = init_train.mean(dim=0).detach()

    # rotations
    rand_rots = get_random_cameras(num_rotations, zgt=1.3, device=device)
    identity_rot = torch.eye(3, device=device).unsqueeze(0)
    all_rots = torch.cat([rand_rots, identity_rot], dim=0)            # [tr, 3, 3]
    train_rots = torch.stack([all_rots.clone() for _ in range(num_latents)], dim=1)
    train_rots = train_rots.reshape(-1, 3, 3)
    rotations = nn.Parameter(torch.cat([train_rots, identity_rot.clone()], dim=0))   # [tc, 3, 3]
    translations = nn.Parameter(torch.zeros(total_candidates, 3, device=device))

    optimizer = optim.Adam([
        {"params": [rotations],     "lr": cfg.abs.lr_rotations},
        {"params": [translations],  "lr": cfg.abs.lr_rotations},
        {"params": [search_cond],   "lr": cfg.abs.w_lr},
    ])

    background = torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0],
                              dtype=torch.float32, device=device)

    # State for top-k tracking & best
    top_k = min(8, total_candidates)
    top_k_renders = []; top_k_inputs = []; top_k_invdepths = []; top_k_losses = []
    top_k_idxs = []; top_k_rotations = []; top_k_translations = []
    best = {"loss": float("inf"), "render": None, "input": None,
            "invdepth": None, "aligned_invdepth": None,
            "rotation": None, "translation": None, "cond": None,
            "iteration": -1, "candidate_idx": -1}
    trajectory = []

    used_idxs = list(range(total_candidates))

    zgt = 1.3
    sorted_idxs_by_loss = []

    def _set_used(it):
        """Pick which candidates to evaluate for this iteration."""
        nonlocal used_idxs
        if it == 0:
            used_idxs = list(range(total_candidates))
        elif it < narrow_s1_end:
            used_idxs = sorted({pinned_idx, *sorted_idxs_by_loss[:32]})
        elif it < narrow_s2_end:
            keep = sorted_idxs_by_loss[:max(batch_size, 12)]
            used_idxs = sorted({pinned_idx, *keep})
        else:
            keep = sorted_idxs_by_loss[:batch_size]
            used_idxs = sorted({pinned_idx, *keep})

    print(f"[real_ood] starting search for {num_iterations} iters")
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()
    pbar = tqdm(total=num_iterations, desc="iter")

    for iteration in range(num_iterations):
        _set_used(iteration)
        all_losses_and_idxs = []
        n_batches = math.ceil(len(used_idxs) / batch_size)

        # Stage-based lambdas (default = constants). λ_dino warm-up still applies on top.
        lams = _current_lambdas(iteration, lambda_schedule)
        cur_lambda_rgb   = lams["rgb"]
        cur_lambda_bg    = lams["bg"]
        cur_lambda_depth = lams["depth"]
        cur_lambda_dino  = lams["dino"] if iteration >= lambda_dino_warmup_iters else 0.0
        cur_lambda_lpips = lams.get("lpips", 0.0)

        for b in range(n_batches):
            cur_idxs = used_idxs[b * batch_size: (b + 1) * batch_size]
            if not cur_idxs:
                continue
            optimizer.zero_grad()

            cur_cond = search_cond[cur_idxs]
            cur_rots = rotations[cur_idxs]
            cur_trans = translations[cur_idxs]
            cur_xT = fixed_xT[cur_idxs]

            selection_only = (iteration == 0)
            if iteration > 0 and cfg.abs.orthogonalize_rotations:
                cur_rots = symmetric_orthogonalization(cur_rots.view(-1, 9))
            # Pin OOD-at-identity
            if pinned_idx in cur_idxs:
                pi = cur_idxs.index(pinned_idx)
                cur_rots = cur_rots.clone()
                cur_trans = cur_trans.clone()
                cur_rots[pi] = torch.eye(3, device=device)
                cur_trans[pi] = torch.zeros(3, device=device)

            B = cur_cond.shape[0]
            T = fast_T_selection if selection_only else fast_T_optim
            if selection_only:
                with torch.no_grad():
                    decoded = generator(cur_xT, cur_cond, T=T).clamp(0, 1)
            else:
                decoded = generator(cur_xT, cur_cond, T=T).clamp(0, 1)
            input_for_pred = decoded.unsqueeze(1).to(device)            # [B,1,3,128,128]

            # SRN: no origin distances, no focals
            splats = gaussian_predictor(
                input_for_pred,
                train_data["view_to_world_transforms"][:1, :cfg.data.input_images, ...].repeat(B, 1, 1, 1),
                train_data["source_cv2wT_quat"][:1, :cfg.data.input_images].repeat(B, 1, 1),
                None,
            )

            renders, inv_depths = [], []
            for i in range(B):
                splats_i = {k: v[i] for k, v in splats.items()}
                rend, invd = render_with_custom_camera_align(
                    splats_i, background, cfg, None,
                    cur_rots[i], zgt,
                    device=device, translation=cur_trans[i],
                    zgt_ood=zgt, return_depth=True,
                )
                renders.append(rend)
                inv_depths.append(invd)
            renders = torch.cat(renders, dim=0)              # [B,3,128,128]
            inv_depths = torch.cat(inv_depths, dim=0)        # [B,1,128,128]

            # ── Losses
            mse_fg = masked_mse(renders, rgb_white.expand_as(renders), mask)
            mse_bg = masked_mse_white(renders, bg_safe)
            depth_loss, _, _ = aligned_inv_depth_loss(inv_depths, da3_inv, mask)
            if cur_lambda_dino > 0:
                dino_loss = dino_cosine_loss(renders, dino_target_grid)
            else:
                dino_loss = torch.zeros(B, device=device)
            if cur_lambda_lpips > 0:
                lpips_per = lpips_loss(renders, rgb_white.expand_as(renders))
            else:
                lpips_per = torch.zeros(B, device=device)

            losses_per = (cur_lambda_rgb * mse_fg + cur_lambda_bg * mse_bg
                          + cur_lambda_depth * depth_loss
                          + cur_lambda_dino * dino_loss
                          + cur_lambda_lpips * lpips_per)
            # w-reg, tied to the same per-candidate objective
            eff_w_reg = float(cfg.abs.w_reg) if w_reg_override is None else float(w_reg_override)
            w_reg_per = eff_w_reg * (cur_cond - w_avg).norm(dim=1)
            total_per = losses_per + w_reg_per
            total = total_per.sum()

            if not selection_only:
                total.backward()
                optimizer.step()

            with torch.no_grad():
                # Include w_reg in rank so OOD/degenerate latents are not picked as "best".
                rank_metric = (mse_fg + mse_bg + depth_loss + dino_loss + lpips_per
                               + rank_w_reg_mult * w_reg_per).detach()
                for i, ci in enumerate(cur_idxs):
                    all_losses_and_idxs.append((ci, float(losses_per[i].item())))
                    rm_i = float(rank_metric[i].item())
                    # top-k
                    if len(top_k_losses) < top_k:
                        top_k_renders.append(renders[i].detach()); top_k_inputs.append(decoded[i].detach())
                        top_k_invdepths.append(inv_depths[i].detach())
                        top_k_losses.append(rm_i); top_k_idxs.append(ci)
                        top_k_rotations.append(cur_rots[i].detach()); top_k_translations.append(cur_trans[i].detach())
                    elif ci in top_k_idxs:
                        ei = top_k_idxs.index(ci)
                        if rm_i < top_k_losses[ei]:
                            top_k_losses[ei] = rm_i
                            top_k_renders[ei] = renders[i].detach(); top_k_inputs[ei] = decoded[i].detach()
                            top_k_invdepths[ei] = inv_depths[i].detach()
                            top_k_rotations[ei] = cur_rots[i].detach(); top_k_translations[ei] = cur_trans[i].detach()
                    else:
                        worst = int(np.argmax(top_k_losses))
                        if rm_i < top_k_losses[worst]:
                            top_k_losses[worst] = rm_i; top_k_idxs[worst] = ci
                            top_k_renders[worst] = renders[i].detach(); top_k_inputs[worst] = decoded[i].detach()
                            top_k_invdepths[worst] = inv_depths[i].detach()
                            top_k_rotations[worst] = cur_rots[i].detach(); top_k_translations[worst] = cur_trans[i].detach()

                    if rm_i < best["loss"]:
                        # also recompute aligned inv-depth for the best for nice viz
                        s, t = fit_scale_shift_torch(inv_depths[i:i+1].detach(), da3_inv, mask)
                        aligned = (s * inv_depths[i:i+1].detach() + t)
                        best.update(dict(
                            loss=rm_i, render=renders[i].detach().cpu(),
                            input=decoded[i].detach().cpu(),
                            invdepth=inv_depths[i:i+1].detach().cpu(),
                            aligned_invdepth=aligned.detach().cpu(),
                            rotation=cur_rots[i].detach().cpu(),
                            translation=cur_trans[i].detach().cpu(),
                            cond=cur_cond[i].detach().cpu(),
                            iteration=iteration, candidate_idx=ci,
                            breakdown={
                                "rgb": float(mse_fg[i].item()),
                                "bg":  float(mse_bg[i].item()),
                                "dep": float(depth_loss[i].item()),
                                "din": float(dino_loss[i].item()),
                                "lpi": float(lpips_per[i].item()),
                                "tot": rm_i,
                            },
                        ))

        # global ranking for next iter's `_set_used`
        sorted_idxs_by_loss = [i for i, _ in sorted(all_losses_and_idxs, key=lambda x: x[1])]

        # trajectory log — record both current-iter mean (what's happening NOW)
        # and best-so-far. The best breakdown can be stale if DINO warm-up just
        # ended; the per-iter mean shows the live optimisation surface.
        with torch.no_grad():
            iter_losses = sorted(all_losses_and_idxs, key=lambda x: x[1])[:5]
            iter_mean_top5 = float(np.mean([x[1] for x in iter_losses])) if iter_losses else float("nan")
            traj_entry = {
                "iteration": iteration,
                "elapsed_s": time.time() - start_time,
                "best_total": best["loss"],
                "best_rgb":   best.get("breakdown", {}).get("rgb", float("nan")),
                "best_bg":    best.get("breakdown", {}).get("bg",  float("nan")),
                "best_dep":   best.get("breakdown", {}).get("dep", float("nan")),
                "best_din":   best.get("breakdown", {}).get("din", float("nan")),
                "best_lpi":   best.get("breakdown", {}).get("lpi", float("nan")),
                "iter_mean_top5": iter_mean_top5,
                "iter_min":       float(iter_losses[0][1]) if iter_losses else float("nan"),
                "lambda_rgb_active":   float(cur_lambda_rgb),
                "lambda_bg_active":    float(cur_lambda_bg),
                "lambda_depth_active": float(cur_lambda_depth),
                "lambda_dino_active":  float(cur_lambda_dino),
                "lambda_lpips_active": float(cur_lambda_lpips),
                "n_used": len(used_idxs),
            }
            trajectory.append(traj_entry)

        if wandb_run is not None:
            try:
                wandb.log({
                    "iter": iteration,
                    "elapsed_s": traj_entry["elapsed_s"],
                    "best_total": traj_entry["best_total"],
                    "best_rgb":   traj_entry["best_rgb"],
                    "best_bg":    traj_entry["best_bg"],
                    "best_dep":   traj_entry["best_dep"],
                    "best_din":   traj_entry["best_din"],
                    "iter_mean_top5": traj_entry["iter_mean_top5"],
                    "iter_min": traj_entry["iter_min"],
                    "lambda_rgb_active":   traj_entry["lambda_rgb_active"],
                    "lambda_bg_active":    traj_entry["lambda_bg_active"],
                    "lambda_depth_active": traj_entry["lambda_depth_active"],
                    "lambda_dino_active":  traj_entry["lambda_dino_active"],
                    "lambda_lpips_active": traj_entry["lambda_lpips_active"],
                    "n_used": traj_entry["n_used"],
                })
            except Exception as e:
                print(f"Warning: per-iter wandb.log failed: {e}")

        if iteration % save_every == 0 or iteration == num_iterations - 1:
            png = out_dir / f"progress_{iteration:04d}.png"
            save_progress_grid(
                str(png),
                ood_rgb_t=rgb_white, ood_mask_t=mask, ood_depth_t=da3_depth,
                ood_bgsafe_t=bg_safe,
                best_render_t=best["render"] if best["render"] is not None else renders[0].detach().cpu(),
                best_invdepth_t=best["invdepth"] if best["invdepth"] is not None else inv_depths[:1].detach().cpu(),
                best_aligned_invdepth_t=best["aligned_invdepth"] if best["aligned_invdepth"] is not None else inv_depths[:1].detach().cpu(),
                top_k_render_ts=top_k_renders, top_k_input_ts=top_k_inputs,
                top_k_losses=top_k_losses,
                iteration=iteration, best_loss=best["loss"],
                breakdown=best.get("breakdown", {"none": 0.0}),
            )
            save_loss_curves(str(out_dir / "loss_curves.png"), trajectory)

            panel_path = out_dir / f"panel_{iteration:04d}.png"
            try:
                if best["render"] is not None and best["aligned_invdepth"] is not None:
                    save_wandb_panel(
                        str(panel_path),
                        ood_rgb_t=rgb_white, ood_depth_t=da3_depth,
                        ood_dino_grid=dino_target_grid,
                        best_render_t=best["render"], best_input_t=best["input"],
                        best_aligned_invdepth_t=best["aligned_invdepth"],
                        top_k_render_ts=top_k_renders, top_k_input_ts=top_k_inputs,
                        top_k_invdepth_ts=top_k_invdepths,
                        top_k_losses=top_k_losses,
                        da3_inv=da3_inv, mask=mask,
                        iteration=iteration, best_loss=best["loss"],
                    )
            except Exception as e:
                print(f"Warning: panel build failed: {e}")
                panel_path = None

            if wandb_run is not None:
                try:
                    log_imgs = {"iter": iteration,
                                "progress_grid": wandb.Image(str(png)),
                                "loss_curves": wandb.Image(str(out_dir / "loss_curves.png"))}
                    if panel_path is not None and panel_path.exists():
                        log_imgs["panel"] = wandb.Image(str(panel_path))
                    wandb.log(log_imgs)
                except Exception as e:
                    print(f"Warning: image wandb.log failed: {e}")

        pbar.update(1)
    pbar.close()

    # final dump
    final_payload = {
        "best": best,
        "trajectory": trajectory,
        "top_k": {
            "renders": [t.cpu() for t in top_k_renders],
            "inputs":  [t.cpu() for t in top_k_inputs],
            "invdepths": [t.cpu() for t in top_k_invdepths],
            "losses":  top_k_losses,
            "rotations": [t.cpu() for t in top_k_rotations],
            "translations": [t.cpu() for t in top_k_translations],
            "candidate_idxs": top_k_idxs,
        },
        "config": {
            "num_iterations": num_iterations,
            "num_latents": num_latents, "num_rotations": num_rotations,
            "batch_size": batch_size,
            "lambdas": dict(rgb=lambda_rgb, bg=lambda_bg, depth=lambda_depth, dino=lambda_dino,
                            lpips=lambda_lpips, dino_warmup=lambda_dino_warmup_iters),
            "lambda_schedule": [{"end": e, **l} for e, l in lambda_schedule],
            "narrow_s1_end": narrow_s1_end,
            "narrow_s2_end": narrow_s2_end,
            "fast_T_optim": fast_T_optim, "fast_T_selection": fast_T_selection,
            "image_path": image_path,
            "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1e9),
            "total_time_s": float(time.time() - start_time),
            "w_reg_eff": float(cfg.abs.w_reg) if w_reg_override is None else float(w_reg_override),
            "rank_w_reg_mult": float(rank_w_reg_mult),
        },
    }
    torch.save(final_payload, out_dir / "final.pth")
    with open(out_dir / "trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(final_payload["config"] | {"best_loss": best["loss"], "best_iter": best["iteration"]},
                  f, indent=2)
    print(f"[real_ood] DONE  best_loss={best['loss']:.4f} (iter {best['iteration']})  "
          f"VRAM={final_payload['config']['peak_vram_gb']:.1f} GB  "
          f"time={final_payload['config']['total_time_s']:.1f}s")
    print(f"[real_ood] BREAKDOWN  {best.get('breakdown', {})}  "
          f"w_reg_eff={final_payload['config']['w_reg_eff']}  "
          f"rank_w_reg_mult={final_payload['config']['rank_w_reg_mult']}")


@hydra.main(version_base=None, config_path=_cfg.CONFIGS_DIR, config_name="abs_config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")

    image_paths_csv = cfg.real_ood.image_path
    image_paths = [p.strip() for p in image_paths_csv.split(",") if p.strip()]
    out_root = Path(_cfg.RUNS_ROOT) / cfg.general.prefix
    out_root.mkdir(parents=True, exist_ok=True)

    train_dataset = load_train_dataset(cfg)
    gp, gen = load_models(cfg)

    overrides = OmegaConf.select(cfg, "real_ood") or {}
    num_iterations = int(overrides.get("num_iterations", 120))
    save_every     = int(overrides.get("save_every", 20))
    num_latents    = int(overrides.get("num_latents", 0)) or None
    num_rotations  = int(overrides.get("num_rotations", 0)) or None
    batch_size     = int(overrides.get("batch_size", 0)) or None
    lambda_rgb     = float(overrides.get("lambda_rgb",   1.0))
    lambda_bg      = float(overrides.get("lambda_bg",    0.5))
    lambda_depth   = float(overrides.get("lambda_depth", 0.5))
    lambda_dino    = float(overrides.get("lambda_dino",  0.5))
    lambda_lpips   = float(overrides.get("lambda_lpips", 0.0))
    lambda_dino_warmup_iters = int(overrides.get("lambda_dino_warmup_iters", 0))
    fast_T_optim   = int(overrides.get("fast_T_optim", 6))
    fast_T_selection = int(overrides.get("fast_T_selection", 6))
    narrow_s1_end  = int(overrides.get("narrow_s1_end", 4))
    narrow_s2_end  = int(overrides.get("narrow_s2_end", 0)) or None  # None → num_iterations//3
    w_reg_override = overrides.get("w_reg", None)
    w_reg_override = float(w_reg_override) if w_reg_override is not None else None
    rank_w_reg_mult = float(overrides.get("rank_w_reg_mult", 1.0))

    stages_cfg = overrides.get("lambda_stages", None)
    lambda_schedule = _build_lambda_schedule(
        stages_cfg,
        defaults={"rgb": lambda_rgb, "bg": lambda_bg,
                  "depth": lambda_depth, "dino": lambda_dino,
                  "lpips": lambda_lpips},
        num_iterations=num_iterations,
    )

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    for ip in image_paths:
        stem = Path(ip).stem[-32:]
        out_subdir = str(overrides.get("out_subdir", None) or stem)
        out_dir = out_root / out_subdir

        if cfg.wandb.run_name is not None:
            wandb_run_name = cfg.wandb.run_name
        elif cfg.general.prefix is not None:
            wandb_run_name = cfg.general.prefix
        else:
            wandb_run_name = os.path.basename(__file__).split(".")[0]
        wandb_run_name = (wandb_run_name + "-" + out_subdir + "-"
                          + datetime.datetime.now().strftime("%d-%m-%y-%H-%M-%S"))
        wandb_run = wandb.init(project=cfg.wandb.project, reinit=True,
                               name=wandb_run_name, config=dict_cfg)

        try:
            run_one(
                cfg, ip, out_dir, train_dataset, gp, gen,
                num_iterations=num_iterations, save_every=save_every,
                num_latents=num_latents, num_rotations=num_rotations,
                batch_size=batch_size,
                lambda_rgb=lambda_rgb, lambda_bg=lambda_bg,
                lambda_depth=lambda_depth, lambda_dino=lambda_dino,
                lambda_lpips=lambda_lpips,
                lambda_dino_warmup_iters=lambda_dino_warmup_iters,
                fast_T_optim=fast_T_optim, fast_T_selection=fast_T_selection,
                wandb_run=wandb_run,
                lambda_schedule=lambda_schedule,
                narrow_s1_end=narrow_s1_end, narrow_s2_end=narrow_s2_end,
                w_reg_override=w_reg_override,
                rank_w_reg_mult=rank_w_reg_mult,
            )
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
