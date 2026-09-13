"""Core reusable pipeline for broad-prior Infer3D:
   EqM (ImageNet EBM) latent  --VAE-->  256px image  --SAM3 mask + center-->  128px
   white-bg object  --Objaverse Splatter lifter Phi-->  3D Gaussians  --render-->  views.

All components frozen. Everything on the decode->lift->render path is differentiable so an
analysis-by-synthesis loop can backprop a render/appearance loss to the EqM latent.

Reuses: splatter-image GaussianSplatPredictor + render_predicted + objaverse geometry
(get_loop_cameras, make_poses_relative_to_first), EqM models, diffusers SD-VAE, SAM3.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, sys
import numpy as np
import torch
import torch.nn.functional as F

ORIG = f"{_EXT}/splatter-image"
EQM = f"{_EXT}/rebuttal_infer3d_baselines/eqm/EqM"
os.environ.setdefault("OBJAVERSE_ROOT", f"{_EXT}/datasets/objaverse/views_release")
os.environ.setdefault("OBJAVERSE_LVIS_ANNOTATION_PATH", f"{_EXT}/diffae/lvis-annotations.json")
sys.path.insert(0, ORIG)
sys.path.insert(0, EQM)

from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted
from utils.camera_utils import get_loop_cameras
from utils.general_utils import matrix_to_quaternion
from utils.graphics_utils import getProjectionMatrix, fov2focal

DEV = "cuda"
LIFTER_CKPT = f"{ORIG}/checkpoints/model_objaverse.pth"
EQM_CKPT = f"{EQM}/pretrained_models/EqM-XL-2-1400ep.pt"
VAE_SCALE = 0.18215


# ----------------------------- config ----------------------------- #
def objaverse_cfg():
    GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{ORIG}/configs")
    cfg = compose(config_name="abs_config", overrides=["+dataset=objaverse",
                                                       f"opt.pretrained_ckpt={LIFTER_CKPT}"])
    OmegaConf.set_struct(cfg, False)
    cfg.data.setdefault("input_images", 1) if hasattr(cfg.data, "setdefault") else None
    if "input_images" not in cfg.data: cfg.data.input_images = 1
    if "subset" not in cfg.data: cfg.data.subset = -1
    if "imgs_per_obj" not in cfg.opt: cfg.opt.imgs_per_obj = 12
    return cfg


# ----------------------------- lifter ----------------------------- #
def load_lifter(cfg):
    gp = GaussianSplatPredictor(cfg).to(memory_format=torch.channels_last).to(DEV).eval()
    gp.load_state_dict(torch.load(LIFTER_CKPT, map_location=DEV, weights_only=False)["model_state_dict"])
    for p in gp.parameters():
        p.requires_grad_(False)
    return gp


def lift(gp, img128):
    """img128: [3,128,128] or [1,3,128,128] in [0,1], white bg, object-centered.
    Returns splats dict (per-Gaussian tensors), predicted in source-relative frame."""
    if img128.dim() == 3:
        img128 = img128.unsqueeze(0)
    x = img128.unsqueeze(1).to(DEV)                              # [1,1,3,H,W]
    v2w = torch.eye(4, device=DEV).view(1, 1, 4, 4)             # source = identity (relative frame)
    quat = matrix_to_quaternion(torch.eye(3, device=DEV).transpose(0, 1)).view(1, 1, 4)
    sp = gp(x, v2w, quat, None)
    return {k: v[0] for k, v in sp.items()}


# ------------------------ turntable cameras ------------------------ #
def build_turntable(cfg, num=120, max_elevation=np.pi / 12):
    """Deterministic orbit in the lifter's source-relative frame. Frame 0 == source view.
    Returns lists of (world_view_transform[4,4], full_proj_transform[4,4], camera_center[3])."""
    loops = get_loop_cameras(num, radius=2.0, max_elevation=max_elevation)
    proj = getProjectionMatrix(znear=cfg.data.znear, zfar=cfg.data.zfar,
                               fovX=cfg.data.fov * 2 * np.pi / 360,
                               fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0, 1)
    wv, v2w, cc = [], [], []
    for c2w in loops:
        c2w = torch.from_numpy(c2w).float()
        v2w.append(c2w.transpose(0, 1))
        wv.append(c2w.inverse().transpose(0, 1))
        cc.append(c2w.transpose(0, 1)[3, :3].clone())
    wv = torch.stack(wv); v2w = torch.stack(v2w); cc = torch.stack(cc)
    # relative to source (frame 0)
    inv0 = wv[0].inverse().clone()
    fp = []
    for i in range(len(wv)):
        wv[i] = inv0 @ wv[i]
        cc[i] = wv[i].inverse()[3, :3]
        fp.append(wv[i] @ proj)
    fp = torch.stack(fp)
    return wv.to(DEV), fp.to(DEV), cc.to(DEV)


def render_view(gp_cfg, splats, wv, fp, cc, bg=None):
    if bg is None:
        bg = torch.tensor([1., 1, 1], device=DEV)
    out = render_predicted(splats, wv.unsqueeze(0), fp.unsqueeze(0), cc.unsqueeze(0), bg, gp_cfg,
                           focals_pixels=None)
    return out["render"].clamp(0, 1)


def turntable_frames(cfg, splats, cams, stride=1):
    wv, fp, cc = cams
    frames = []
    for i in range(0, wv.shape[0], stride):
        with torch.no_grad():
            r = render_view(cfg, splats, wv[i], fp[i], cc[i])
        frames.append((r.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    return frames


# ----------------------------- EqM prior ----------------------------- #
def load_eqm():
    from models import EqM_models
    m = EqM_models["EqM-XL/2"](input_size=32, num_classes=1000, uncond=True, ebm="none").to(DEV)
    sd = torch.load(EQM_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(sd["ema"] if "ema" in sd else sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def eqm_grad(model, z, y, t=None):
    """Equilibrium gradient field g(z): sampling does z <- z + g*eta (descent toward manifold)."""
    if t is None:
        t = torch.zeros((z.shape[0],), device=DEV)
    out = model.forward(z, t, y)
    return out if torch.is_tensor(out) else out[0]


def load_vae():
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(DEV).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def vae_decode(vae, z):
    return (vae.decode(z / VAE_SCALE).sample.clamp(-1, 1) + 1) / 2   # [B,3,256,256] in [0,1]


def vae_encode(vae, img01):
    x = img01 * 2 - 1
    return vae.encode(x).latent_dist.mean * VAE_SCALE


# ----------------------------- SAM3 ----------------------------- #
class SAM3:
    def __init__(self, model="facebook/sam3"):
        from transformers import Sam3Model, Sam3Processor
        self.model = Sam3Model.from_pretrained(model).to(DEV).eval()
        self.proc = Sam3Processor.from_pretrained(model)

    @torch.no_grad()
    def mask(self, img_uint8, text, threshold=0.3):
        from PIL import Image
        img = Image.fromarray(img_uint8)
        H, W = img_uint8.shape[:2]
        inp = self.proc(images=img, text=text, return_tensors="pt").to(DEV)
        out = self.model(**inp)
        res = self.proc.post_process_instance_segmentation(out, threshold=threshold,
                                                           mask_threshold=0.5, target_sizes=[(H, W)])[0]
        if len(res["scores"]) == 0:
            return None
        best = int(torch.as_tensor(res["scores"]).argmax())
        m = res["masks"][best]
        m = m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m)
        return m > 0.5


def center_on_white(img01, mask_bool, out=128, margin=0.15):
    """img01: [3,H,W] tensor; mask_bool: [H,W] np. Composite object on white, square-crop
    around mask bbox with margin, resize to out. Returns [3,out,out] tensor (differentiable
    in img01 pixels; crop box from mask is fixed)."""
    H, W = img01.shape[-2:]
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return F.interpolate(img01.unsqueeze(0), (out, out), mode="bilinear", align_corners=False)[0]
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half = max(y1 - y0, x1 - x0) * (1 + margin) / 2
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    m = torch.from_numpy(mask_bool).float().to(img01.device).unsqueeze(0)
    comp = img01 * m + (1 - m) * 1.0                       # white bg
    crop = comp[:, y0:y1, x0:x1]
    # pad to square
    ch, cw = crop.shape[-2:]
    s = max(ch, cw)
    pad = [(s - cw) // 2, s - cw - (s - cw) // 2, (s - ch) // 2, s - ch - (s - ch) // 2]
    crop = F.pad(crop.unsqueeze(0), pad, value=1.0)
    return F.interpolate(crop, (out, out), mode="bilinear", align_corners=False)[0]
