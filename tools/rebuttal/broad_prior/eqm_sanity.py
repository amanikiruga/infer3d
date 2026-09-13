"""EqM prior sanity check: load EqM-XL/2 + SD-VAE, generate class-conditional samples
via NAG-GD (single GPU, no DDP). Confirms the broad ImageNet EBM prior runs on this
cluster before we wire it to the Objaverse lifter.

  srun --jobid=35454875 --overlap bash -c 'cd .../broad_prior && CUDA_VISIBLE_DEVICES=0 \
    mamba run -n test2 python eqm_sanity.py --steps 250 --classes 279,555,817,933'
(279=Arctic fox, 555=fire engine, 817=sports car, 933=cheeseburger — arbitrary ImageNet ids)
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
import numpy as np
import torch
from PIL import Image

EQM = f"{_EXT}/rebuttal_infer3d_baselines/eqm/EqM"
sys.path.insert(0, EQM)
from models import EqM_models
from diffusers.models import AutoencoderKL

dev = "cuda"
CKPT = f"{EQM}/pretrained_models/EqM-XL-2-1400ep.pt"
OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/eqm_samples"


def load_eqm():
    model = EqM_models["EqM-XL/2"](input_size=32, num_classes=1000, uncond=True, ebm="none").to(dev)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "ema" in sd:
        model.load_state_dict(sd["ema"]); print("loaded ema weights")
    elif isinstance(sd, dict) and "model" in sd:
        model.load_state_dict(sd["model"]); print("loaded model weights")
    else:
        model.load_state_dict(sd); print("loaded raw state_dict")
    model.eval()
    return model


@torch.no_grad()
def nag_gd_sample(model, ys, steps, stepsize=0.0017, mu=0.3, cfg=1.5):
    n = ys.size(0)
    z = torch.randn(n, 4, 32, 32, device=dev)
    use_cfg = cfg > 1.0
    if use_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=dev)
        y = torch.cat([ys, y_null], 0)
        model_fn = lambda x, t: model.forward_with_cfg(x, t, y, cfg)
    else:
        y = ys
        model_fn = lambda x, t: model.forward(x, t, y)
    xt = z
    m = torch.zeros_like(xt)
    t = torch.zeros((xt.size(0),), device=dev)  # uncond -> t ignored internally
    for i in range(steps - 1):
        out = model_fn(xt + stepsize * m * mu, t)
        if not torch.is_tensor(out):
            out = out[0]
        m = out
        xt = xt + out * stepsize
    if use_cfg:
        xt, _ = xt.chunk(2, dim=0)
    return xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--classes", type=str, default="279,555,817,933")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    model = load_eqm()
    print(f"EqM params: {sum(p.numel() for p in model.parameters()):,}")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(dev).eval()

    ys = torch.tensor([int(c) for c in args.classes.split(",")], device=dev)
    lat = nag_gd_sample(model, ys, args.steps, cfg=args.cfg)
    with torch.no_grad():
        imgs = vae.decode(lat / 0.18215).sample
    imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    strip = np.concatenate(list(imgs), axis=1)
    Image.fromarray(strip).save(f"{OUT}/eqm_classes_{args.classes.replace(',', '_')}.png")
    print("saved", f"{OUT}/eqm_classes_{args.classes.replace(',', '_')}.png",
          "shape", strip.shape, "range", strip.min(), strip.max())


if __name__ == "__main__":
    main()
