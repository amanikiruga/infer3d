#!/usr/bin/env python3
"""
Minimal loader for the frozen DiffAE generative prior (re10k_autoenc_256), isolated to a
single place so the rest of the RE10K pipeline has exactly one dependency on the external
DiffAE repo + checkpoint.

External dependency (resolved via infer3d.config, env-overridable):
  DIFFAE_ROOT   clone of the diffae repo, providing `templates` and `experiment.LitModel`
                and `checkpoints/re10k_autoenc_256/last.ckpt`.
Exposes .encode / .encode_stochastic / .render (the DiffAE semantic-cond + stochastic-xT
autoencoder API) used by the blind calibration selector.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('WANDB_MODE', 'disabled')
import torch

from infer3d import config as C

_DIFFAE = os.getenv("DIFFAE_ROOT", C.DIFFAE_ROOT)
if _DIFFAE not in sys.path:
    sys.path.insert(0, _DIFFAE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_diffae(checkpoint_path: str | None = None):
    """Load the frozen re10k_autoenc_256 DiffAE. Returns a LitModel in eval mode on device.
    checkpoint_path defaults to $DIFFAE_ROOT/checkpoints/re10k_autoenc_256/last.ckpt."""
    from templates import re10k_autoenc_256
    from experiment import LitModel
    conf = re10k_autoenc_256()
    model = LitModel(conf)
    ckpt = checkpoint_path or f"{_DIFFAE}/checkpoints/{conf.name}/last.ckpt"
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=False)
    for m in (model.ema_model, model.model):
        m.eval(); m.to(device)
    return model
