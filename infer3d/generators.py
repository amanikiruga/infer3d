"""Generative priors used by Infer3D's analysis-by-synthesis.

Two generators are supported, both used as *frozen* image priors whose latent is
optimized to match the input view:

* ``DiffAEGenerator``  - a Diffusion-Autoencoder (external ``diffae`` repo).
* ``StyleGANGenerator`` - a class-conditional StyleGAN2 (external ``stylegan3`` repo).

The external repos are not vendored; point at them with ``DIFFAE_ROOT`` /
``STYLEGAN3_ROOT`` (see ``infer3d/config.py``). Checkpoints are selected by the
category keys in ``config.DIFFAE_CONF_NAMES`` / ``config.STYLEGAN_CKPTS``.
"""
import sys
import torch
import torch.nn as nn

from infer3d import config


# --------------------------------------------------------------------------- #
#  DiffAE
# --------------------------------------------------------------------------- #
class DiffAEGenerator(nn.Module):
    def __init__(self, category, checkpoint_path=None, device="cuda"):
        super().__init__()
        if config.DIFFAE_ROOT not in sys.path:
            sys.path.append(config.DIFFAE_ROOT)
        from templates import (  # noqa: F401  (diffae template factory functions)
            co3d_hydrants_autoenc_128, co3d_vases_autoenc_128, nmr_train_autoenc,
        )
        conf_factory = {
            "hydrants": co3d_hydrants_autoenc_128,
            "vases": co3d_vases_autoenc_128,
            "shapenet_nmr": nmr_train_autoenc,
        }[category]
        from templates import LitModel
        conf = conf_factory()
        self.conf = conf
        model = LitModel(conf)
        ckpt = checkpoint_path or f"{config.DIFFAE_ROOT}/checkpoints/{conf.name}/last.ckpt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=False)
        model.ema_model.eval().to(device)
        self.model = model
        self.device = device

    def forward(self, latent_code, cond=None, T=12):
        if cond is None:
            cond = torch.randn(1, 512, device=self.device)
        return self.model.render(latent_code, cond, T=T)

    def encode(self, img):
        return self.model.encode(img)

    def encode_stochastic(self, img, cond, T=250):
        return self.model.encode_stochastic(img, cond, T=T)


# --------------------------------------------------------------------------- #
#  StyleGAN2 (class-conditional)
# --------------------------------------------------------------------------- #
class StyleGANGenerator(nn.Module):
    """Thin compat wrapper around a StyleGAN3-repo ``G_ema`` loaded from a .pkl."""

    def __init__(self, category=None, checkpoint_path=None, device="cuda"):
        super().__init__()
        if config.STYLEGAN3_ROOT not in sys.path:
            sys.path.append(config.STYLEGAN3_ROOT)
        import dnnlib
        import legacy

        ckpt = checkpoint_path or config.STYLEGAN_CKPTS[category]
        with dnnlib.util.open_url(ckpt) as f:
            self.G = legacy.load_network_pkl(f)["G_ema"].to(device)
        self.G.eval()
        self.device = device

        class _Compat:
            def __init__(self, G):
                self.G = G

            def style(self, z, c=None):
                return self.G.mapping(z, c)[:, 0, :]

            def make_noise(self):
                return []

        self.model = _Compat(self.G)

    def forward(self, cond=None, n_sample=1, label=None):
        assert label is not None, "StyleGAN generator requires a class label"
        w = self.model.style(cond if cond is not None
                             else torch.randn(n_sample, self.G.z_dim, device=self.device), label)
        return self.forward_latent_w(w)

    def forward_latent_w(self, latent_w):
        if latent_w.dim() == 1:
            latent_w = latent_w.unsqueeze(0)
        ws = latent_w.unsqueeze(1).repeat(1, self.G.num_ws, 1)
        img = self.G.synthesis(ws).clamp(-1, 1)
        return (img + 1) / 2  # -> [0, 1]
