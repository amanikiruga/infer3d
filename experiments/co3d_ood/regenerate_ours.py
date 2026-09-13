"""Reconstruct Infer3D's optimized splats from a saved optimization checkpoint.

Two variants exist and they do NOT agree:

``paper``
    The version from splatter-image commit ``e534f0e``, which produced the cached eval
    CSVs behind the paper's CO3D cells. Rotates about the fixed point ``[0, 0, zgt]``
    (``render_with_custom_camera``) and hands the lifter the OOD view's camera.

``aligned``
    A later, post-paper version (commits 5ceaf9c / c1d2e70) that rotates about the
    empirical centroid of the predicted splats (``render_with_custom_camera_align``)
    and hands the lifter view 0 instead of the OOD view. On CO3D in-distribution
    objects this under-reproduces the paper's ``ours`` PSNR by 2-4 dB.

Which one reproduces the paper depends on the cell, which was measured directly by
running the full ICP eval over the same cached Hydrants OOD checkpoints under both
(paired, n=38 objects):

    aligned - paper   PSNR +1.07 [+0.29,+1.85]   SSIM +0.014   LPIPS -0.020   CD -0.100

so ``aligned`` reproduces the published OOD cells (and is geometrically the more
sensible choice: it rotates about the object's actual centre rather than a fixed
point), while ``paper`` reproduces the published ID cells. The published Table 1
therefore mixes the two conventions across its rows. ``aligned`` is the default
because the OOD rows are the paper's claim; REPRODUCTION.md reports both so the
choice is visible rather than buried.

Select with ``$INFER3D_OURS_REGEN`` (``aligned`` default, or ``paper``). Both eval
entrypoints import from here so they can no longer drift apart, which they had:
the two copies previously differed in whether ``zgt_ood`` was passed to the
inverse-rotation call.
"""
import os

import torch

from infer3d.utils.optim import render_with_custom_camera, render_with_custom_camera_align

VARIANT = os.getenv("INFER3D_OURS_REGEN", "aligned").lower()


def _common(checkpoint, cfg, device):
    """Unpack the pieces both variants need."""
    ood_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in checkpoint["ood_data"].items()}
    ood_random_rotation = checkpoint.get("ood_random_rotation")
    if ood_random_rotation is None:
        raise ValueError("ood_random_rotation not found in checkpoint")

    ood_image_index = checkpoint["ood_image_index"]
    input_image = checkpoint["best_input_image"]
    if input_image.dim() == 4:
        input_image = input_image.squeeze(0)
    if input_image.shape[0] > 3:
        input_image = input_image[:3, :, :]

    ood_origin_distances = ood_data["origin_distances"][0, ood_image_index].to(device)
    return dict(
        ood_data=ood_data,
        ood_image_index=ood_image_index,
        ood_random_rotation=ood_random_rotation.to(device),
        best_rotation_matrix=checkpoint["best_rotation_matrix"].to(device),
        best_translation_matrix=checkpoint["best_translation_matrix"].to(device),
        zgt=checkpoint["zgt"],
        focals=ood_data["focals_pixels"][0, ood_image_index].to(device),
        input_images=torch.cat([input_image.unsqueeze(0).unsqueeze(1).to(device),
                                ood_origin_distances.unsqueeze(0).unsqueeze(1)], dim=2),
        background=torch.tensor([1, 1, 1] if cfg.data.white_background else [0, 0, 0],
                                dtype=torch.float32, device=device),
    )


def regenerate_ours_splats(checkpoint, gaussian_predictor, cfg, device, variant=None):
    """Return (transformed_splats, ood_data) for the selected variant."""
    variant = (variant or VARIANT).lower()
    if variant not in ("paper", "aligned"):
        raise ValueError(f"unknown INFER3D_OURS_REGEN variant {variant!r}")
    c = _common(checkpoint, cfg, device)
    n_in = cfg.data.input_images
    # The lifter's camera: the OOD view (paper) or view 0 (aligned).
    start = c["ood_image_index"] if variant == "paper" else 0

    with torch.no_grad():
        pred_splats = gaussian_predictor(
            c["input_images"],
            c["ood_data"]["view_to_world_transforms"][:1, start:start + n_in],
            c["ood_data"]["source_cv2wT_quat"][:1, start:start + n_in],
            c["focals"].unsqueeze(0).unsqueeze(0),
        )
    pred_splats = {k: v[0] for k, v in pred_splats.items()}

    shared = dict(device=device, return_splats=True)
    if variant == "paper":
        splats = render_with_custom_camera(
            pred_splats, c["background"], cfg, c["focals"], c["best_rotation_matrix"],
            c["zgt"], translation=c["best_translation_matrix"], **shared)
        splats = render_with_custom_camera(
            splats, c["background"], cfg, c["focals"], c["ood_random_rotation"].T,
            c["zgt"], translation=None, **shared)
    else:
        splats = render_with_custom_camera_align(
            pred_splats, c["background"], cfg, c["focals"], c["best_rotation_matrix"],
            c["zgt"], translation=c["best_translation_matrix"], zgt_ood=c["zgt"], **shared)
        splats = render_with_custom_camera(
            splats, c["background"], cfg, c["focals"], c["ood_random_rotation"].T,
            c["zgt"], translation=None, zgt_ood=c["zgt"], **shared)
    return splats, c["ood_data"]
