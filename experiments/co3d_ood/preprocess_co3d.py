import torch
import torchvision
import numpy as np
import warnings

import math
import os
import tqdm
from PIL import Image
from omegaconf import DictConfig

from infer3d import config as _cfg

from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2,
)
from pytorch3d.implicitron.tools.config import expand_args_fields

# Where the raw CO3D download lives, and where to write the processed copy.
#   CO3D_RAW_ROOT   the --download_folder you gave co3d/download_dataset.py
#   CO3D_OUT_ROOT   defaults to infer3d.config.CO3D_DATASET_ROOT_HQ
#   CO3D_RESOLUTION 1080 for the HQ copy used by the experiments, or 128
CO3D_RAW_ROOT = os.getenv("CO3D_RAW_ROOT")
CO3D_OUT_ROOT = os.getenv("CO3D_OUT_ROOT", _cfg.CO3D_DATASET_ROOT_HQ)
RESOLUTION = int(os.getenv("CO3D_RESOLUTION", "1080"))

assert (
    CO3D_RAW_ROOT is not None
), "Change CO3D_RAW_ROOT to where your raw CO3D data resides"
assert (
    CO3D_OUT_ROOT is not None
), "Change CO3D_OUT_ROOT to where you want to save the processed CO3D data"


def _load_mask(path):
    with Image.open(path) as pil_im:
        mask = np.array(pil_im)
    mask = mask.astype(np.float32) / 255.0
    return mask[None]  # fake feature channel


def _get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1] + 1


def _get_bbox_from_mask(mask, thr, decrease_quant: float = 0.05):
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(f"Empty masks_for_bbox (thr={thr}) => using full image.")

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def _clamp_box_to_image_bounds_and_round(bbox_xyxy, image_size_hw):
    bbox_xyxy = bbox_xyxy.clone()
    bbox_xyxy[[0, 2]] = torch.clamp(bbox_xyxy[[0, 2]], 0, image_size_hw[-1])
    bbox_xyxy[[1, 3]] = torch.clamp(bbox_xyxy[[1, 3]], 0, image_size_hw[-2])
    if not isinstance(bbox_xyxy, torch.LongTensor):
        bbox_xyxy = bbox_xyxy.round().long()
    return bbox_xyxy  # pyre-ignore [7]


def _bbox_xywh_to_xyxy(xywh, clamp_size=None) -> torch.Tensor:
    xyxy = xywh.clone()
    if clamp_size is not None:
        xyxy[2:] = torch.clamp(xyxy[2:], clamp_size)
    xyxy[2:] += xyxy[:2]
    return xyxy


def _get_clamp_bbox(
    bbox,
    box_crop_context=0.0,
    image_path="",
):
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    bbox = bbox.clone()  # do not edit bbox in place

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.float()
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        raise ValueError(
            f"squashed image {image_path}!! The bounding box contains no pixels."
        )

    bbox[2:] = torch.clamp(bbox[2:], 2)  # set min height, width to 2 along both axes
    bbox_xyxy = _bbox_xywh_to_xyxy(bbox, clamp_size=2)

    return bbox_xyxy


def _bbox_xyxy_to_xywh(xyxy):
    wh = xyxy[2:] - xyxy[:2]
    xywh = torch.cat([xyxy[:2], wh])
    return xywh


def _crop_around_box(tensor, bbox, impath: str = ""):
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox = _clamp_box_to_image_bounds_and_round(
        bbox,
        image_size_hw=tensor.shape[-2:],
    )
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"
    return tensor


def _resize_image(image, image_height=800, image_width=800, mode="bilinear"):
    if image_height is None or image_width is None:
        # skip the resizing
        imre_ = torch.from_numpy(image)
        return imre_, 1.0, torch.ones_like(imre_[:1])
    # takes numpy array, returns pytorch tensor
    minscale = min(
        image_height / image.shape[-2],
        image_width / image.shape[-1],
    )
    imre = torch.nn.functional.interpolate(
        torch.from_numpy(image)[None],
        scale_factor=minscale,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        recompute_scale_factor=True,
    )[0]
    # pyre-fixme[19]: Expected 1 positional argument.
    imre_ = torch.zeros(image.shape[0], image_height, image_width)
    imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
    # pyre-fixme[6]: For 2nd param expected `int` but got `Optional[int]`.
    # pyre-fixme[6]: For 3rd param expected `int` but got `Optional[int]`.
    mask = torch.zeros(1, image_height, image_width)
    mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
    return imre_, minscale, mask


def _load_crop_fg_probability(
    mask_path, image_size, image_path, box_crop_mask_thr=0.4, box_crop_context=0.3
):
    fg_probability = None
    full_path = None
    bbox_xywh = None
    clamp_bbox_xyxy = None
    crop_box_xywh = None

    full_path = mask_path
    mask = _load_mask(full_path)  # done

    if mask.shape[-2:] != image_size:
        raise ValueError(f"bad mask size: {mask.shape[-2:]} vs {image_size}!")

    bbox_xywh = torch.tensor(_get_bbox_from_mask(mask, box_crop_mask_thr))  # done

    clamp_bbox_xyxy = _clamp_box_to_image_bounds_and_round(  # done
        _get_clamp_bbox(  # done
            bbox_xywh,
            image_path=image_path,
            box_crop_context=box_crop_context,
        ),
        image_size_hw=tuple(mask.shape[-2:]),
    )
    crop_box_xywh = _bbox_xyxy_to_xywh(clamp_bbox_xyxy)  # done

    mask = _crop_around_box(mask, clamp_bbox_xyxy, full_path)  # done

    fg_probability, _, _ = _resize_image(mask, mode="nearest")

    return fg_probability, full_path, bbox_xywh, clamp_bbox_xyxy, crop_box_xywh


def update_scores(top_scores, top_names, new_score, new_name):
    for sc_idx, sc in enumerate(top_scores):
        if new_score > sc:
            # shift scores and names to the right, start from the end
            for sc_idx_next in range(len(top_scores) - 1, sc_idx, -1):
                top_scores[sc_idx_next] = top_scores[sc_idx_next - 1]
                top_names[sc_idx_next] = top_names[sc_idx_next - 1]
            top_scores[sc_idx] = new_score
            top_names[sc_idx] = new_name
            break
    return top_scores, top_names


def main(dataset_name, category):

    subset_name = "fewview_dev"

    expand_args_fields(JsonIndexDatasetMapProviderV2)
    dataset_map = JsonIndexDatasetMapProviderV2(
        category=category,
        subset_name=subset_name,
        test_on_train=False,
        only_test_set=False,
        load_eval_batches=True,
        dataset_root=CO3D_RAW_ROOT,
        dataset_JsonIndexDataset_args=DictConfig(
            {"remove_empty_masks": False, "load_point_clouds": True}
        ),
    ).get_dataset_map()

    created_dataset = dataset_map[dataset_name]

    sequence_names = [k for k in created_dataset.seq_annots.keys()]

    bkgd = 0.0  # black background

    out_folder_path = os.path.join(
        CO3D_OUT_ROOT, "co3d_{}_for_gs".format(category), dataset_name
    )
    os.makedirs(out_folder_path, exist_ok=True)

    bad_sequences = []
    camera_Rs_all_sequences = {}
    camera_Ts_all_sequences = {}

    for sequence_name in tqdm.tqdm(sequence_names):

        folder_outname = os.path.join(out_folder_path, sequence_name)

        # # Skip if already processed
        # if (
        #     os.path.exists(os.path.join(folder_outname, "images_full.npy"))
        #     and os.path.exists(os.path.join(folder_outname, "images_fg.npy"))
        #     and os.path.exists(os.path.join(folder_outname, "focal_lengths.npy"))
        # ):
        #     print(f"Skipping already processed sequence: {sequence_name}")
        #     continue

        frame_idx_gen = created_dataset.sequence_indices_in_order(sequence_name)
        frame_idxs = []
        focal_lengths_this_sequence = []
        rgb_full_this_sequence = []
        rgb_fg_this_sequence = []
        fname_order = []

        # Preprocess cameras with Viewset Diffusion protocol
        cameras_this_seq = read_seq_cameras(created_dataset, sequence_name)

        camera_Rs_all_sequences[sequence_name] = cameras_this_seq.R
        camera_Ts_all_sequences[sequence_name] = cameras_this_seq.T

        while True:
            try:
                frame_idx = next(frame_idx_gen)
                frame_idxs.append(frame_idx)
            except StopIteration:
                break
        
        # Preprocess images
        for frame_idx in frame_idxs:
            # Read the original uncropped image
            frame = created_dataset[frame_idx]
            rgb_image = (
                torchvision.transforms.functional.pil_to_tensor(
                    Image.open(frame.image_path)
                ).float()
                / 255.0
            )
            # ============= Foreground mask =================
            # Initialise the foreground mask at the original resolution
            fg_probability = torch.zeros_like(rgb_image)[:1, ...]
            # Find size of the valid region in the 800x800 image (non-padded)
            resized_image_mask_boundary_y = (
                torch.where(frame.mask_crop > 0)[1].max() + 1
            )
            resized_image_mask_boundary_x = (
                torch.where(frame.mask_crop > 0)[2].max() + 1
            )

            mask_path = os.path.dirname(os.path.dirname(frame.image_path))
            mask_path = os.path.join(
                mask_path,
                "masks",
                os.path.basename(frame.image_path).replace(".jpg", ".png"),
            )

            image_size_hw = tuple([i.item() for i in frame.image_size_hw])

            _, _, _, _, crop_bbox_xywh = _load_crop_fg_probability(
                mask_path, image_size_hw, frame.image_path
            )

            # Resize the foreground mask to the original scale
            x0, y0, box_w, box_h = crop_bbox_xywh
            resized_mask = torchvision.transforms.functional.resize(
                frame.fg_probability[
                    :, :resized_image_mask_boundary_y, :resized_image_mask_boundary_x
                ],
                (box_h, box_w),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            # Fill in the foreground mask at the original scale in the correct location based
            # on where it was cropped
            fg_probability[:, y0 : y0 + box_h, x0 : x0 + box_w] = resized_mask

            # ============== Crop around principal point ================
            # compute location of principal point in Pytorch3D NDC coordinate system in pixels
            # scaling * 0.5 is due to the NDC min and max range being +- 1
            principal_point_cropped = (
                frame.camera.principal_point * 0.5 * frame.image_rgb.shape[1]
            )
            # compute location of principal point from top left corer, i.e. in image grid coords
            scaling_factor = max(box_h, box_w) / 800
            principal_point_x = (
                frame.image_rgb.shape[2] * 0.5 - principal_point_cropped[0, 0]
            ) * scaling_factor + x0
            principal_point_y = (
                frame.image_rgb.shape[1] * 0.5 - principal_point_cropped[0, 1]
            ) * scaling_factor + y0
            # Get the largest center-crop that fits in the foreground
            max_half_side = get_max_box_side(
                frame.image_size_hw, principal_point_x, principal_point_y
            )
            # After this transformation principal point is at (0, 0)
            rgb = crop_image_at_non_integer_locations(
                rgb_image, max_half_side, principal_point_x, principal_point_y
            )
            fg_probability_cc = crop_image_at_non_integer_locations(
                fg_probability, max_half_side, principal_point_x, principal_point_y
            )
            assert (
                frame.image_rgb.shape[1] == frame.image_rgb.shape[2]
            ), "Expected square images"

            # =============== Resize to 128 and save =======================
            # Resize raw rgb
            pil_rgb = torchvision.transforms.functional.to_pil_image(rgb)
            pil_rgb = torchvision.transforms.functional.resize(
                pil_rgb,
                RESOLUTION,
                interpolation=torchvision.transforms.InterpolationMode.LANCZOS,
            )
            rgb = torchvision.transforms.functional.pil_to_tensor(pil_rgb) / 255.0
            # Resize mask
            fg_probability_cc = torchvision.transforms.functional.resize(
                fg_probability_cc,
                RESOLUTION,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            # Save rgb
            rgb_full_this_sequence.append(rgb[:3, ...])
            # Save masked rgb
            rgb_fg = rgb[:3, ...] * fg_probability_cc + bkgd * (1 - fg_probability_cc)
            rgb_fg_this_sequence.append(rgb_fg)

            fname_order.append("{:05d}.png".format(frame_idx))

            # ============== Intrinsics transformation =================
            # Transform focal length according to the crop
            # Focal length is in NDC conversion so we do not need to change it when resizing
            # We should transform focal length to non-cropped image and then back to cropped but
            # the scaling factor of the full non-cropped image cancels out.
            transformed_focal_lengths = (
                frame.camera.focal_length * max(box_h, box_w) / (2 * max_half_side)
            )
            focal_lengths_this_sequence.append(transformed_focal_lengths)

        os.makedirs(folder_outname, exist_ok=True)
        focal_lengths_this_sequence = torch.stack(focal_lengths_this_sequence)

        if (
            torch.all(torch.logical_not(torch.stack(rgb_full_this_sequence).isnan()))
            and torch.all(torch.logical_not(torch.stack(rgb_fg_this_sequence).isnan()))
            and torch.all(torch.logical_not(focal_lengths_this_sequence.isnan()))
        ):

            np.save(
                os.path.join(folder_outname, "images_full.npy"),
                torch.stack(rgb_full_this_sequence).numpy(),
            )
            np.save(
                os.path.join(folder_outname, "images_fg.npy"),
                torch.stack(rgb_fg_this_sequence).numpy(),
            )
            np.save(
                os.path.join(folder_outname, "focal_lengths.npy"),
                focal_lengths_this_sequence.numpy(),
            )

            with open(os.path.join(folder_outname, "frame_order.txt"), "w+") as f:
                f.writelines([fname + "\n" for fname in fname_order])
        else:
            print("Warning! bad sequence {}".format(sequence_name))
            bad_sequences.append(sequence_name)

    # convert camera data to numpy archives and save
    for dict_to_save, dict_name in zip(
        [camera_Rs_all_sequences, camera_Ts_all_sequences], ["camera_Rs", "camera_Ts"]
    ):

        np.savez(
            os.path.join(out_folder_path, dict_name + ".npz"),
            **{k: v.detach().cpu().numpy() for k, v in dict_to_save.items()},
        )

    return bad_sequences


def get_max_box_side(hw, principal_point_x, principal_point_y):
    # assume images are always padded on the right - find where the image ends
    # find the largest center crop we can make
    max_x = hw[1]  # x-coord of the rightmost boundary
    min_x = 0.0  # x-coord of the leftmost boundary
    max_y = hw[0]  # y-coord of the top boundary
    min_y = 0.0  # y-coord of the bottom boundary

    max_half_w = min(principal_point_x - min_x, max_x - principal_point_x)
    max_half_h = min(principal_point_y - min_y, max_y - principal_point_y)
    max_half_side = min(max_half_h, max_half_w)

    return max_half_side


def crop_image_at_non_integer_locations(
    img, max_half_side: float, principal_point_x: float, principal_point_y: float
):
    """
    Crops the image so that its center is at the principal point.
    The boundaries are specified by half of the image side.
    """
    # number of pixels that the image spans. We don't want to resize
    # at this stage. However, the boundaries might be such that
    # the crop side is not an integer. Therefore there will be
    # minimal resizing, but it's extent will be sub-pixel.
    # We don't apply low-pass filtering at this stage and cropping is
    # done with bilinear sampling
    max_pixel_number = math.floor(2 * max_half_side)
    half_pixel_side = 0.5 / max_pixel_number
    x_locations = torch.linspace(
        principal_point_x - max_half_side + half_pixel_side,
        principal_point_x + max_half_side - half_pixel_side,
        max_pixel_number,
    )
    y_locations = torch.linspace(
        principal_point_y - max_half_side + half_pixel_side,
        principal_point_y + max_half_side - half_pixel_side,
        max_pixel_number,
    )
    grid_locations = torch.stack(
        torch.meshgrid(x_locations, y_locations, indexing="ij"), dim=-1
    ).transpose(0, 1)
    grid_locations[:, :, 1] = (grid_locations[:, :, 1] - img.shape[1] / 2) / (
        img.shape[1] / 2
    )
    grid_locations[:, :, 0] = (grid_locations[:, :, 0] - img.shape[2] / 2) / (
        img.shape[2] / 2
    )
    image_crop = torch.nn.functional.grid_sample(
        img.unsqueeze(0), grid_locations.unsqueeze(0)
    )
    return image_crop.squeeze(0)


def read_seq_cameras(dataset, sequence_name):

    frame_idx_gen = dataset.sequence_indices_in_order(sequence_name)
    frame_idxs = []
    while True:
        try:
            frame_idx = next(frame_idx_gen)
            frame_idxs.append(frame_idx)
        except StopIteration:
            break

    cameras_start = []
    for frame_idx in frame_idxs:
        cameras_start.append(dataset[frame_idx].camera)
    cameras_start = join_cameras_as_batch(cameras_start)
    cameras = cameras_start.clone()

    return cameras


if __name__ == "__main__":
    for category in ["vase"]:
        for split in ["train"]:
            bad_sequences_val = main(split, category)