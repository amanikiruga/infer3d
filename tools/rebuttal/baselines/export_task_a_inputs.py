"""Export identical Task-A (NMR car SO(3)) input bundles for every baseline.

Mirrors exactly how our SO(3) numbers were produced
(experiments/stylegan3-ours/stylegan3_nmr_abs_w_inversion_with_baseline.py):
  * input  = so3-rendered view of the object, deterministic frame from
             nmr_test_split.json, image = gt_images[0, 0] of the val dataset.
  * targets = the 24 `test/`-folder views, with cameras re-expressed RELATIVE to
              the input camera via make_data_relative_to (copied verbatim).

Writes per object under rebuttal/results/task_a_inputs/{obj}/:
  input.png, input_cam_absolute.npz (privileged; FINV† + diagnostics only),
  targets/{i:02d}.png, cams_relative.npz
Also exports the side-top canonical-input control bundle under
task_a_inputs_control/{obj}/ (same structure) for the renderer-gap control.

Run (CPU is fine):
  SPLATTER_IMAGE_ROOT=<worktree> SHAPENET_NMR_ROOT=.../srn_nmr_classes \
  WANDB_MODE=disabled mamba run -n test2 python export_task_a_inputs.py
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

WORKTREE = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, WORKTREE)
os.environ.setdefault("SPLATTER_IMAGE_ROOT", WORKTREE)
os.environ.setdefault(
    "SHAPENET_NMR_ROOT",
    f"{_EXT}/datasets/srn_nmr_classes")

from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from splatter_image_datasets.shapenet_nmr import ShapenetNMR
from utils.general_utils import matrix_to_quaternion

OUT_ROOT = f"{WORKTREE}/rebuttal/results"


# ---- helpers copied verbatim from the SO(3) script (numerics must match) ----
def make_poses_relative_to_first(d):
    inverse_first_camera = d["world_view_transforms"][:, 0].inverse().clone()
    for c in range(d["world_view_transforms"].shape[1]):
        d["world_view_transforms"][:, c] = torch.bmm(
            inverse_first_camera, d["world_view_transforms"][:, c])
        d["view_to_world_transforms"][:, c] = torch.bmm(
            d["view_to_world_transforms"][:, c], inverse_first_camera.inverse())
        d["full_proj_transforms"][:, c] = torch.bmm(
            inverse_first_camera, d["full_proj_transforms"][:, c])
        d["camera_centers"][:, c] = d["world_view_transforms"][:, c].inverse()[:, 3, :3]
    return d


def get_source_cw2wT(source_cameras_view_to_world):
    qs = []
    for c_idx in range(source_cameras_view_to_world.shape[0]):
        qs.append(matrix_to_quaternion(
            source_cameras_view_to_world[c_idx, :3, :3].transpose(0, 1)))
    return torch.stack(qs, dim=0)


def make_data_relative_to(target_data, reference_data, batch_size=1):
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
            for k, ak in [("world_view_transforms", "world_view_transforms_absolute"),
                          ("view_to_world_transforms", "view_to_world_transforms_absolute"),
                          ("full_proj_transforms", "full_proj_transforms_absolute"),
                          ("camera_centers", "camera_centers_absolute")]:
                comb_data[k] = torch.cat([comb_data[k], data[ak]], dim=1)
    comb_data = make_poses_relative_to_first(comb_data)
    source_cw2wTs = []
    for b in range(batch_size):
        source_cw2wTs.append(get_source_cw2wT(comb_data["view_to_world_transforms"][b]))
    comb_data["source_cv2wT_quat"] = torch.stack(source_cw2wTs)
    for k in ["world_view_transforms", "view_to_world_transforms",
              "full_proj_transforms", "camera_centers", "source_cv2wT_quat"]:
        comb_data[k] = comb_data[k][:, num_across_first_dim:]
    comb_data["gt_images"] = target_data["gt_images"]
    return comb_data
# -----------------------------------------------------------------------------


def to_png(t):  # [3,H,W] float 0..1
    return Image.fromarray(
        np.clip(t.permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8))


def export_bundle(obj_id, test_idx, input_view, out_dir, cfg):
    os.makedirs(f"{out_dir}/targets", exist_ok=True)
    val_ds = ShapenetNMR(cfg, "test", override_overall_view=input_view,
                         override_example_ids=[obj_id],
                         deterministic_test_idxs=[test_idx])
    eval_ds = ShapenetNMR(cfg, "test", override_overall_view="test",
                          override_example_ids=[obj_id],
                          deterministic_test_idxs=[0])
    ood = next(iter(DataLoader(val_ds, batch_size=1, shuffle=False)))
    ev = next(iter(DataLoader(eval_ds, batch_size=1, shuffle=False)))

    rel = make_data_relative_to(target_data={k: v.clone() if torch.is_tensor(v) else v
                                             for k, v in ev.items()},
                                reference_data=ood)

    to_png(torch.clamp(ood["gt_images"][0, 0], 0, 1)).save(f"{out_dir}/input.png")
    np.savez(f"{out_dir}/input_cam_absolute.npz",
             view_to_world=ood["view_to_world_transforms_absolute"][0, 0].numpy(),
             world_view=ood["world_view_transforms_absolute"][0, 0].numpy())
    # ABSOLUTE cam2world (splatter convention: cam2world = view_to_world_transforms.T)
    # for input (so3) and all targets (test views), all in the SAME NMR canonical
    # world frame -> lets baselines that reconstruct in a canonical frame place the
    # target cameras correctly via a similarity anchored at the estimated input pose.
    np.savez(f"{out_dir}/cams_absolute.npz",
             input_v2w=ood["view_to_world_transforms_absolute"][0, 0].numpy(),
             targets_v2w=ev["view_to_world_transforms_absolute"][0].numpy())
    n = rel["gt_images"].shape[1]
    for i in range(n):
        to_png(torch.clamp(rel["gt_images"][0, i], 0, 1)).save(
            f"{out_dir}/targets/{i:02d}.png")
    np.savez(f"{out_dir}/cams_relative.npz",
             world_view_transforms=rel["world_view_transforms"][0].numpy(),
             view_to_world_transforms=rel["view_to_world_transforms"][0].numpy(),
             full_proj_transforms=rel["full_proj_transforms"][0].numpy(),
             camera_centers=rel["camera_centers"][0].numpy(),
             fov_deg=float(cfg.data.fov), znear=float(cfg.data.znear),
             zfar=float(cfg.data.zfar), resolution=128)
    return n


def main():
    with initialize_config_dir(version_base=None, config_dir=f"{WORKTREE}/configs"):
        cfg = compose(config_name="abs_config", overrides=["+dataset=shapenet-nmr"])
    bench = json.load(open(f"{OUT_ROOT}/task_a_bench25.json"))  # [[obj,'ood',idx],...]
    ids = [e[0] for e in bench]
    idx_map = {e[0]: int(e[2]) for e in bench}
    print(f"{len(ids)} car objects (bench25)")
    for view, sub in [("so3", "task_a_inputs"), ("side-top", "task_a_inputs_control")]:
        for oid in ids:
            out = f"{OUT_ROOT}/{sub}/{oid}"
            n = export_bundle(oid, idx_map[oid], view, out, cfg)
            print(f"[{view}] {oid}: input + {n} targets")
    print("DONE")


if __name__ == "__main__":
    main()
