# Stage B — CO3D geometry and novel-view evaluation

Turns the optimization checkpoints from Stage A into Chamfer distance and novel-view
metrics. Four stages, orchestrated by `run_eval_pipeline.py`:

1. **Generate splats** (`splats_to_pointcloud_via_mesh_batch_folder.py`) — rebuild the
   optimized Gaussians, the feed-forward baseline, and the pseudo-GT, and export PLYs.
2. **gs2mesh** — TSDF-fuse each PLY into a point cloud (external `gs2mesh`; CPU-bound and
   the slowest stage by far).
3. **Point-cloud lists** (`generate_pointcloud_lists_for_eval.py`).
4. **Sim(3)-ICP evaluation** (`eval_renderings_with_icp_batch.py`) — align prediction to
   pseudo-GT, render all novel views, write per-object Chamfer + PSNR/SSIM/LPIPS.

```bash
PYTHONPATH=. WANDB_MODE=disabled python tools/eval/run_eval_pipeline.py \
    --checkpoint_dir $INFER3D_RUNS_ROOT/hydrants-ood \
    --output_base_dir $INFER3D_RUNS_ROOT/eval \
    --dataset_name hydrants --pretrained_ckpt <lifter.pth>
```

`WANDB_MODE=disabled` is required — a wandb-0.22 bug crashes the ICP stage.

## `regenerate_ours.py` — which variant rebuilds the optimized splats

Two versions of this routine exist and they disagree. `$INFER3D_OURS_REGEN` selects:

* `aligned` (**default**) — rotates about the empirical centroid of the predicted splats.
* `paper` — commit `e534f0e`; rotates about the fixed point `[0,0,zgt]`.

Measured head-to-head on the same cached checkpoints: on **OOD** `aligned` is +1.07 dB
PSNR (95% CI [+0.29, +1.85], n=38); on **ID** the two are statistically
indistinguishable (n=18 and n=26, all CIs spanning zero). `aligned` is therefore used
everywhere. Both eval entry points import this one implementation — they previously
carried separate copies that had silently drifted apart. Full numbers in
[../../REPRODUCTION.md](../../REPRODUCTION.md).
