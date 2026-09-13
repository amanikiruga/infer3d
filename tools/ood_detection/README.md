# OOD detection and adaptive inference (Tables 4, 6, 7)

Infer3D's detector is free: run one feed-forward encode→decode→lift→render pass and use
the resulting reconstruction error as the score. No extra classifier is trained.

| script | produces |
|---|---|
| `diffae_only.py` | per-object detection scores + timing (`diffae_only_results.csv`) |
| `auroc_step_t.py` | AUROC as a function of optimization step (Table 6) |
| `f1_threshold_sweep.py` | threshold calibration and F1 vs number of validation objects (Table 7) |
| `adaptive_stream.py` | mixed ID/OOD stream simulation: feed-forward vs optimize-all vs routed (Table 4) |

```bash
# score one category, ID and OOD passes
PYTHONPATH=. python tools/ood_detection/diffae_only.py \
    abs=diffae_abs +dataset=vases general.split=0 general.total_splits=1 \
    general.prefix=detect-vases-ood general.run_indist=false \
    opt.pretrained_ckpt=<lifter.pth> \
    +general.test_imgs_csv_path=$(pwd)/assets/co3d_vases_test_paths_1080.csv

# then AUROC / F1 / adaptive routing from the resulting CSVs and loss trajectories
PYTHONPATH=. python tools/ood_detection/auroc_step_t.py --id_dir <...> --ood_dir <...> --output_csv <...>
```

Measured here: AUROC **0.971** on CO3D Hydrants and **0.939** on Vases (LPIPS alone
0.999), detection latency **283 ms**. Note that MSE alone is close to uninformative on
Vases (AUROC 0.53) — the signal is essentially all LPIPS, which is worth knowing since
the paper credits the two equally.
