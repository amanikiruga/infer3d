# Component ablation (co3d) — leave-one-out from the full system

Feed-forward Splatter baseline (same objects, no optimization): **12.49 dB** (n=18)

| Arm | Removes | PSNR↑ | ΔPSNR (paired) | SSIM↑ | LPIPS↓ | n |
|---|---|---|---|---|---|---|
| Full Infer3D (control) | - | 15.79 ± 0.60 | — | 0.667 | 0.274 | 18 |
| - depth loss | L_depth | 16.14 ± 0.57 | +0.35 ± 0.44 | 0.679 | 0.264 | 18 |
| - latent prior (w_reg) | L_prior | 16.03 ± 0.51 | +0.24 ± 0.38 | 0.666 | 0.273 | 18 |
| - LPIPS loss | L_LPIPS | 15.10 ± 0.33 | -0.69 ± 0.58 | 0.588 | 0.344 | 18 |
| - MSE loss | L_MSE | 15.29 ± 0.48 | -0.50 ± 0.36 | 0.664 | 0.272 | 18 |
| - latent-code opt | latent-code optimization | 14.10 ± 0.29 | -1.69 ± 0.50 | 0.614 | 0.338 | 18 |
| - rendering-param (pose) opt | rendering-param optimization | 14.28 ± 0.51 | -1.51 ± 0.48 | 0.602 | 0.345 | 18 |
