# Component ablation (cars) — leave-one-out from the full system

Feed-forward Splatter baseline (same objects, no optimization): **16.78 dB** (n=25)

| Arm | Removes | PSNR↑ | ΔPSNR (paired) | SSIM↑ | LPIPS↓ | n |
|---|---|---|---|---|---|---|
| Full Infer3D (control) | - | 20.29 ± 0.41 | — | 0.860 | 0.145 | 25 |
| - rendering-param (pose) opt | rendering-param optimization | 17.19 ± 0.45 | -3.10 ± 0.42 | 0.820 | 0.187 | 25 |
| - latent-code opt | latent-code optimization | 18.02 ± 0.37 | -2.27 ± 0.35 | 0.833 | 0.180 | 25 |
| - ALL gradient opt (search only) | latent + pose optimization | 16.21 ± 0.36 | -4.08 ± 0.32 | 0.806 | 0.205 | 25 |
| - LPIPS loss (MSE only) | L_LPIPS | 19.76 ± 0.48 | -0.53 ± 0.25 | 0.856 | 0.163 | 25 |
| - MSE loss (LPIPS only) | L_MSE | 19.09 ± 0.40 | -1.20 ± 0.25 | 0.850 | 0.152 | 25 |
| - noise-map regularizer | L_noise-reg | 20.17 ± 0.41 | -0.12 ± 0.15 | 0.860 | 0.144 | 25 |
| reduce pose multi-start (30->10 rot) | pose multi-start breadth | 19.97 ± 0.43 | -0.32 ± 0.17 | 0.857 | 0.151 | 25 |
| reduce latent multi-start (20->2 lat) | latent multi-start breadth | 19.49 ± 0.41 | -0.80 ± 0.28 | 0.849 | 0.155 | 25 |
