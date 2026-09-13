# Cumulative ablation — ShapeNet cars / StyleGAN (build-up complement to the LOO table)

Each rung adds one component; increments are paired per-object. Cumulative increments are **order-dependent** (compare the two optimization orderings below) — the per-component LOO table is the order-independent statistic.

### A. Optimization build-up (losses = full), latent-first

| rung (cumulative) | PSNR↑ | Δ vs previous (paired) | n |
|---|---|---|---|
| feed-forward lifter (no optimization) | 16.78 ± 0.47 | — | 25 |
| + multi-start search & selection (no gradient) | 16.21 ± 0.36 | -0.56 ± 0.43 | 25 |
| + latent-code gradient opt (pose frozen) | 17.19 ± 0.45 | +0.98 ± 0.32 | 25 |
| + pose gradient opt = full Infer3D | 20.29 ± 0.41 | +3.10 ± 0.42 | 25 |



### A'. Same, pose-first (shows order-dependence)

| rung (cumulative) | PSNR↑ | Δ vs previous (paired) | n |
|---|---|---|---|
| feed-forward lifter (no optimization) | 16.78 ± 0.47 | — | 25 |
| + multi-start search & selection (no gradient) | 16.21 ± 0.36 | -0.56 ± 0.43 | 25 |
| + pose gradient opt (latent frozen) | 18.02 ± 0.37 | +1.81 ± 0.21 | 25 |
| + latent-code gradient opt = full Infer3D | 20.29 ± 0.41 | +2.27 ± 0.35 | 25 |



### B. Loss build-up (optimization = full) — cars analog of Table 8

| rung (cumulative) | PSNR↑ | Δ vs previous (paired) | n |
|---|---|---|---|
| MSE only | 19.68 ± 0.48 | — | 25 |
| + LPIPS | 20.17 ± 0.41 | +0.49 ± 0.23 | 25 |
| + noise-map regularizer = full Infer3D | 20.29 ± 0.41 | +0.12 ± 0.15 | 25 |

