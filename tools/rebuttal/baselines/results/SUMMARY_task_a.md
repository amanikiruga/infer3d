# Task A — SO(3) viewpoint generalization, ShapeNet cars (25 objects)

Single image, no GT pose given to any method (unless †). Control = in-distribution
side-top input (isolates pose-OOD from renderer/appearance).

## A1. Novel-view synthesis (POSE-SENSITIVE — the metric that tests SO(3))
Rendered at each method's ESTIMATED input pose (no ICP). PSNR / SSIM / LPIPS, mean/25.

| Method | steps | Control (ID pose) | SO(3) OOD | drop |
|---|---|---|---|---|
| nerf-from-image | 30  | 18.15 / .855 / .099 | 13.92 / .795 / .207 | -4.2 dB |
| nerf-from-image | 300 | 19.85 / .878 / .084 | 14.28 / .802 / .202 | -5.6 dB |
| **Infer3D (ours)** | full | 21.43 / .878 / .123 | 20.04 / .858 / .151 | -1.4 dB (shared-harness SO3=19.16 n=24, ctrl=20.72 vs NFI 14.28) |

NFI reconstructs fine at ID pose (~18-20 dB -> NMR renderer is NOT the problem) but
collapses ~5 dB under SO(3); 10x more optimization barely helps. NFI DOES estimate+
optimize pose (encoder+PnP+latent) => generic optimization-based inversion failing on
VIEWPOINT OOD.

## A2. Geometry Chamfer (POSE-REMOVED shape control — ICP-aligned, gauge-free)
Vs GT ShapeNet mesh; FINAL 25-object summaries (task_a_geom/chamfer_*_so3_summary.json;
NFI n=24). Control (ID) from chamfer_*_control.csv. (An earlier 17-object interim table
lived here; superseded.)

| Method | cd_sym SO(3) | cd_sym control (ID) |
|---|---|---|
| Infer3D (ours) | 0.0091 | 0.0077 |
| nerf-from-image | 0.0037 (n=24) | 0.0010 |
| EG3D-PTI | 0.0127 | 0.0086 |
| pi-GAN | 0.0075 | 0.0058 |
| FINV-SV | 0.0119 | 0.0096 |

## The point (answers the AC)
Generic TTO RECOVERS SHAPE under OOD (NFI's ICP-Chamfer even beats ours; ours is
Gaussian-centers vs NFI's clean SDF mesh). Baselines FAIL at correct OOD POSE: NFI's
pose estimate is accurate at ID pose (NVS 18-20) but collapses under SO(3) (NVS 14)
while shape stays fine. Ours' joint pose+latent search recovers BOTH => NVS ~20.
Separates "generic TTO benefit" (shape) from "this specific design" (pose under OOD).
