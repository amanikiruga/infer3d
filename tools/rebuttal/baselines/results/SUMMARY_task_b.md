# Task B — Appearance / synthetic->real shift, RealCars (20 ARKit scenes)

Single SAM2-masked 128px image (same input ours uses). All priors synthetic-only.
Metric: ICP-aligned Chamfer vs LiDAR pseudo-GT (m^2), paper's realcars pipeline.
align() pre-scales to GT diameter + FPFH+RANSAC + ICP => REMOVES pose & scale.

| Method | prior / representation | cd_pred->gt | cd_sym |
|---|---|---|---|
| pi-GAN | CARLA / full mesh (50k) | 0.088 | 0.112 |
| **Infer3D (ours)** | SRN + DINOv2 / single-view splats (16k) | 0.131 | 0.327 |
| EG3D-PTI | ShapeNet / full mesh | 0.154 | 0.384 |
| Splatter Image (feed-fwd) | SRN / single-view splats | 0.219 | -- |
| SF3D | Objaverse / mesh | 0.242 | -- |
| LGM | Objaverse / splats | 0.405 | -- |
| nerf-from-image | shapenet_cars / SDF | 0.491 | 0.431 |

(Splatter/SF3D/LGM from paper Table 5, same pipeline. ICP fit 0.83-0.99 all methods.)

## HONEST reading (do NOT claim ours-best-CD here)
This metric removes pose+scale and cd_pred->gt rewards emitting a COMPLETE plausible car,
so it interacts with representation:
- pi-GAN (0.088) and EG3D (0.154) score low by producing a clean generic full car whose
  surface overlaps a real car after alignment -- NOT by recovering the correct car/pose
  (pi-GAN assumes frontal; neither has a real-car prior). Their large win is a
  full-mesh + pose/scale-removed artifact.
- ours/Splatter are single-view splats (visible half -> large cd_sym); this metric
  penalizes representation, not fidelity.

## The fair, discriminative results
1. nerf-from-image (0.491) -- the direct NeRF-inversion analog to ours -- GENUINELY
   collapses, WORSE than feed-forward Splatter (0.219). Same representation family, so
   this is real geometry breakage: its pixel/latent inversion locks onto real texture
   the synthetic prior cannot represent. This is the clean answer to the AC on the
   appearance axis: generic optimization-based inversion does not help here.
2. Among comparable single-view methods, ours (0.131) > Splatter (0.219) > LGM (0.405):
   the paper Table 5 claim holds. Ours' DINOv2 semantic loss bypasses texture.
3. pi-GAN/EG3D's low CD is a metric/representation artifact (pose/scale removed); they do
   not recover the correct in-the-wild pose -- the same pose-sensitivity gap as §1 SO(3).
