# FINV (Sun et al., 3DV'24) — analysis & why it is not a single-view apples-to-apples baseline

FINV = "Partial-view Object View Synthesis via Filtered Inversion". PTI inversion of a
pretrained 3D GAN (GET3D or EG3D) with **multi-start particle inversion + pruning** — the
ingredient the AC highlights as closest to ours.

## Inherent design constraints (verified in the FINV code, not a setup shortcoming)
1. **Multi-view by design.** `configs/global_config.py`: `num_train=5`; both dataset
   loaders gate on `if len(image_names) > 5:` before running. It consumes a *set* of
   partial views; a single view is off-config (only appears in a DEBUG block with
   1 optimization step).
2. **Requires known camera pose per view.** `coach.train(image_names, target_images,
   camera_params, instance_masks)` takes `camera_params` (pose+intrinsics) as a mandatory
   input; there is **no pose estimation/optimization** anywhere. This violates our
   single-image / no-GT-pose protocol.
3. **The "filtering" needs multiple views.** `num_active_ws=10`, `keep_best=3`: it prunes
   latent hypotheses using cross-view consistency. With one view there is nothing to
   filter across → it reduces to ordinary single-latent PTI inversion.
4. **Weights not shipped + brittle stack.** `pretrained_models/` absent; car ckpts are
   author snapshots (EG3D `eg3d-car-network-snapshot-003600.pkl`, GET3D `shapenet_car.pt`)
   not on disk; GET3D path needs nvdiffrast+kaolin built against torch 1.12 (our env is
   torch 2.9).

## Fair treatment for the rebuttal
Run single-view, FINV's algorithm **degenerates to PTI inversion of a 3D GAN** — which we
DO run and report as **3D-GAN-Inversion (EG3D + PTI)** (SO(3) shape CD 0.0127; RealCars
pred→gt 0.154). That is the single-view instantiation of FINV's core mechanism, evaluated
under identical conditions to ours. We therefore:
- report EG3D+PTI as the single-view stand-in for FINV's inversion, and
- state FINV's actual regime (multi-view + known pose), which our single-view / no-GT-pose
  method does not assume.

This is fair to FINV (we do not penalize it for a single-view/no-pose setting it was never
designed for) and directly answers the AC (its multi-start-particle idea, run single-view,
gives the EG3D+PTI numbers — it does not close the OOD-pose or appearance gap that our
specific design does).
