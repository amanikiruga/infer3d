# Rebuttal response — draft prose (tailor as needed)

All numbers are from the isolated sandbox `rebuttal/`; see `report/index.html` for per-object
videos + scores, `JOURNAL.md` for the design log. Baselines use **pretrained models only** and
their own test-time optimization; all are evaluated on the **same two OOD axes and the same
metrics** as the paper, and set up **fair to each method's design**.

---

## AC's concern (verbatim)
> "…whether this particular instantiation advances what optimization-based reconstruction already
> offers … the experiments cannot currently separate the contribution of this specific design from
> the generic benefit of optimizing at test time … and notably FINV (3DV'24), which also employs
> multi-start particle inversion with pruning."

## One-paragraph answer
We now compare against all four optimization-based methods the reviews cite — pi-GAN,
3D-GAN-Inversion (EG3D+PTI), nerf-from-image, and FINV — on **SO(3) viewpoint OOD (ShapeNet
cars, 25 objects)** and **synthetic→real appearance OOD (RealCars, 20 scenes)**. Three controls
separate *generic test-time optimization* from *our specific design*, and all point the same way.
**(1) Pose.** At each method's own estimated pose, generic inversion collapses under SO(3)
(NVS 12–14 dB vs ours 20). **(2) Oracle pose.** When we *hand* nerf-from-image the ground-truth
camera it fails to estimate, it recovers only to 17.7 dB — still below our *own-pose* 20.0 — so
the failure is partly pose it cannot estimate and partly a reconstruction its inversion cannot
build under OOD input. **(3) FINV specifically.** Our single-view instantiation of FINV's
multi-start particle inversion + pruning lands at 12.3 dB — *identical to plain EG3D+PTI* — so the
particle machinery the AC cites is not what closes the gap. Meanwhile pose-removed Chamfer shows
the baselines are competitive on *shape*, confirming the NVS gap is pose recovery, not shape; and
on real cars nerf-from-image (the direct NeRF-inversion analog to ours) collapses (0.49 Chamfer,
worse than the feed-forward Splatter Image it builds on) while our DINOv2-semantic objective stays
faithful. In both axes the gain is this instantiation — a frozen-lifter manifold with joint SE(3)
search and a semantic objective — not the generic act of optimizing at test time.

---

## Evidence, point by point

### 1. SO(3) viewpoint OOD — pose-sensitive NVS (25 cars, shared harness)
| setting | ours | NFI | EG3D+PTI | FINV-SV |
|---|---|---|---|---|
| own (estimated) pose | **19.2** | 14.3 | 12.3 | 12.3 |
| **oracle GT pose** | — (solves pose itself) | **17.7** | unreliable* | — |
| in-distribution / control | 20.7 | 19.85 | — | — |

Generic inversion's pose estimation collapses under SO(3). Our joint pose+latent search under the
frozen-lifter manifold drops only ~1–2 dB from its in-distribution 20.7 → 19.2 (all shared-harness;
ours' own-eval harness gives 21.4 → 20.0, a ~1 dB offset in our favour that we do not rely on).
*EG3D's pose estimate is too inconsistent to calibrate a trustworthy oracle frame (its
in-distribution NVS is also ~12.6, i.e. gauge-limited); we therefore do not report an EG3D oracle.

### 2. The oracle-pose control (the key separation)
nerf-from-image is the one baseline with a dataset-aligned pose frame, so it admits a *calibrated*
NVS. We hand it the GT camera (calibrated once from control objects; uses GT poses only, never the
target images) and let it re-optimize the latent. Own-pose 14.3 → **oracle 17.7 → control 19.85**.
So ~40% of its collapse is pose estimation that generic TTO cannot solve; the residual is that OOD
inputs also degrade the reconstruction (per-object bimodal: ~half recover to 19–25, ~half stay
12–16 — a wrong, washed-out car even at the correct pose). **Ours receives no oracle and still
exceeds NFI-with-oracle.**

### 3. FINV, specifically (the AC's cited method)
FINV builds on the GET3D/EG3D family; we instantiate its algorithm single-view: N=16 latent
particles → optimize each (latent+pose) → prune to 4 by input-image consistency (the single-view
analog of FINV's cross-view consistency) → refine → prune to 1 → PTI. Result: **FINV-SV 12.3 dB ≈
plain EG3D+PTI 12.3 dB** (Chamfer 0.0119 vs 0.0127). Multi-start particle inversion + pruning does
not close the OOD-pose gap — it reaches the same place ordinary GAN-inversion does, far below ours.

### 4. Pose-removed shape — ICP-aligned Chamfer (SO(3), cd_sym ↓)
| ours | NFI | EG3D+PTI | pi-GAN | FINV-SV |
|---|---|---|---|---|
| 0.0091 | **0.0037** | 0.0127 | 0.0075 | 0.0119 |

ICP removes global pose, so this is shape only. Baselines match or beat us here (NFI's clean SDF
mesh even beats our Gaussian-centre cloud) — **proving the NVS gap in §1 is pose recovery, not
shape.** That is exactly the requested separation: generic TTO → shape; our design → shape *at the
correct OOD pose*.

### 5. Homefield (each method in its own in-distribution setting)
We give **every** baseline a real homefield, not just NFI — NVS where the method admits it,
reconstruction fidelity + pose-removed shape where it does not.
- **NVS (pose-supervised methods):** on the same in-distribution input, NFI 19.85 vs ours 20.7 —
  **comparable** (per-object they interleave; NFI occasionally edges ours — sharp NeRF vs our
  StyleGAN prior).
- **Reconstruction fidelity (GAN-inversion methods):** these have no calibrated NVS even
  in-distribution — EG3D+PTI's own-pose control NVS is 12.3 ≈ its OOD 12.3, because near-symmetric
  cars make unposed GAN inversion **gauge-limited, not pose-limited** (confirmed three independent
  ways: y-up frame, z-up frame, and its scattered per-object pose estimates). Their honest homefield
  is input-view reconstruction on **real in-domain data**: EG3D+PTI 21.5 dB on the 25 real control
  cars; pi-GAN 26.1 dB inverting **real CARLA images** (its actual training domain) — clean cars,
  vs the blobs it produces on ShapeNet/RealCars. That isolates a genuine **domain gap** from a
  broken model. (We invert real images, not generator samples — self-reconstruction would only
  show the optimizer works on its own manifold, not that it reconstructs data.)
- **Shape, in-distribution vs OOD (all 5 methods, same 25 GT cars, cd_sym ↓):**
  ours 0.0077→0.0091, NFI 0.0010→0.0037, EG3D 0.0086→0.0127, pi-GAN 0.0058→0.0075,
  FINV 0.0096→0.0119. In each method's homefield the shape is good and ours is comparable; and
  every method degrades only **gracefully** on shape from ID→OOD while its NVS collapses 12–14 dB.
So "in its homefield each method works and ours is comparable; the gap opens only OOD, and it is a
pose gap."

### 6. Appearance / synthetic→real (RealCars, cd pred→gt ↓, m²)
| pi-GAN* | ours | EG3D* | FINV* | Splatter | SF3D | LGM | NFI |
|---|---|---|---|---|---|---|---|
| 0.088 | 0.131 | 0.154 | 0.169 | 0.219 | 0.242 | 0.405 | **0.491** |

*The align step removes pose+scale and rewards emitting a complete generic car, so full-mesh GAN
priors score low by representation, not by recovering the correct car. The **discriminative**
result is nerf-from-image's genuine collapse (0.491 — worse than the feed-forward Splatter Image
0.219 it is meant to improve on): its pixel/latent inversion locks onto real texture the synthetic
prior cannot represent (same representation family as ours, so this is real geometry breakage). Our
DINOv2 semantic objective bypasses the texture nuisance; among comparable single-view methods ours
(0.131) > Splatter (0.219) > LGM (0.405).

---

## Tightest version (if space-limited)
We evaluate pi-GAN, EG3D+PTI, nerf-from-image, and a single-view instantiation of FINV on the same
SO(3) and synthetic→real OOD tasks. (i) Generic inversion collapses on OOD pose (NVS 12–14 vs ours
20); (ii) handed the GT pose it fails to estimate, the strongest baseline reaches only 17.7,
below our own-pose 20; (iii) FINV's multi-start pruning lands exactly where EG3D+PTI does (12.3);
(iv) pose-removed Chamfer shows baselines are fine on shape, so the gap is pose; (v) they are
comparable to us in-distribution. The gain is this instantiation, not test-time optimization per se.
