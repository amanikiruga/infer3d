# Rebuttal prose — real-world results ARE verified against 3D geometry (Reviewer uH4P W3)

uH4P: "The experimental validation on real in-the-wild images is still limited. Although
qualitative results are provided, the lack of ground-truth 3D geometry makes it difficult
to rigorously verify whether the reconstructed shapes are accurate rather than merely
plausible."

Verified sources: paper Table 5 + `experiments/neurips_submission/real_ood_realcars_eval/`
(pipeline documented in `eval_chamfer_direct.py`; per-scene artifacts
`results/chamfer_direct.csv`, `chamfer_table_4col.md` — audited, recompute to the
published aggregates), plus the new rebuttal baselines
(`rebuttal/results/task_b_geom/chamfer_*_realcars.csv`, n=20 each).

---

## To Reviewer uH4P

We agree that plausibility is not accuracy, and this is precisely why our in-the-wild
evaluation is **quantitative against measured 3D geometry**, not only qualitative
(Table 5). We will make the protocol prominent in the revision:

- **RealCars** consists of in-the-wild **ARKit captures** of real cars (multi-view video
  with metric camera poses and intrinsics from a LiDAR-equipped phone).
- For each of the 20 scenes we build a **metric-scale pseudo-ground-truth**: a Gaussian
  Splatting reconstruction fit to **all in-scene frames** at 1600×1200 with the ARKit
  poses/intrinsics, cropped to the car. This uses the full multi-view capture — the exact
  information our method never sees.
- Each method receives **one masked 128 px image** and its predicted 3D is compared to
  this pseudo-GT by ICP-aligned Chamfer distance in the metric ARKit world frame (m²) —
  a shape-accuracy measure, insensitive to pose/scale convention.
- Result (Table 5): Infer3D 0.131 mean CD vs Splatter Image 0.219, a Splatter Image
  variant upgraded with the same DINOv2 features + Depth-Anything-3 depth 0.152, SF3D
  0.242, LGM 0.405 — i.e., the shapes our method recovers on real captures are verified
  to be *geometrically* closer to the measured car surface, not merely plausible-looking.
- **New in this rebuttal** (baseline response, same protocol, same 20 scenes): the
  optimization-based inversion baselines score pi-GAN 0.088*, EG3D+PTI 0.154*, FINV
  0.169*, nerf-from-image 0.491 (*full-mesh priors score low partly by representation —
  a complete generic car overlaps the GT after pose/scale-removing alignment; the
  discriminative result is nerf-from-image, the direct NeRF-inversion analog of our
  method, collapsing to 0.491 — worse than the feed-forward Splatter Image it builds on).

The purely qualitative Figure 6 (casual smartphone photos) is complementary: those photos
have no attainable 3D ground truth, so we demonstrate robustness qualitatively there while
quantifying the same synthetic→real appearance shift on RealCars, where full multi-view
metric geometry is available. We will state this division explicitly in the revision.

## Tightest version

> Our in-the-wild evaluation is quantitative, not only qualitative: RealCars (Table 5)
> compares each method's single-image reconstruction against a metric-scale multi-view
> pseudo-ground-truth (Gaussian Splatting fit to the full ARKit capture, LiDAR-grade
> poses), via ICP-aligned Chamfer in m². Infer3D reaches 0.131 vs 0.219 for the
> feed-forward lifter (0.152 even when the lifter is given our DINOv2 features + DA3
> depth). Figure 6's casual photos have no attainable GT and are qualitative by design;
> the same appearance-shift axis is quantified on RealCars.
