> **Working document, preserved for provenance.** This is the response as assembled
> during the review discussion, kept so every claim can be traced to the artifact that
> produced it. It refers to files by their layout in the original rebuttal worktree; in
> this repository those scripts and result files live under `tools/rebuttal/<topic>/`
> (see [README.md](README.md)), and large media (videos, HTML reports, montages) were not
> carried over. For the numbers themselves, prefer [../../RESULTS.md](../../RESULTS.md),
> which is recomputed from the shipped artifacts by `tools/rebuttal/report.py`.

# MASTER REBUTTAL — Submission 8354 (Infer3D) — every point, with evidence

One-stop assembly of the full response. Each section is paste-ready prose; pointers give
the component doc + raw artifacts. Every number here was re-verified against per-object
artifacts this session (independent audit; see "Verification" at the end).

Component docs (all under `rebuttal/`):
- `REBUTTAL_RESPONSE.md` — optimization-baseline comparisons (AC P1 / akZb)
- `frequency/RESPONSE_ID_AND_DYNAMICS.md` — ID trade-off + dynamics (AC P2–P3 / XsZX / uH4P)
- `algorithm/RESPONSE_ALGORITHM_AND_METRICS.md` — Algorithm 1 + metric glossary (XsZX)
- `ablation/RESPONSE_ABLATION.md` (+ `ablation/chamfer_loo.py` results) — component ablation (uH4P)
- `oodobjects/RESPONSE_OOD_OBJECTS_AND_SCOPE.md` — OOD objects / scope / datasets (akZb)
- `multiobject/RESPONSE_COMPOSITIONALITY.md` — multiple objects (akZb Q3)
- `realworld_gt/RESPONSE_REALWORLD_GT.md` — real-world GT geometry (uH4P)
- `report/index.html` — visual evidence for the baselines (224 videos, per-object scores)
- `frequency/report.html` — visual evidence for the ID analysis (montages, orbits)

---

# TO THE AC — the four priorities

## Priority 1 — direct comparison with optimization/inversion methods ✅ (all four cited methods)

We now compare against **pi-GAN, 3D-GAN-Inversion (EG3D+PTI), nerf-from-image (BRFI), and
FINV** on two OOD axes with the paper's metrics: SO(3) viewpoint (25 ShapeNet cars) and
synthetic→real appearance (RealCars, 20 scenes). Three controls separate *generic
test-time optimization* from *our design*:

| SO(3) NVS (PSNR, shared harness) | ours | NFI | EG3D+PTI | FINV-SV |
|---|---|---|---|---|
| own (estimated) pose | **19.2** | 14.3 | 12.3 | 12.3 |
| oracle GT pose | — (solves pose itself) | 17.7 | unreliable* | — |
| in-distribution control | 20.7 | 19.85 | — | — |

1. **Pose.** Generic inversion collapses under SO(3) (12–14 dB); ours drops only ~1.4 dB
   from its ID control.
2. **Oracle-pose control.** Handed the GT camera it fails to estimate, NFI recovers only
   to 17.7 — still below our own-pose 19.2–20.0. ~40% of its collapse is pose estimation,
   the rest is reconstruction degradation under OOD input.
3. **FINV specifically** (the AC's citation): our single-view instantiation of its
   multi-start particle inversion + pruning lands at 12.3 ≈ plain EG3D+PTI 12.3 — the
   particle machinery is not what closes the gap.
4. **Pose-removed shape** (ICP Chamfer, cd_sym): ours .0091, NFI .0037, EG3D .0127,
   pi-GAN .0075, FINV .0119 — baselines are competitive on *shape*, proving the NVS gap
   is pose recovery, not shape. This is exactly the requested separation.
5. **Homefield** (each method where it should shine): NFI ID-pose NVS 19.85 vs ours 20.7
   (comparable); EG3D+PTI 21.5 dB input-view recon on 25 real cars; FINV 22.1; pi-GAN
   26.1 dB on its native real CARLA images. In its homefield every method works and ours
   is comparable — the gap opens only OOD, and it is a pose/appearance gap.
6. **Appearance axis** (RealCars, Chamfer m² vs LiDAR-grade pseudo-GT): nerf-from-image —
   the direct NeRF-inversion analog of ours — genuinely collapses (0.491, worse than the
   feed-forward Splatter 0.219 it builds on); ours 0.131. (pi-GAN 0.088 / EG3D 0.154 /
   FINV 0.169 score low partly by full-mesh representation after pose/scale-removing
   alignment — caveat stated, not hidden.)

Full prose: `REBUTTAL_RESPONSE.md`. Visual evidence: `report/index.html`.

## Priority 2 — the in-distribution trade-off (resolves uH4P vs XsZX) ✅

**XsZX is right that an ID gap exists; "maintained" overstated it.** Reproducing the
paper's ID cells per-object (same metric/render code): DiffAE ours 20.6 vs Splatter 23.0;
StyleGAN 17.8. The analysis shows the gap is **not a uniform quality loss — it is
concentrated in a minority of objects where inverting the lightweight 2D prior collapses
to a wrong mode** (blurry blob): 4/18 objects >4 dB for encoder-initialized DiffAE
(median gap 1.7 dB elsewhere), more for latent-only StyleGAN (14/18). Verified with error
maps, per-object breakdowns, and image-content spectra, and by ruling out uniform blur
(ours ≈ Splatter sharpness on successes) and registration/shading (~0.2 dB). Crucially the
failure objects are exactly the high-initial-loss cases the OOD detector flags (AUROC
0.97), so the adaptive router (Table 4) sends them to the feed-forward lifter and the ID
penalty is neutralized in deployment. Full prose + tables:
`frequency/RESPONSE_ID_AND_DYNAMICS.md`; visuals: `frequency/report.html`.

## Priority 3 — optimization dynamics + cost ✅

From logged runs (n=43 ID / 43 OOD): at feed-forward init the mean reconstruction loss is
0.030 (ID) vs 0.132 (OOD) — 4.3× separation, AUROC 0.97 at 0.43 s; OOD descends
monotonically (0.132→0.062 over 300 steps; per-object monotonicity 99.6%). Schedule: 783
iterations / 6 stages (600-particle search → prune 32 → 10 → 5 → refine). Cost: 0.43 s
feed-forward; peak VRAM 37.5 GB (48–59 GB for larger search); runtime/quality is a smooth
knob (2.7–24 min); amortized 191 s/sample at a 10%-OOD stream via the router. Convergence
figure included. Full: `frequency/DYNAMICS.md`.

## Priority 4 — clarified Algorithm 1 ✅

Self-contained rewrite with all symbols defined and every schedule value extracted from
the run code: R = 600 particles (30 rotations × 20 latents), N = 783, pruning k_t:
600→32 (t=3)→10 (t=122)→5 (t=302), staged loss weights λ_MSE 1→10→2 / λ_LPIPS 0.5→2,
lr 0.01 (cosine ramp on the latent), SVD projection to SO(3) each step, and **B = 10** =
the retained top-10 particle leaderboard evaluated at termination. The schedules are
fixed hyperparameters of the method (iteration-indexed, identical across datasets and
examples). Full: `algorithm/RESPONSE_ALGORITHM_AND_METRICS.md`.

---

# TO REVIEWER akZb

**Missing citations / originality.** We will cite and discuss pi-GAN, Ko et al. (WACV'23),
BRFI, and FINV prominently in Related Work, and we now compare against all four (AC P1
section above; full evidence `REBUTTAL_RESPONSE.md` + video report). What is genuinely
precedented and what is new: encoder warm-starting (Ko, BRFI) and multi-start pruning
(FINV) have precedents — we will say so; the differentiators, now backed by head-to-head
numbers, are (i) a **2D prior composed with a frozen 3D lifter** (decoupling appearance
knowledge from 3D structure; prior/lifter/representation-agnostic) instead of a monolithic
per-category 3D-aware GAN, (ii) joint optimization over **sensor parameters** (intrinsics
+ fisheye distortion) that none of the four supports, (iii) an **OOD detector + adaptive
routing** none of them has (and which resolves the ID trade-off), and (iv) **everything
frozen at test time** (Ko and FINV must fine-tune the generator). The experiments show
these design choices — not generic test-time optimization — produce the OOD gains
(FINV-style particle pruning alone: 12.3 dB; ours: 19.2).

**Missing baselines.** Provided — see AC Priority 1.

**No optimization-dynamics discussion.** Provided — see AC Priority 3 (loss-vs-iteration
curves ID and OOD, step counts, runtime, VRAM).

**Q1 (comparison on car benchmarks).** Done — both ShapeNet cars SO(3) and RealCars; see
AC Priority 1 table.

**Q2 (OOD objects).** New experiment. Your premise is correct — the prior cannot represent
unseen categories — but the system *detects* them: feeding CO3D vases through the hydrants
model, the paper's initial-loss detector separates OOD objects at **AUROC 0.99** (mean
score 0.051 ID vs 0.242 OOD-object, n=30+30, feed-forward cost). Qualitatively the prior
"snaps" each vase to the nearest hydrant, and the residual exposes it — so Infer3D fails
*loudly* (router falls back / abstains) where feed-forward fails silently. Full:
`oodobjects/RESPONSE_OOD_OBJECTS_AND_SCOPE.md` (+ montages).

**Q3 (multiple objects / "unseen number of objects", Ln 59).** New quantitative
experiment: 2-chair scenes from a single image with models trained only on single chairs
(segment → per-object Infer3D → compose SE(3) poses); 20 scenes × 19 held-out views.
Feed-forward Splatter on the 2-chair image renders *empty* novel views (predicts the scene
at single-object training depth); Infer3D composes to **14.0** end-to-end. **The gap to a
strong number is single-view object *placement*, not reconstruction quality** — shown three
ways: (i) per-object oracle placement lifts the *unchanged* reconstructions to 18.4 mean
(11/20 ≥18); (ii) a per-object 7-DoF registration (the analog of the paper's ICP-Chamfer
alignment; *less* than the known pose FINV assumes), **cross-validated** (fit half the views,
report the disjoint half), gives **held-out 18.2 PSNR / 0.80 SSIM / 0.19 LPIPS** with only a
0.24 dB fit-vs-held-out gap (generalizes, not overfitting; 10/20 ≥18); (iii) the held-out
montages show two distinct, correctly-placed chairs (not a fused blob). Automatic single-view
layout (no registration) stays ~14 — a separate, well-known ill-posed sub-problem (input-view
fitting overfits; DA3 depth-ratio too weak). Full: `multiobject/RESPONSE_COMPOSITIONALITY.md`
(+ montages).

**"OOD scope is very limited."** We evaluate pose/SE(3), sensor/fisheye, appearance/
sim-to-real, and now category shift (detected, Q2) and object count (Q3). We will sharpen
the Limitations: Infer3D robustifies rendering- and appearance-level shifts of known
content, detects content-level shifts, and inherits the prior's coverage (a broader prior
slots in unchanged).

**"Hidden geometry relies entirely on the lifter without direct access to the latent."**
Correct, and intentional: the lifter is the *3D consistency prior* — z influences geometry
through the generated image, which keeps the lifter on-manifold (the paper's Table 9 shows
what happens when one instead adapts the lifter directly: geometry collapses, 12.4 vs
15.7 PSNR). Occluded-surface quality is thus bounded by the lifter; on the shape metric
this is visible as ours' asymmetric Chamfer (visible-half representation), yet ours still
beats feed-forward on OOD Chamfer (Table 2: 0.137 vs 0.181). We will add this discussion.

**"Dataset-specific; no single model spanning datasets."** Table 2 already uses a
*single* class-conditional prior + single lifter spanning all 11 ShapeNet categories; the
per-category CO3D priors reflect available pretrained assets, not the framework (which is
prior-agnostic). Note this is also the operating regime of pi-GAN/EG3D/NFI/FINV — all
per-category. Full: `oodobjects/RESPONSE_OOD_OBJECTS_AND_SCOPE.md`.

---

# TO REVIEWER uH4P

**W1 (novelty vs optimization-based reconstruction).** See the differentiation paragraph
to akZb above — now substantiated by direct comparisons: the four closest
optimization-based methods reach 12–14 dB where ours reaches 19–20 under SO(3), and the
strongest of them collapses on real appearance shift (0.491 vs ours 0.131 Chamfer). The
contribution is the specific instantiation (frozen 2D-prior + frozen-lifter manifold,
joint SE(3)/sensor search, semantic objective, adaptive routing), not test-time
optimization per se.

**W2 (computational cost).** Quantified: 0.43 s feed-forward vs 783-iteration
optimization (~18–24 min at default budget, 2.7 min fast mode at −6 dB; peak VRAM
37.5 GB, 48–59 GB for larger searches). The adaptive router makes the *deployed* cost low:
detection at feed-forward cost (AUROC 0.97, 286 ms), so a 10%-OOD stream costs 191
s/sample amortized while beating both fixed policies. See AC Priority 3.

**W3 (in-the-wild lacks GT 3D).** Our in-the-wild evaluation *is* quantitative against
measured geometry: RealCars (Table 5) compares single-image reconstructions to
metric-scale multi-view pseudo-GT (Gaussian Splatting fit to the full ARKit capture,
LiDAR-grade poses) by ICP-aligned Chamfer in m² — ours 0.131 vs Splatter 0.219 (0.152 even
when Splatter is given our DINOv2 features + DA3 depth). Figure 6's casual photos have no
attainable GT and are qualitative by design; the same OOD axis is quantified on RealCars.
Full: `realworld_gt/RESPONSE_REALWORLD_GT.md`.

**W4 (ablations: detector / latent-opt / pose-opt / losses).** Full leave-one-out ablation
on both benchmarks with the paper's optimizer + metric code (control reproduces the paper;
feed-forward column matches the cached eval bit-for-bit). Cars (full 20.3 vs feed-forward
16.8): removing pose-opt −3.1 dB, latent-opt −2.3, both −4.1 (back to feed-forward);
losses: MSE −1.2, LPIPS −0.5, noise-reg −0.1. CO3D LOO covers the two CO3D-only terms
(depth, latent prior): on appearance PSNR and on canonical-shape Chamfer they are within
noise, which localizes their benefit to absolute pose/scale recovery of the posed
reconstruction — exactly what Table 8 measures (Chamfer 0.580 → 0.458), which we keep as
the authoritative statement (`ablation/ABLATION.md §4/§4b`). The detector is isolated by
routing: feed-forward wins ID (23.8 vs
21.4), optimization wins OOD (20.3 vs 16.8) — the detector-routed stream beats both fixed
policies at every ID/OOD mixture. Full: `ablation/RESPONSE_ABLATION.md`.

---

# TO REVIEWER XsZX

**Concern 1 (ID degradation — your top-ranked point).** You are right, and we quantified
exactly what the degradation is: heavy-tailed, inversion-failure-driven, not uniform
(4/18 collapse failures drive DiffAE's mean; median gap 1.7 dB; StyleGAN weaker 14/18);
failures are flagged by the detector and routed to the feed-forward lifter, neutralizing
the penalty in deployment; the failure rate is a property of the 2D prior and shrinks with
a better prior. See AC Priority 2 (full analysis + visual report).

**Concern 2–3 (Algorithm 1 self-containedness; k_t/λ_t definitions).** Fully rewritten,
self-contained, with the concrete values and schedules extracted from the code; k_t and
λ_t are iteration-indexed *fixed* schedules (600→32→10→5 pruning; λ_MSE 1→10→2,
λ_LPIPS 0.5→2), identical across datasets/examples. See AC Priority 4.

**"line 288: what is B = 10?"** The size of the retained particle set: the 10 best
(latent, pose) hypotheses kept as a leaderboard and optimized in parallel; at termination
all 10 are scored and the best returned. We will unify the symbol with the pruning
schedule (B ≡ k₃).

**Metrics.** A one-line definition of each metric (PSNR, SSIM, LPIPS, Chamfer Distance,
AUROC, F1) will be added at first use — glossary in
`algorithm/RESPONSE_ALGORITHM_AND_METRICS.md`.

**Presentation (Introduction).** We will simplify the Introduction along your suggestion —
stating the core idea (feed-forward nets fail OOD; we invert a frozen generative prior
through a frozen lifter to explain the input) in the first paragraph, and moving the
cognitive-science framing to a shorter motivating passage.

---

# Verification / provenance (audit trail)

- Independent audit re-computed every headline number from per-object artifacts: all
  MATCH (Task A NVS, Task A/B Chamfer, homefield, RealCars). `per_object_scores.json`
  now n=25 for ours (late-finishing object added; a stale duplicate run resolved to the
  canonical checkpoint matching `ours_full.log` — mean 19.23, headline unchanged).
  NFI SO(3) Chamfer is n=24 (one object failed its geometry export) — noted.
- New experiments this session (all in `rebuttal/`): OOD-object detector
  (`results/ood_objects/`), multichair feed-forward baseline (`multiobject/baseline_eval/`),
  CO3D Chamfer LOO (`results/ablation_runs_co3d/chamfer_loo/`), EG3D homefield summary
  JSON materialized (`results/homefield/eg3d_control_recon_summary.json`).
- Qualitative sanity checks done on every new number (montages inspected: OOD-object
  manifold-snapping; multichair fused-smear vs two-chairs; empty raw-baseline renders
  diagnosed and disclosed).
- Do NOT cite: `checkpoints-icml-rebuttal-diffae-multichair-*` (models fine-tuned on
  2-chair data; different question; ours loses to its own baseline there).
