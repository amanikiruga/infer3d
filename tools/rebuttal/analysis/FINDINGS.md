# In-distribution gap — what it actually is (rebuttal, XsZX / AC priority 2)

**This supersedes an earlier draft that claimed the gap was "purely low-frequency, fine
detail matched." Qualitative inspection (error maps + per-object breakdown + image-content
spectra) REFUTED that. The corrected, visually-grounded finding is below.**

Metric/render code is the paper's own; the ID cells reproduce (DiffAE ID ours 20.58 vs
paper 20.55; Splatter 23.0; StyleGAN ours 17.76). Evidence assets in `report_assets/` and
the HTML report `report.html`.

## What the ID gap actually is

The gap is **driven by 2D-prior inversion quality on a minority of hard objects**, not by a
uniform quality loss:

| prior | median gap | mean gap | objects with gap >4 dB (failures) |
|---|---|---|---|
| DiffAE (encoder) | 1.7 dB | 2.6 dB | **4 / 18** |
| StyleGAN (latent-only) | 5.2 dB | 5.5 dB | 14 / 18 |

- **On objects where inversion succeeds**, Infer3D ≈ Splatter (median DiffAE gap 1.7 dB;
  sharpness ratio ours/Splatter ≈ 1.0). Both are soft relative to the real GT photo —
  that softness is the 128² Gaussian-splat render pipeline, shared by both methods, not
  specific to ours (`montage_typical.png`).
- **On failure objects**, ours COLLAPSES: the generator can't invert the input (e.g. an
  unusual white hydrant) and optimization lands in a wrong mode → a blurry blob, while the
  feed-forward lifter still reconstructs a recognizable object (`montage_failure.png`,
  `orbit_failure.mp4`). These few objects drive the mean (`gap_distribution.png`).
- **DiffAE's encoder makes this much rarer** (4/18 vs 14/18) — the gap is a property of the
  2D prior's inversion fidelity, and improves directly with a better prior.

## What it is NOT (things I checked and ruled out)

- **NOT primarily fine-detail blur uniformly.** On successful objects ours is not
  systematically blurrier than Splatter.
- **NOT registration or shading.** A per-image small-shift + per-channel gain/bias search
  closes only ~0.2 dB of the 3 dB cars gap.
- **NOT what the averaged *error* power spectrum suggested.** ⟨|FFT(err)|²⟩ is dominated by
  the shared object silhouette and by GT texture both methods miss, so it wrongly read
  "high-freq matched." The **image-content** spectrum + per-object + error maps are the
  correct tools and tell the real (failure-driven) story. **Lesson: trust the pixels.**

## Rebuttal implication (connects to the paper, does not contradict it)

The ID cost is concentrated in exactly the **high-reconstruction-loss failure cases the OOD
detector flags** (Sec 3.5, Table 6, AUROC 0.97). The adaptive router (Table 4) sends those
to the feed-forward lifter, so the ID penalty is neutralized in deployment; on the bulk of
inputs the gap is ~1–2 dB (DiffAE). A stronger 2D prior shrinks the failure rate directly.

### Honest takeaway sentence
> The in-distribution gap is not a uniform quality loss: it is concentrated in a minority
> of objects where inverting the lightweight 2D prior lands in a wrong mode and the
> reconstruction collapses (4/18 for encoder-initialized DiffAE, more for latent-only
> StyleGAN). These are precisely the high-loss cases our OOD detector flags and the router
> sends to the feed-forward lifter, so the in-distribution penalty is neutralized in
> deployment; on the remaining objects Infer3D matches the feed-forward lifter to ~1–2 dB.
