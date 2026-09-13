# In-distribution trade-off, frequency analysis, optimization dynamics

Answers AC Priorities 2 and 3 (Reviewers XsZX and uH4P): *why* Infer3D gives up some
in-distribution accuracy, and what the optimization actually does over its 783 steps.

## The finding

The in-distribution gap is **not a uniform quality loss**. It is concentrated in a
minority of objects where inverting the lightweight 2D prior collapses to a wrong mode
(a blurry blob): 4/18 objects worse than 4 dB for the encoder-initialized DiffAE prior,
with a median gap of 1.7 dB on the rest; the latent-only StyleGAN prior fails more often
(14/18). Two alternative explanations were tested and ruled out — uniform blur (ours
matches the feed-forward lifter's sharpness on the successes) and registration/shading
(~0.2 dB).

Split by viewpoint, the feed-forward advantage is **+5.7 dB near the input view** but
only **+0.9 dB at the most occluded view**. The lifter inside Infer3D never sees the
input image directly, so it cannot copy visible detail; on views that actually test 3D
structure the gap nearly closes.

The failing objects are exactly the high-initial-loss cases the OOD detector flags
(AUROC 0.97 at 0.43 s), so the adaptive router sends them to the feed-forward path and
the penalty is neutralized in deployment.

## Dynamics

At feed-forward initialization the mean reconstruction loss is 0.030 (ID) versus 0.132
(OOD) — a 4.3× separation, which is what makes detection cheap. The OOD loss descends
monotonically, 0.132 → 0.062 over 300 steps, with 99.6% per-object monotonicity.
Schedule: 783 iterations in 6 stages (600-particle search → prune to 32 → 10 → 5 →
refine). Peak VRAM 37.5 GB at the default search (48–59 GB for larger ones); runtime is a
smooth quality knob from 2.7 to 24 minutes.

## Files

| file | what it does |
|---|---|
| `FINDINGS.md` | the frequency / failure-mode analysis, with its correction history |
| `DYNAMICS.md` | convergence, step counts, VRAM, runtime, routing |
| `RESPONSE_ID_AND_DYNAMICS.md`, `RESPONSE_FINAL.md` | the written responses |
| `render_id_pair_co3d.py`, `render_id_pair.py` | render ours + feed-forward at ID poses |
| `frequency_gap.py`, `plot_freq.py` | low-pass sweep and radial error power spectrum |
| `perobj.py` | per-object breakdown that located the failure objects |
| `blur_check.py` | sharpness control (rules out uniform blur) |
| `reg_control.py` | registration/shading control |
| `compare_variants.py` | compares the two `regenerate_ours_splats` variants |
| `plot_dynamics.py` | loss-vs-iteration and detector curves |
| `build_report.py`, `make_report_assets.py` | rebuild the visual HTML report |

`FINDINGS.md` keeps its correction history on purpose: two earlier drafts of this
analysis reached the wrong conclusion ("the gap is lost fine detail", then "the gap is
purely low-frequency"), and both were refuted by looking at error maps and per-object
breakdowns rather than one averaged statistic. The diagnostics that caught it are
`blur_check.py`, `perobj.py`, `reg_control.py` and `compare_variants.py`.

These scripts read cached optimization checkpoints and eval output; set
`REBUTTAL_ASSETS` and `SPLATTER_REPO_ROOT` (see `infer3d/config.py`) before running one.
