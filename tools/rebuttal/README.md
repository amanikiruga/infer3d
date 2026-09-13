# Rebuttal experiments

Everything added during the review discussion, with the per-object artifacts that back
each number. The written-up tables are in [`../../RESULTS.md`](../../RESULTS.md) Part II.

**Check every number without a GPU:**

```bash
PYTHONPATH=. python tools/rebuttal/report.py
```

That recomputes all of Part II from the JSON/CSV files shipped here — no datasets, no
model weights, a few seconds. Re-running the experiments themselves needs GPUs, the
datasets, and the baseline checkpoints (see *Re-running* below).

## Layout

| directory | experiment | key result |
|---|---|---|
| `baselines/` | pi-GAN, 3D-GAN-Inversion (EG3D+PTI), nerf-from-image (BRFI), FINV on SO(3) cars + RealCars | ours 19.23 vs BRFI 14.28 SO(3) NVS |
| `broad_prior/` | Equilibrium Matching (ImageNet) prior + Objaverse lifter, 25 occluded objects | CD 0.206 → 0.145, 25/25 wins |
| `ablation/` | leave-one-out component ablation, cars + CO3D | pose −3.10 dB, latent −2.27 dB, both −4.08 dB |
| `ood_objects/` | CO3D vases through the hydrants model | detection AUROC 0.988 |
| `multiobject/` | 2-chair scenes from a single image | held-out 18.24 dB with per-object registration |
| `analysis/` | ID trade-off, frequency analysis, optimization dynamics | ID gap is failure-driven, not uniform |
| `algorithm/` | Algorithm 1 symbol table + metric glossary | R=600, N=783, k_t 600→32→10→5 |

`MASTER_REBUTTAL.md`, `INDEX.md` and `PROTOCOL.md` are the original working documents
from the rebuttal period, kept for provenance.

## What each experiment answers

**`baselines/`** — the central question: is the gain just "test-time optimization", or
the specific design? Two OOD axes (SO(3) viewpoint on 25 ShapeNet cars; synthetic→real
appearance on 20 RealCars scenes) plus three controls that separate the two:

* an **oracle-pose** arm (hand each baseline the GT camera it failed to estimate),
* a **pose-removed shape** arm (ICP-aligned Chamfer — baselines do fine here, which
  localizes the gap to pose recovery),
* a **homefield** arm (each method on its own native setting — they all work there).

Entry points: `nfi_task_a.py`, `eg3d_task_a.py`, `finv_sv.py`, `homefield_*.py`,
`ours_nvs.py`; geometry via `*_geom.py` then `icp_chamfer.py`; scoring via
`score_task_a.py` / `build_scores.py`. Read `results/SUMMARY_task_a.md` and
`results/SUMMARY_task_b.md` first — they include the honest caveats about what the
RealCars Chamfer metric does and does not show.

**`broad_prior/`** — modularity: swap in a general image prior (EqM trained on ImageNet)
and a general lifter (Objaverse Splatter Image), change nothing else, and the method
still works on categories no component was specialized for. `invert_batch_objaverse.py`
drives it; `fuse_cd.py` scores.

**`ablation/`** — `ours_nmr_so3_ablate.py` (cars) and `ours_co3d_diffae_ablate.py` (CO3D)
run each leave-one-out arm with identical examples and initializations;
`collect_ablation.py` assembles the tables; `routing_sim.py` isolates the detector;
`chamfer_loo.py` adds the CO3D geometry view.

**`ood_objects/`** — `detector_ood_objects.py`. Content shift is the one OOD axis the
method cannot *fix* (the prior has no vases), so the claim is that it *detects* it.

**`multiobject/`** — `baseline_multichair.py` (feed-forward reference),
`oracle_align_perchair.py` (placement ceiling), `placement_crossval.py` (the honest
held-out number: fit registration on half the views, report the disjoint half).

**`analysis/`** — `frequency_gap.py`, `plot_dynamics.py`, `compare_variants.py`,
`perobj.py`, `blur_check.py`. These produced the finding that the in-distribution gap is
concentrated in a few inversion failures rather than spread over all objects.

## Re-running

The harnesses need the four upstream baseline repositories with their released
checkpoints, and the large per-object intermediates (rendered orbits, GT point clouds).
Both are resolved through `infer3d/config.py`:

```bash
export BASELINES_ROOT=/path/to/rebuttal_infer3d_baselines   # upstream clones + ckpts
export REBUTTAL_ASSETS=/path/to/rebuttal                    # per-object intermediates
export SHAPENET_CORE_V2=/path/to/ShapeNetCore.v2            # GT meshes for Chamfer
```

`BASELINES_ROOT` is expected to contain `nerf-from-image/`, `3D-GAN-Inversion/`,
`pi-GAN/`, `FINV/` and `eqm/EqM/`, each at its released commit with the released
weights. These are third-party repositories under their own licenses and are **not**
vendored here; clone them from upstream.

These scripts are research code preserved as run, lightly edited only to remove
hardcoded cluster paths. They are less polished than `infer3d/` and `tools/eval/`, and
several expect intermediates produced by an earlier script in the same directory — read
the module docstring before running one.
