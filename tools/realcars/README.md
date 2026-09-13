# RealCars — in-the-wild appearance shift (Table 5)

Quantitative 3D evaluation on real ARKit car captures. Each method reconstructs from a
single 128×128 SAM-cropped image; the reference is a per-scene FastGS pseudo-ground-truth
fitted to the *full* multi-view capture with LiDAR-grade poses. Predictions are pre-scaled
to the GT diameter and aligned with FPFH+RANSAC→ICP before scoring, so the metric is
asymmetric Chamfer (pred→gt, m²) free of pose and scale gauge.

| script | role |
|---|---|
| `splats_to_ply_realcars.py`, `splats_to_ply_dino_baseline.py` | export predictions as PLY |
| `eval_chamfer_direct.py` | Infer3D and the Splatter-Image baseline |
| `eval_chamfer_lgm_sf3d.py` | LGM and SF3D baselines |
| `eval_chamfer_dino_baseline.py` | Splatter Image given our DINOv2 features + DA3 depth |
| `build_table_4col.py` | assemble the table from the per-scene CSVs |

```bash
PYTHONPATH=. python tools/realcars/build_table_4col.py      # needs only numpy
```

Per-scene CSVs ship in `results/`, so the table can be rebuilt without the dataset.
Reproduced: Infer3D **0.131** mean / **0.093** median vs Splatter Image 0.219 / 0.179 and
Splatter+DINOv2+depth 0.152 / 0.134 — matching the paper exactly. The LGM and SF3D
*means* differ from the published ones (0.491 vs 0.405; 0.268 vs 0.242) because the
committed table was built from an older per-scene CSV; their medians match and the
conclusion is unchanged.
