# Tools

Runnable entry points. Everything reads its paths from [`infer3d/config.py`](../infer3d/config.py).

## No GPU, no data, no install

| command | what it does |
|---|---|
| `PYTHONPATH=. python3 tools/make_tables.py` | paper Tables 1 & 2 from bundled per-object CSVs |
| `PYTHONPATH=. python3 tools/rebuttal/report.py` | every rebuttal table from bundled artifacts |
| `scripts/verify.sh` | both of the above |

## Needs GPU + data + weights

| path | experiment |
|---|---|
| `check_config.py` | report which configured paths exist on this machine and what each unlocks (no GPU) |
| `smoke_test.py` | ~30 s install check: build lifter, load checkpoint, forward, CUDA render |
| `optimize_co3d_diffae.py`, `optimize_co3d_stylegan.py` | Stage A test-time optimization on CO3D (Table 1) |
| `optimize_nmr_se3.py`, `optimize_nmr_so3.py` | Stage A on ShapeNet-NMR (Table 2) |
| `eval/` | Stage B for CO3D: splats → gs2mesh → Sim(3)-ICP → Chamfer + novel-view metrics |
| `ood_detection/` | Tables 4, 6, 7: the DiffAE-only detector, AUROC vs step, F1 calibration sweep, adaptive stream |
| `realcars/` | Table 5: RealCars Chamfer against ARKit multi-view pseudo-GT |
| `re10k/` | Table 3: fisheye synthesis, blind calibration, CATSplat eval |
| `rebuttal/` | everything added during review (see its README) |

Stage A costs roughly 55 minutes of H100 time per CO3D object (783 iterations).
