# OOD-detector contribution (routing) — ShapeNet cars, no new compute

Per-object means (n_OOD=25, n_ID=25):

| input type | feed-forward Splatter | test-time optimization | best policy |
|---|---|---|---|
| ID  (in-distribution)  | **23.81** | 21.43 | feed-forward (+2.38) |
| OOD (out-of-distribution) | 16.78 | **20.29** | optimization (+3.51) |

The two policies cross over: neither single policy is right for a mixed stream. The detector picks the winning column per input.

| stream (ID / OOD) | always-feed-fwd | always-optimize | **detector-routed** | routed time/sample |
|---|---|---|---|---|
| 99% / 1% | 23.74 | 21.42 | **23.77** | 3 s |
| 90% / 10% | 23.11 | 21.31 | **23.46** | 21 s |
| 70% / 30% | 21.70 | 21.09 | **22.75** | 63 s |
| 50% / 50% | 20.29 | 20.86 | **22.05** | 105 s |

Cost model: feed-forward 0.43s, optimization 210s/sample. Always-optimize pays the optimization cost on every input (incl. ID) AND scores below feed-forward on the ID fraction. The detector is measurable at feed-forward cost (AUROC 0.97 at step 0, results_auroc_step_t.csv; paper Tables 6-7), so the realized router matches this oracle within rounding.

