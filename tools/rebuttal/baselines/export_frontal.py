"""Export a FRONTAL homefield bench: input = near-frontal test view (idx 4, az~-13 el~15),
targets = the test-view spiral. This is pi-GAN's / EG3D's in-distribution setting (frontal,
ID appearance) — no OOD rotation. Same 25 objects as the SO(3) bench. Reuses export_bundle."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import json, os, sys
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
os.environ.setdefault("SPLATTER_IMAGE_ROOT", WT)
os.environ.setdefault("SHAPENET_NMR_ROOT", f"{_EXT}/datasets/srn_nmr_classes")
from hydra import compose, initialize_config_dir
import export_task_a_inputs as E

FRONTAL_IDX = 4  # test-view spiral: az~-13 el~15 -> near-frontal-low


def main():
    with initialize_config_dir(version_base=None, config_dir=f"{WT}/configs"):
        cfg = compose(config_name="abs_config", overrides=["+dataset=shapenet-nmr"])
    bench = json.load(open(f"{WT}/rebuttal/results/task_a_bench25.json"))
    ids = [e[0] for e in bench]
    for oid in ids:
        out = f"{WT}/rebuttal/results/task_a_frontal/{oid}"
        n = E.export_bundle(oid, FRONTAL_IDX, "test", out, cfg)
        print(f"[frontal] {oid}: input=testview{FRONTAL_IDX} + {n} targets")
    print("DONE")


if __name__ == "__main__":
    main()
