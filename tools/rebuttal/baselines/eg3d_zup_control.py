"""EG3D+PTI on the REAL in-distribution control set (25 NMR cars), z-up camera family.

The original control attempt used the y-up LookAtPoseSampler family, which cannot
represent the control cameras' roll (cars are z-up in this checkpoint's canonical
frame): pose estimates scattered, control NVS was gauge-limited (~12.6) and the W
calibration was declared unreliable. Importing homefield_gan patches E.make_c to the
z-up family (the true poses are inside the search family); everything else is exactly
eg3d_task_a: pose preheat + joint w/pose + PTI, own-pose NVS renders anchored at the
estimated pose, c2w_hat saved in meta.json for the W calibration pass.

  python eg3d_zup_control.py [--shard i --nshard n] [--bundles ... --out ...]
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, os, sys
WT = f"{_EXT}/splatter-image-rebuttal"
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import homefield_gan  # noqa: F401 — imported for the E.make_c z-up patch side effect
import eg3d_task_a as E
import lpips as lpips_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", default=f"{WT}/rebuttal/results/task_a_inputs_control")
    ap.add_argument("--out", default=f"{WT}/rebuttal/results/task_a_runs/eg3d_control_zup")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    lp = lpips_lib.LPIPS(net="vgg").to("cuda")
    ids = sorted(os.listdir(args.bundles))
    ids = [o for i, o in enumerate(ids) if i % args.nshard == args.shard]
    for oid in ids:
        odir = f"{args.out}/{oid}"
        if os.path.exists(f"{odir}/meta.json"):
            print(f"skip {oid}", flush=True); continue
        os.makedirs(odir, exist_ok=True)
        E.run_object(None, lp, f"{args.bundles}/{oid}", odir, [1, -1, -1], 350, 200)
        print(f"done {oid}", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
