"""Report which configured paths exist on this machine, and what each one is needed for.

Run this after editing `.env` to see what you can and cannot reproduce yet. Nothing here
is required to check the reported numbers -- `scripts/verify.sh` needs none of it.

    PYTHONPATH=. python tools/check_config.py
"""
import os

from infer3d import config as C

# (attribute, env var, what it unlocks)
ENTRIES = [
    ("RUNS_ROOT",             "INFER3D_RUNS_ROOT",     "where runs are written (must be WRITABLE)"),
    ("ASSETS_DIR",            "INFER3D_ASSETS_DIR",    "bundled splits/intrinsics/eval CSVs"),
    ("CO3D_DATASET_ROOT",     "CO3D_DATASET_ROOT",     "Tables 1, 4, 6-9"),
    ("SHAPENET_NMR_ROOT",     "SHAPENET_NMR_ROOT",     "Table 2"),
    ("SHAPENET_NMR_SE3_ROOT", "SHAPENET_NMR_SE3_ROOT", "Table 2, OOD SE(3)"),
    ("DIFFAE_ROOT",           "DIFFAE_ROOT",           "DiffAE prior (CO3D, RE10K)"),
    ("STYLEGAN3_ROOT",        "STYLEGAN3_ROOT",        "StyleGAN prior (CO3D, ShapeNet)"),
    ("SPLATTER_REPO_ROOT",    "SPLATTER_REPO_ROOT",    "frozen lifters + cached eval CSVs"),
    ("CATSPLAT_ROOT",         "CATSPLAT_ROOT",         "Table 3 (RE10K lifter)"),
    ("CATSPLAT_CKPT",         "CATSPLAT_CKPT",         "Table 3 (RE10K lifter weights)"),
    ("RE10K_CLIPS",           "RE10K_CLIPS",           "Table 3 (built by tools/re10k/build_clips.py)"),
    ("REALCARS_ROOT",         "REALCARS_ROOT",         "Table 5"),
    ("GS2MESH_ROOT",          "GS2MESH_ROOT",          "Stage B meshing (CO3D Chamfer)"),
    ("SHAPENET_CORE_V2",      "SHAPENET_CORE_V2",      "rebuttal Task A Chamfer ground truth"),
    ("BASELINES_ROOT",        "BASELINES_ROOT",        "rebuttal inversion baselines"),
    ("REBUTTAL_ASSETS",       "REBUTTAL_ASSETS",       "rebuttal per-object intermediates"),
]


def main():
    print(f"{'status':7s} {'setting':22s} {'from':9s} path / purpose")
    print("-" * 100)
    missing = []
    for attr, env, purpose in ENTRIES:
        path = getattr(C, attr)
        ok = os.path.exists(path)
        src = "env" if os.getenv(env) else "default"
        print(f"{'OK' if ok else 'MISSING':7s} {attr:22s} {src:9s} {path}")
        print(f"{'':40s} -> {purpose}")
        if not ok:
            missing.append(attr)

    print("\n--- frozen lifter checkpoints ---")
    for name, path in C.LIFTER_CKPTS.items():
        print(f"{'OK' if os.path.exists(path) else 'MISSING':7s} {name:22s} {path}")
    print("\n--- StyleGAN checkpoints ---")
    for name, path in C.STYLEGAN_CKPTS.items():
        print(f"{'OK' if os.path.exists(path) else 'MISSING':7s} {name:22s} {path}")

    writable = os.access(C.RUNS_ROOT, os.W_OK) if os.path.exists(C.RUNS_ROOT) else \
        os.access(os.path.dirname(C.RUNS_ROOT) or ".", os.W_OK)
    print(f"\nRUNS_ROOT writable: {writable}")
    if missing:
        print(f"\n{len(missing)} path(s) missing: {', '.join(missing)}")
        print("That is fine unless you need the experiments they unlock (see DATA.md).")
    else:
        print("\nAll configured paths resolve.")


if __name__ == "__main__":
    main()
