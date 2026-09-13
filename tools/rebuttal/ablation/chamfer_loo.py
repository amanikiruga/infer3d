"""Geometry (Chamfer) LOO for the two CO3D-only loss terms (depth, latent prior) —
the follow-up ABLATION.md flagged: their PSNR LOO was within noise because they are
GEOMETRY regularizers; this measures shape directly.

For arms {b0_control, no_depth, no_latent_prior} x 18 objects (17 with pseudo-GT):
load the arm's optimization checkpoint -> lift best_input_image with the hydrants
Splatter lifter (same as the run) -> opacity-filtered Gaussian centers -> ICP-aligned
symmetric Chamfer vs the FastGS pseudo-GT point cloud (align/chamfer_sym reused verbatim
from the validated rebuttal geometry pipeline icp_chamfer.py; ICP removes pose/scale so
only shape is measured). All arms share identical treatment => paired LOO deltas.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, json, os, sys
import numpy as np
import torch

WT = f"{_EXT}/splatter-image-rebuttal"
ORIG = f"{_EXT}/splatter-image"
sys.path.insert(0, WT)
sys.path.insert(0, f"{WT}/experiments/stylegan3-ours-co3d")
sys.path.insert(0, f"{WT}/rebuttal/baselines")
import eval_renderings_with_icp_batch as E          # noqa: E402
from icp_chamfer import align, chamfer_sym          # noqa: E402
from hydra import initialize_config_dir, compose    # noqa: E402
import hydra                                        # noqa: E402
from plyfile import PlyData                         # noqa: E402

device = "cuda"
ARMS = ["b0_control", "no_depth", "no_latent_prior"]
RUNS = f"{WT}/rebuttal/results/ablation_runs_co3d"
OUT = f"{WT}/rebuttal/results/ablation_runs_co3d/chamfer_loo"
GT_LIST = f"{ORIG}/eval_output/checkpoints-icml-diffae-co3d-hydrants-baseline-se3-new-stylegan-10-depth-ood-final/lists/pseudo_gt.txt"
HYDRANTS_LIFTER = f"{ORIG}/experiments_out/2025-10-07/16-13-13/model_latest.pth"
N_PTS = 50000


def load_ply_xyz(path, n=N_PTS):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if len(xyz) > n:
        xyz = xyz[np.random.RandomState(0).choice(len(xyz), n, replace=False)]
    return xyz


def main():
    os.makedirs(OUT, exist_ok=True)
    gt_map = {}
    for ln in open(GT_LIST):
        ln = ln.strip()
        if ln:
            gt_map[os.path.basename(ln).replace("_pointcloud.ply", "")] = ln

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{WT}/configs")
    cfg = compose(config_name="abs_config", overrides=[
        "general.split=0", "general.total_splits=1", "+dataset=hydrants",
        "general.prefix=chamfer-loo", "abs=diffae_abs",
        f"opt.pretrained_ckpt={HYDRANTS_LIFTER}",
        "general.data_example_ids_path=not_needed.json"])
    gp = E.load_gaussian_predictor(cfg, device)
    ds = E.CO3DDataset(cfg, "test", use_hq=False)

    per_arm = {a: {} for a in ARMS}
    for arm in ARMS:
        for ck_path in sorted(glob.glob(f"{RUNS}/{arm}/shard_*/*/topk_best_everything_latest.pth")):
            oid = os.path.basename(os.path.dirname(ck_path)).split("-")[0]
            if oid not in gt_map:
                print(f"skip {arm}/{oid}: no pseudo-GT")
                continue
            npy = f"{OUT}/{arm}_{oid}.npy"
            if not os.path.exists(npy):
                c = torch.load(ck_path, map_location=device, weights_only=False)
                idx, _ = ds.find_index_for_sequence_prefix(c["example_id"])
                od = ds[idx]
                od = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in od.items()}
                oii = c.get("ood_image_index", 0)
                img = c["best_input_image"].to(device)
                if img.dim() == 4:
                    img = img.squeeze(0)
                img = img[:3]
                n_in = cfg.data.input_images
                inp = torch.cat([img.unsqueeze(0).unsqueeze(1),
                                 od["origin_distances"][:, oii].unsqueeze(1)], dim=2)
                foc = od["focals_pixels"][:, oii].unsqueeze(1) if "focals_pixels" in od else None
                with torch.no_grad():
                    sp = gp(inp, od["view_to_world_transforms"][:1, oii:oii + n_in],
                            od["source_cv2wT_quat"][:1, oii:oii + n_in], foc)
                sp = {k: v[0] for k, v in sp.items()}
                xyz = sp["xyz"].detach().cpu().numpy()
                op = sp["opacity"]
                op = torch.sigmoid(op[:, 0] if op.dim() > 1 else op).detach().cpu().numpy()
                keep = op > 0.05
                xyz = xyz[keep] if keep.sum() > 100 else xyz
                if len(xyz) > N_PTS:
                    xyz = xyz[np.random.RandomState(0).choice(len(xyz), N_PTS, replace=False)]
                np.save(npy, xyz.astype(np.float32))
            xyz = np.load(npy)
            gt = load_ply_xyz(gt_map[oid])
            aligned, fit = align(xyz, gt)
            cds, c_p2g, c_g2p = chamfer_sym(aligned, gt)
            per_arm[arm][oid] = {"cd_sym": float(cds), "cd_pred2gt": float(c_p2g), "icp_fit": float(fit)}
            print(f"{arm}/{oid}: cd_sym {cds:.4f} p2g {c_p2g:.4f} fit {fit:.2f}")

    common = sorted(set.intersection(*[set(per_arm[a]) for a in ARMS]))
    summary = {"n_common": len(common), "objects": common}
    for a in ARMS:
        v = np.array([per_arm[a][o]["cd_sym"] for o in common])
        summary[a] = {"cd_sym_mean": float(v.mean()), "cd_sym_median": float(np.median(v))}
    for a in ARMS[1:]:
        d = np.array([per_arm[a][o]["cd_sym"] - per_arm["b0_control"][o]["cd_sym"] for o in common])
        summary[f"delta_{a}"] = {"paired_mean": float(d.mean()),
                                 "paired_sem": float(d.std(ddof=1) / np.sqrt(len(d))),
                                 "worse_count": int((d > 0).sum())}
    print(json.dumps(summary, indent=1))
    json.dump({"per_arm": per_arm, "summary": summary},
              open(f"{OUT}/chamfer_loo_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
