"""OOD-OBJECT detection experiment (akZb Q2: "How would the method work for OOD objects?").

Protocol = paper Sec 3.5 detector, applied to a category never seen by the prior:
feed CO3D VASES (test split, view-0 = in-distribution POSE, only the OBJECT is OOD)
through the HYDRANTS model (DiffAE hydrants encoder+decoder -> hydrants Splatter lifter
-> render at the input camera), score = initial reconstruction error (MSE + LPIPS)
against the input. Compare score distribution vs ID hydrants (same protocol) -> AUROC.

Everything frozen, one encode-decode pass per object (the paper's 286 ms detector path).
Outputs: rebuttal/results/ood_objects/{scores.json, montage_{hydrants,vases}.png}
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, os, sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

WT = f"{_EXT}/splatter-image-rebuttal"
ORIG = f"{_EXT}/splatter-image"
DIFFAE_ROOT = f"{_EXT}/diffae"
sys.path.insert(0, WT)
sys.path.insert(0, f"{WT}/experiments/stylegan3-ours-co3d")
sys.path.append(DIFFAE_ROOT)

import eval_renderings_with_icp_batch as E  # noqa: E402
from hydra import initialize_config_dir, compose  # noqa: E402
import hydra  # noqa: E402
from templates import co3d_hydrants_autoenc_128, LitModel  # noqa: E402
import lpips as lpips_lib  # noqa: E402

device = "cuda"
HYDRANTS_LIFTER = f"{ORIG}/experiments_out/2025-10-07/16-13-13/model_latest.pth"
OUT = f"{WT}/rebuttal/results/ood_objects"


def make_cfg(dataset):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{WT}/configs")
    return compose(config_name="abs_config", overrides=[
        "general.split=0", "general.total_splits=1", f"+dataset={dataset}",
        "general.prefix=oodobj", "abs=diffae_abs",
        f"opt.pretrained_ckpt={HYDRANTS_LIFTER}",
        "general.data_example_ids_path=not_needed.json"])


class DiffAE(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        m = LitModel(conf)
        st = torch.load(f"{DIFFAE_ROOT}/checkpoints/{conf.name}/last.ckpt", map_location="cpu")
        m.load_state_dict(st["state_dict"], strict=False)
        m.ema_model.eval(); m.ema_model.to(device)
        m.model.eval(); m.model.to(device)
        self.m = m

    @torch.no_grad()
    def encode(self, img):           # img in [-1,1], [B,3,128,128]
        return self.m.encode(img)

    @torch.no_grad()
    def decode(self, xT, cond, T=12):  # -> [0,1]
        return self.m.render(xT, cond, T=T)


@torch.no_grad()
def initial_hypothesis(item, dae, gp, cfg_model, xT, lpips_fn, norm):
    """One forward pass: I -> E -> G -> Phi -> render@input-cam. Returns scores + images."""
    d = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) for k, v in item.items()}
    I = d["gt_images"][:, 0, :3]                      # [1,3,H,W] in [0,1]
    z = dae.encode(norm(I))
    I_gen = dae.decode(xT, z).clamp(0, 1)             # [1,3,128,128]
    od = d["origin_distances"][:, 0]                  # [1,1,H,W]
    inp = torch.cat([I_gen.unsqueeze(1), od.unsqueeze(1)], dim=2)
    n_in = cfg_model.data.input_images
    foc = d["focals_pixels"][:, 0].unsqueeze(1) if "focals_pixels" in d else None
    splats = gp(inp, d["view_to_world_transforms"][:, 0:n_in],
                d["source_cv2wT_quat"][:, 0:n_in], foc)
    splats = {k: v[0] for k, v in splats.items()}
    bg = torch.tensor([1., 1, 1] if cfg_model.data.white_background else [0., 0, 0], device=device)
    fp = d["focals_pixels"][0, 0] if "focals_pixels" in d else None
    R = E.render_predicted(splats, d["world_view_transforms"][0, 0:1],
                           d["full_proj_transforms"][0, 0:1], d["camera_centers"][0, 0:1],
                           bg, cfg_model, focals_pixels=fp)["render"].clamp(0, 1).unsqueeze(0)
    mse = float(torch.mean((R - I) ** 2))
    lp = float(lpips_fn(R * 2 - 1, I * 2 - 1))
    mse_dec = float(torch.mean((I_gen - I) ** 2))
    return {"mse": mse, "lpips": lp, "score": mse + lp, "mse_decode": mse_dec}, \
        (I[0], I_gen[0], R[0])


def montage(rows, path):
    strips = []
    for (I, G, R), cap in rows:
        s = torch.cat([I, G, R], dim=2).permute(1, 2, 0).cpu().numpy()
        strips.append((np.clip(s, 0, 1) * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(strips, axis=0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cfg_model = make_cfg("hydrants")
    gp = E.load_gaussian_predictor(cfg_model, device)
    dae = DiffAE(co3d_hydrants_autoenc_128())
    lpips_fn = lpips_lib.LPIPS(net="vgg").to(device).eval()
    norm = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128),
                               transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    g = torch.Generator(device="cpu").manual_seed(0)
    xT = torch.randn(1, 3, 128, 128, generator=g).to(device)

    results = {}
    for group, ds_name in [("hydrants_id", "hydrants"), ("vases_oodobj", "vases")]:
        cfg_d = make_cfg(ds_name)
        ds = E.CO3DDataset(cfg_d, "test", use_hq=False)
        n = min(args.limit, len(ds))
        rows, scores = [], []
        for i in range(n):
            try:
                item = ds[i]
                sc, imgs = initial_hypothesis(item, dae, gp, cfg_model, xT, lpips_fn, norm)
                sc["example_id"] = item.get("example_id", str(i)) if isinstance(item, dict) else str(i)
                scores.append(sc)
                rows.append((imgs, sc))
                print(f"{group} {i}: mse {sc['mse']:.4f} lpips {sc['lpips']:.4f} score {sc['score']:.4f}")
            except Exception as ex:
                import traceback; traceback.print_exc(); print(f"skip {group} {i}: {ex}")
        results[group] = scores
        montage(rows[:8], f"{OUT}/montage_{group}.png")

    a = np.array([s["score"] for s in results["hydrants_id"]])
    b = np.array([s["score"] for s in results["vases_oodobj"]])

    def auroc(pos, neg):  # pos = OOD (higher score)
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.ones_like(pos), np.zeros_like(neg)]
        return float(roc_auc_score(y, np.r_[pos, neg]))

    summary = {
        "n_id": len(a), "n_ood": len(b),
        "id_score_mean": float(a.mean()), "ood_score_mean": float(b.mean()),
        "auroc_score": auroc(b, a),
        "auroc_mse": auroc(np.array([s["mse"] for s in results["vases_oodobj"]]),
                           np.array([s["mse"] for s in results["hydrants_id"]])),
        "auroc_lpips": auroc(np.array([s["lpips"] for s in results["vases_oodobj"]]),
                             np.array([s["lpips"] for s in results["hydrants_id"]])),
    }
    print(json.dumps(summary, indent=1))
    json.dump({"summary": summary, "scores": results}, open(f"{OUT}/scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
