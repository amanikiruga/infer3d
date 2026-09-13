"""STEP 0 de-risk: does model_objaverse.pth work as a general-purpose lifter Phi
under its CORRECT config (fov 49.13, znear/zfar 0.8/3.2, radius 2.0, 3-ch RGB,
focals=None, white bg)? Pure feed-forward: reconstruct from view-0, render the other
GT views of held-out objaverse (LVIS) objects, report PSNR/SSIM/LPIPS + montages.

If Phi reconstructs recognizable novel views here, the broad-prior Infer3D idea is viable.
Run on the H200 debug node:
  srun --jobid=35454875 --overlap bash -c 'cd .../broad_prior && CUDA_VISIBLE_DEVICES=0 \
    mamba run -n test2 python eval_objaverse_lifter.py --n 20'
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import argparse, json, os, sys

ORIG = f"{_EXT}/splatter-image"
os.environ.setdefault("OBJAVERSE_ROOT", f"{ORIG.rsplit('/',1)[0]}/datasets/objaverse/views_release")
os.environ.setdefault("OBJAVERSE_LVIS_ANNOTATION_PATH",
                      f"{_EXT}/diffae/lvis-annotations.json")
sys.path.insert(0, ORIG)

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted
from splatter_image_datasets.objaverse import ObjaverseDataset
import lpips as lpips_lib
from utils.loss_utils import ssim as ssim_fn

dev = "cuda"
CKPT = f"{ORIG}/checkpoints/model_objaverse.pth"
OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/lifter_eval"


def build_cfg():
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    initialize_config_dir(version_base=None, config_dir=f"{ORIG}/configs")
    cfg = compose(config_name="abs_config", overrides=["+dataset=objaverse",
                                                       f"opt.pretrained_ckpt={CKPT}"])
    OmegaConf.set_struct(cfg, False)
    # defensive defaults (only used by dataset/train paths)
    if "input_images" not in cfg.data: cfg.data.input_images = 1
    if "subset" not in cfg.data: cfg.data.subset = -1
    if "imgs_per_obj" not in cfg.opt: cfg.opt.imgs_per_obj = 12
    return cfg


def psnr(a, b):
    return float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = build_cfg()
    print("cfg.data:", OmegaConf.to_container(cfg.data))

    gp = GaussianSplatPredictor(cfg).to(memory_format=torch.channels_last).to(dev).eval()
    sd = torch.load(CKPT, map_location=dev, weights_only=False)
    gp.load_state_dict(sd["model_state_dict"])
    print("loaded", CKPT)

    ds = ObjaverseDataset(cfg, "test")
    lp = lpips_lib.LPIPS(net="vgg").to(dev).eval()
    bg = torch.tensor([1., 1, 1], device=dev)

    rows = []
    n = min(args.num, len(ds))
    for i in range(n):
        try:
            item = ds[i]
            V = item["gt_images"].shape[0]
            src = 0
            x = item["gt_images"][src].unsqueeze(0).unsqueeze(0).to(dev)          # [1,1,3,H,W]
            v2w = item["view_to_world_transforms"][src].unsqueeze(0).unsqueeze(0).to(dev)
            quat = item["source_cv2wT_quat"][src].unsqueeze(0).unsqueeze(0).to(dev)
            with torch.no_grad():
                sp = gp(x, v2w, quat, None)
            sp = {k: v[0] for k, v in sp.items()}

            novel_ps, novel_ss, novel_lp = [], [], []
            renders = []
            with torch.no_grad():
                for v in range(V):
                    wvt = item["world_view_transforms"][v].unsqueeze(0).to(dev)
                    fpt = item["full_proj_transforms"][v].unsqueeze(0).to(dev)
                    cc = item["camera_centers"][v].unsqueeze(0).to(dev)
                    r = render_predicted(sp, wvt, fpt, cc, bg, cfg, focals_pixels=None)["render"].clamp(0, 1)
                    gt = item["gt_images"][v].to(dev).clamp(0, 1)
                    renders.append((gt.cpu(), r.cpu()))
                    if v != src:
                        novel_ps.append(psnr(r, gt))
                        novel_ss.append(ssim_fn(r, gt).item())
                        novel_lp.append(lp(r.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).item())
            row = {"object": item.get("object_id", str(i)),
                   "psnr_novel": float(np.mean(novel_ps)), "ssim_novel": float(np.mean(novel_ss)),
                   "lpips_novel": float(np.mean(novel_lp)), "n_novel": len(novel_ps)}
            rows.append(row)
            print(f"[{i}] {row['object']}: novel PSNR {row['psnr_novel']:.2f} "
                  f"SSIM {row['ssim_novel']:.3f} LPIPS {row['lpips_novel']:.3f}")

            # montage: src input | 3 GT novel | 3 rendered novel
            picks = [src] + [v for v in range(V) if v != src][:3]
            def strip(kind):
                out = []
                for v in picks:
                    gt, r = renders[v]
                    img = (item["gt_images"][src] if (kind == "in" and v == src) else (r if kind == "pred" else gt))
                    out.append((np.clip(img.numpy().transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8))
                return np.concatenate(out, axis=1)
            grid = np.concatenate([strip("gt"), strip("pred")], axis=0)
            Image.fromarray(grid).save(f"{OUT}/montage_{i:02d}_{row['object']}.png")
        except Exception as ex:
            import traceback; traceback.print_exc(); print(f"skip {i}: {ex}")

    if rows:
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ["psnr_novel", "ssim_novel", "lpips_novel"]}
        agg["n_objects"] = len(rows)
        print("\n==== model_objaverse feed-forward, held-out objaverse novel views ====")
        print(json.dumps(agg, indent=1))
        json.dump({"agg": agg, "rows": rows,
                   "config": "objaverse fov49.13 zfar3.2 radius2.0 3ch focals=None"},
                  open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
