"""Compare regenerate_ours_splats variants on 2 CO3D DiffAE-ID objects vs cached truth.
paper=e534f0e, head=current worktree, clean=infer3d-clean. Cached: 250->17.16, 286->22.77."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import sys, glob, numpy as np, torch
WT = f"{_EXT}/splatter-image-rebuttal"
ORIG = f"{_EXT}/splatter-image"
sys.path.insert(0, WT); sys.path.insert(0, f"{WT}/experiments/stylegan3-ours-co3d")
sys.path.insert(0, f"{WT}/rebuttal/frequency")
import eval_renderings_with_icp_batch as E
from hydra import initialize_config_dir, compose
import hydra
from utils.abs_utils import render_with_custom_camera, render_with_custom_camera_align
import render_id_pair_co3d as R   # reuse icp_align, render_views, regenerate_ours_splats_paper

def make_ours(checkpoint, gp, cfg, device, variant):
    bi = checkpoint["best_input_image"]; rot = checkpoint["best_rotation_matrix"].to(device)
    tr = checkpoint["best_translation_matrix"].to(device); zgt = checkpoint["zgt"]
    ood = {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in checkpoint["ood_data"].items()}
    oii = checkpoint["ood_image_index"]; orr = checkpoint["ood_random_rotation"].to(device)
    bg = torch.tensor([1,1,1] if cfg.data.white_background else [0,0,0], dtype=torch.float32, device=device)
    od = ood["origin_distances"][0,oii].to(device); foc = ood["focals_pixels"][0,oii].to(device)
    img = bi.squeeze(0) if bi.dim()==4 else bi
    if img.shape[0]>3: img = img[:3]
    inp = torch.cat([img.unsqueeze(0).unsqueeze(1).to(device), od.unsqueeze(0).unsqueeze(1)], dim=2)
    fp = foc.unsqueeze(0).unsqueeze(0)
    sl = oii if variant=="paper" else 0                       # locus 2
    with torch.no_grad():
        ps = gp(inp, ood["view_to_world_transforms"][:1, sl:sl+cfg.data.input_images],
                ood["source_cv2wT_quat"][:1, sl:sl+cfg.data.input_images], fp)
    ps = {k:v[0] for k,v in ps.items()}
    if variant=="paper":                                       # locus 1
        t = render_with_custom_camera(ps, bg, cfg, foc, rot, zgt, device=device, return_splats=True, translation=tr)
    else:
        t = render_with_custom_camera_align(ps, bg, cfg, foc, rot, zgt, device=device, return_splats=True, translation=tr, zgt_ood=zgt)
    inv = orr.T                                                # locus 3
    if variant=="head":
        t = render_with_custom_camera(t, bg, cfg, foc, inv, zgt, device=device, return_splats=True, translation=None, zgt_ood=zgt)
    else:  # paper & clean: no zgt_ood
        t = render_with_custom_camera(t, bg, cfg, foc, inv, zgt, device=device, return_splats=True, translation=None)
    return t, ood

device="cuda"
hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize_config_dir(version_base=None, config_dir=f"{WT}/configs")
cfg = compose(config_name="abs_config", overrides=["general.split=0","general.total_splits=1","+dataset=hydrants",
    "general.prefix=cmp","abs=stylegan_abs",f"opt.pretrained_ckpt={R.LIFTERS['hydrants']}","general.data_example_ids_path=nn.json"])
gp = E.load_gaussian_predictor(cfg, device)
D = f"{ORIG}/checkpoints-diffae-co3d-hydrants-baseline-se3-new-stylegan-10-depth-encoder-indist"
ED = f"{ORIG}/eval_output/checkpoints-diffae-co3d-hydrants-baseline-se3-new-stylegan-10-depth-encoder-indist"
def plymap(n):
    d={}
    for ln in open(f"{ED}/lists/{n}.txt"):
        ln=ln.strip()
        if ln: d[__import__('os').path.basename(ln).replace('_pointcloud.ply','')]=ln
    return d
om, gm = plymap("ours"), plymap("pseudo_gt")
cached={"250_26808_55348":17.16,"286_30164_59371":22.77}
val=None
for oid in ["250_26808_55348","286_30164_59371"]:
    ck = glob.glob(f"{D}/{oid}*/topk_best_everything_latest.pth")[0]
    c = E.load_checkpoint(ck, device)
    if "ood_data" not in c:
        if val is None: val = E.CO3DDataset(cfg,"test",use_hq=False)
        i,_ = val.find_index_for_sequence_prefix(c["example_id"])
        od = val[i]; c["ood_data"]={k:(v.unsqueeze(0) if isinstance(v,torch.Tensor) else v) for k,v in od.items()}
    sol,off = R.icp_align(om[oid], gm[oid], device)
    oii = c["ood_image_index"]
    line=f"{oid} (cached {cached[oid]}): "
    for var in ["paper","head","clean"]:
        sp,ood = make_ours(c, gp, cfg, device, var)
        sp = E.apply_icp_to_splats(sp, sol, off, device)
        pairs = R.render_views(sp, ood, cfg, device, oii)
        psnr = float(np.mean([-10*np.log10(np.mean((p-g)**2)+1e-12) for p,g in pairs]))
        line += f"{var}={psnr:.2f}  "
    print(line)
