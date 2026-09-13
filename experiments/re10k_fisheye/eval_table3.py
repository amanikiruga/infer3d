#!/usr/bin/env python3
"""
Table 3 (RE10K OOD-fisheye) evaluation harness.

For each test sequence: take the clean source frame, synthesize an OOD fisheye, form each
SOURCE variant, feed it to frozen CATSplat, render the official held-out target poses, and
score PSNR/SSIM/LPIPS against the clean GT targets. Conditions:
  clean        in-distribution upper bound (CATSplat on clean source)
  fisheye      CATSplat directly on the fisheye (OOD failure baseline)
  equidistant  naive equidistant undistortion (given true FOV+focal, k=0) -> CATSplat
  naive_persp  naive wrong-FOV perspective undistortion -> CATSplat
  pinhole90    assume a plain 90-deg pinhole -> CATSplat
  oracle       undistort with TRUE calibration -> CATSplat (undistortion ceiling)
  ours         blind DiffAE-prior undistortion (precomputed by select_calib.py) -> CATSplat

Protocol = official CATSplat RE10K: eval indices (src, src+5, src+10, rand) read from the
split file; pair with dataset.crop_border=true. External deps (env-overridable via
infer3d.config): CATSPLAT_ROOT (+ its checkpoint), and for 'ours' the sources produced by
select_calib.py.

Run (see scripts/reproduce_table3.sh):
  PYTHONPATH=. python experiments/re10k_fisheye/eval_table3.py abs=diffae_abs \
      run.checkpoint=$CATSPLAT_CKPT dataset.data_path=$RE10K_CLIPS +dataset.crop_border=true \
      "+eval3.methods=[clean,fisheye,equidistant,oracle,ours]" +eval3.n_seqs=160 \
      +eval3.ours_dir=$OUT/ours_src +eval3.out=$OUT/eval_final
"""
import os, sys, math, json
os.environ['WANDB_MODE'] = 'disabled'; os.environ['WANDB_DISABLED'] = 'true'; os.environ.pop('WANDB_API_KEY', None)
# NOTE: do NOT set HF_HUB_OFFLINE here — CATSplat resolves the cached UniDepth via HF_HOME.
import numpy as np, torch, torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisheye import fov2f, undistort_np, crop_valid_square, perspective_to_fisheye

from infer3d import config as C
CATSPLAT_ROOT = os.getenv("CATSPLAT_ROOT", C.CATSPLAT_ROOT)
sys.path.insert(0, CATSPLAT_ROOT)

device = torch.device('cuda')

# OOD fisheye operating point (defaults = the locked, paper-matched setting; overridable).
_F = C.RE10K_FISHEYE
IN_FOV, OUT_FOV = _F['in_fov'], _F['out_fov']
K_BASE = (0.3872, -0.71595, 0.4026, 0.0)      # nominal profile; scaled by k_scale below
K_TRUE = tuple(k * _F['k_scale'] for k in K_BASE)
CIRCLE_SCALE = _F['circle_scale']
NAIVE_FOV = 90.0


def fxf_out(S):
    """Fisheye focal used by synthesis AND every undistortion baseline (must match)."""
    return (CIRCLE_SCALE * S / 2.0) / (math.radians(OUT_FOV) / 2)


def np_from_color(color):   # CHW float[0,1] RGB tensor -> HWC uint8 BGR
    a = (color.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def color_from_np(bgr):     # HWC uint8 BGR -> CHW float[0,1] RGB tensor
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def make_fisheye(clean_bgr, S=256):
    fish, lens = perspective_to_fisheye(clean_bgr, (S, S), OUT_FOV, IN_FOV, K_TRUE, circle_scale=CIRCLE_SCALE)
    return fish * (lens > 0)[..., None].astype(np.uint8)


def make_source(clean_bgr, method, S=256, ours_dir=None, seq_key=None):
    """SOURCE image (HWC uint8 BGR) for a condition."""
    if method == 'clean':
        return clean_bgr
    fish = make_fisheye(clean_bgr, S)
    fxf = fxf_out(S)
    if method == 'fisheye':
        return fish
    if method == 'equidistant':   # given true FOV + focal, ignore radial distortion (k=0)
        return undistort_np(fish, fov2f(IN_FOV, S), fxf, (0, 0, 0, 0), S, S)[0]
    if method == 'naive_persp':   # wrong narrow FOV, no distortion
        return undistort_np(fish, fov2f(NAIVE_FOV, S), (S / 2.) / (math.radians(160) / 2), (0, 0, 0, 0), S, S)[0]
    if method == 'pinhole90':     # assume a plain 90-deg pinhole
        return undistort_np(fish, fov2f(90.0, S), fxf, (0, 0, 0, 0), S, S)[0]
    if method == 'oracle':        # undistort with TRUE calibration
        return undistort_np(fish, fov2f(IN_FOV, S), fxf, K_TRUE, S, S)[0]
    if method == 'ours':          # blind DiffAE-prior undistortion (precomputed)
        p = os.path.join(ours_dir, f'{seq_key}.png')
        if not os.path.exists(p):
            raise FileNotFoundError(f'ours source missing: {p} (run select_calib.py)')
        return cv2.imread(p)
    raise ValueError(method)


@hydra.main(version_base=None, config_path=os.path.join(CATSPLAT_ROOT, 'configs'), config_name="config_abs")
def main(cfg: DictConfig):
    from datasets.util import create_datasets
    from evaluation.evaluator import Evaluator
    from experiments.re10k_fisheye.catsplat_io import load_catsplat_model, prepare_catsplat_inputs

    global IN_FOV, OUT_FOV, K_TRUE, CIRCLE_SCALE
    e3 = cfg.get('eval3', {})
    IN_FOV = float(e3.get('in_fov', IN_FOV)); OUT_FOV = float(e3.get('out_fov', OUT_FOV))
    if 'k_scale' in e3:
        K_TRUE = tuple(k * float(e3['k_scale']) for k in K_BASE)
    if 'circle_scale' in e3:
        CIRCLE_SCALE = float(e3['circle_scale'])
    crop_valid = bool(e3.get('crop_valid', False))
    print(f'[eval3] fisheye: in_fov={IN_FOV} out_fov={OUT_FOV} k={K_TRUE} circle_scale={CIRCLE_SCALE} crop_valid={crop_valid}', flush=True)

    mv = e3.get('methods', 'clean,fisheye,equidistant,oracle,ours')
    methods = [str(x) for x in mv] if not isinstance(mv, str) else mv.split(',')
    n_seqs = int(e3.get('n_seqs', 160))
    out = str(e3.get('out', 'experiments_out/table3'))
    ours_dir = str(e3.get('ours_dir', f'{out}/ours_src'))
    export_fisheye = bool(e3.get('export_fisheye', False))
    save_renders = bool(e3.get('save_renders', False))
    fish_dir = f'{out}/fisheye_src'; rend_dir = f'{out}/renders'
    os.makedirs(out, exist_ok=True); os.makedirs(fish_dir, exist_ok=True)
    if save_renders:
        os.makedirs(rend_dir, exist_ok=True)

    cfg.data_loader.batch_size = 1
    catsplat = load_catsplat_model(cfg, cfg.run.checkpoint); catsplat.eval()
    ev = Evaluator(crop_border=cfg.dataset.get('crop_border', 0)).to(device)
    test_dataset, _ = create_datasets(cfg, split="test")

    agg = {m: {'psnr': [], 'ssim': [], 'lpips': []} for m in methods}
    per_seq = {}
    done = 0
    for seq_idx in range(len(test_dataset)):
        if done >= n_seqs:
            break
        try:
            ts = test_dataset[seq_idx]
        except Exception:
            continue
        seq_key = ts[("frame_id", 0)].split("+")[1]
        if 'ours' in methods and not os.path.exists(os.path.join(ours_dir, f'{seq_key}.png')):
            continue   # keep all rows on the same seqs
        seq_data = test_dataset._seq_data[seq_key]; total = len(seq_data["timestamps"])
        # official CATSplat RE10K protocol: exact eval indices (src, src+5, src+10, rand)
        sk, idxs = test_dataset._seq_key_src_idx_pairs[seq_idx]
        assert sk == seq_key, f'{sk} != {seq_key}'
        src_idx = min(total - 1, idxs[0])
        view_indices = [src_idx] + [min(total - 1, j) for j in idxs[1:]]
        try:
            sfd = test_dataset.get_frame_data(seq_key=seq_key, frame_idx=src_idx, pose_data=seq_data, color_aug_fn=lambda x: x)
        except Exception:
            continue
        K_tgt, K_src, inv_K_src, color, color_aug, T_c2w, orig_size, depth = sfd
        clean_bgr = np_from_color(color)
        targets, ok = [], True
        for vi in view_indices[1:]:
            try:
                targets.append(test_dataset.get_frame_data(seq_key=seq_key, frame_idx=int(vi), pose_data=seq_data, color_aug_fn=lambda x: x))
            except Exception:
                ok = False; break
        if not ok or not targets:
            continue
        if export_fisheye:
            cv2.imwrite(f'{fish_dir}/{seq_key}.png', make_fisheye(clean_bgr))
            cv2.imwrite(f'{fish_dir}/{seq_key}_clean.png', clean_bgr)
        pad = cfg.dataset.pad_border_aug
        per_seq[seq_key] = {}
        for m in methods:
            src_bgr = make_source(clean_bgr, m, ours_dir=ours_dir, seq_key=seq_key)
            K_src_m, inv_K_src_m = K_src, inv_K_src
            if crop_valid and m not in ('clean', 'fisheye'):
                src_bgr, zoom = crop_valid_square(src_bgr)
                if zoom != 1.0:
                    K_src_m = K_src.clone(); K_src_m[0, 0] *= zoom; K_src_m[1, 1] *= zoom
                    inv_K_src_m = torch.linalg.pinv(K_src_m)
            scol = color_from_np(cv2.resize(src_bgr, (color.shape[-1], color.shape[-2])))
            scol_aug = scol.clone()
            if pad:
                import torchvision.transforms as T
                scol_aug = T.Pad((pad, pad))(scol_aug)
            source_inputs = {("color", 0, 0): scol, ("color_aug", 0, 0): scol_aug, ("K_tgt", 0): K_tgt,
                             ("K_src", 0): K_src_m, ("inv_K_src", 0): inv_K_src_m, ("T_c2w", 0): T_c2w,
                             ("T_w2c", 0): torch.linalg.inv(T_c2w),
                             ("llava_feat", 0): ts.get(("llava_feat", 0), torch.zeros([39, 5120]))}
            for key in [("depth_sparse", 0), ("unidepth", 0, 0), ("scale_colmap", 0)]:
                if key in ts:
                    source_inputs[key] = ts[key]
            mseq = {'psnr': [], 'ssim': [], 'lpips': []}
            for ti_idx, tfd in enumerate(targets):
                K_tgt_t, _, _, color_t, _, T_c2w_t, _, _ = tfd
                ti = {("color", 1, 0): color_t, ("K_tgt", 1): K_tgt_t, ("T_c2w", 1): T_c2w_t, ("T_w2c", 1): torch.linalg.inv(T_c2w_t)}
                ci = prepare_catsplat_inputs(source_inputs, ti)
                with torch.no_grad():
                    rend = catsplat(ci)[("color_gauss", 1, 0)]
                gt = color_t.unsqueeze(0).to(device)
                if rend.shape[-2:] != gt.shape[-2:]:
                    rend = F.interpolate(rend, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                with torch.no_grad():
                    sc = ev(rend.clamp(0, 1), gt)
                if save_renders and ti_idx == 0:
                    cv2.imwrite(f'{rend_dir}/{seq_key}_{m}.png', np_from_color(rend.clamp(0, 1)[0]))
                    cv2.imwrite(f'{rend_dir}/{seq_key}_GT.png', np_from_color(gt[0]))
                    cv2.imwrite(f'{rend_dir}/{seq_key}_src_{m}.png', src_bgr)
                for k in mseq:
                    mseq[k].append(float(sc[k])); agg[m][k].append(float(sc[k]))
            per_seq[seq_key][m] = {k: float(np.mean(v)) for k, v in mseq.items()}
        done += 1
        print(f"[{done}/{n_seqs}] {seq_key}: " + ' | '.join(f"{m} {np.mean(agg[m]['psnr']):.2f}" for m in methods), flush=True)

    summ = {m: {k: float(np.mean(v)) for k, v in d.items()} | {'n_views': len(d['psnr'])} for m, d in agg.items()}
    summ['_n_seqs'] = done
    json.dump(summ, open(f'{out}/table3_metrics.json', 'w'), indent=2)
    json.dump(per_seq, open(f'{out}/table3_per_seq.json', 'w'), indent=2)
    print("\n=== TABLE 3 (RE10K OOD fisheye) ==="); print(json.dumps(summ, indent=2))


if __name__ == '__main__':
    main()
