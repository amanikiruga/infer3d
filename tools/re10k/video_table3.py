#!/usr/bin/env python3
"""
Qualitative trajectory VIDEOS for Table 3. For each sequence, render the GT-pose trajectory
through frozen CATSplat for each source condition and write a side-by-side mp4:
    [ CATSplat (direct) | Equidistant | Infer3D (ours) | GT ]
Per-frame PSNR overlaid; the exact frames that entered the Table-3 metrics (src+5, src+10,
rand from the official split) get an 'EVAL' border.

  PYTHONPATH=. python tools/re10k/video_table3.py abs=diffae_abs \
      run.checkpoint=$CATSPLAT_CKPT dataset.data_path=$RE10K_CLIPS +dataset.crop_border=true \
      +eval3.ours_dir=$OUT/ours_src +eval3.out=$OUT/videos \
      "+video.seqs=[seqA,seqB]" +video.max_frames=40
"""
import os, sys, math
os.environ['WANDB_MODE'] = 'disabled'; os.environ['WANDB_DISABLED'] = 'true'; os.environ.pop('WANDB_API_KEY', None)
import numpy as np, torch, torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_table3 as E3          # reuse synthesis + source construction + globals
from infer3d import config as C
CATSPLAT_ROOT = os.getenv("CATSPLAT_ROOT", C.CATSPLAT_ROOT)
sys.path.insert(0, CATSPLAT_ROOT)
sys.path.insert(0, os.path.join(CATSPLAT_ROOT, 'experiments', 'diffae_catsplat_re10k'))

device = torch.device('cuda')
METHODS = [('fisheye', 'CATSplat (direct)'), ('equidistant', 'Equidistant'), ('ours', 'Infer3D (ours)')]


def annotate(img, text, sub='', eval_mark=False):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (255, 255, 255), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    if eval_mark:
        cv2.rectangle(img, (2, 2), (img.shape[1] - 3, img.shape[0] - 3), (30, 100, 220), 3)
        cv2.putText(img, 'EVAL', (img.shape[1] - 62, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 100, 220), 2, cv2.LINE_AA)
    return img


def psnr_t(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)


@hydra.main(version_base=None, config_path=os.path.join(CATSPLAT_ROOT, 'configs'), config_name="config_abs")
def main(cfg: DictConfig):
    from datasets.util import create_datasets
    from generate_ood_data_re10k import load_catsplat_model, prepare_catsplat_inputs

    e3 = cfg.get('eval3', {})
    E3.IN_FOV = float(e3.get('in_fov', E3.IN_FOV)); E3.OUT_FOV = float(e3.get('out_fov', E3.OUT_FOV))
    if 'k_scale' in e3:
        E3.K_TRUE = tuple(k * float(e3['k_scale']) for k in E3.K_BASE)
    if 'circle_scale' in e3:
        E3.CIRCLE_SCALE = float(e3['circle_scale'])
    out = str(e3.get('out', 'experiments_out/table3/videos')); ours_dir = str(e3.get('ours_dir'))
    vcfg = cfg.get('video', {})
    want = [str(s) for s in vcfg.get('seqs', [])]
    max_frames = int(vcfg.get('max_frames', 40))
    os.makedirs(out, exist_ok=True)

    cfg.data_loader.batch_size = 1
    catsplat = load_catsplat_model(cfg, cfg.run.checkpoint); catsplat.eval()
    test_dataset, _ = create_datasets(cfg, split="test")
    pad = cfg.dataset.pad_border_aug

    done = 0
    for seq_idx in range(len(test_dataset)):
        sk, idxs = test_dataset._seq_key_src_idx_pairs[seq_idx]
        if want and sk not in want:
            continue
        try:
            ts = test_dataset[seq_idx]
        except Exception:
            continue
        seq_key = ts[("frame_id", 0)].split("+")[1]
        if not os.path.exists(os.path.join(ours_dir, f'{seq_key}.png')):
            continue
        seq_data = test_dataset._seq_data[seq_key]; total = len(seq_data["timestamps"])
        src_idx = min(total - 1, idxs[0]); tgts = [min(total - 1, j) for j in idxs[1:]]
        lo, hi = min([src_idx] + tgts), max([src_idx] + tgts)
        step = max(1, (hi - lo) // max_frames + (1 if (hi - lo) > max_frames else 0))
        traj = sorted(set(list(range(lo, hi + 1, step)) + tgts))
        sfd = test_dataset.get_frame_data(seq_key=seq_key, frame_idx=src_idx, pose_data=seq_data, color_aug_fn=lambda x: x)
        K_tgt, K_src, inv_K_src, color, _, T_c2w, _, _ = sfd
        clean_bgr = E3.np_from_color(color)

        srcs = {}
        for m, _ in METHODS:
            src_bgr = E3.make_source(clean_bgr, m, ours_dir=ours_dir, seq_key=seq_key)
            scol = E3.color_from_np(cv2.resize(src_bgr, (color.shape[-1], color.shape[-2])))
            scol_aug = scol.clone()
            if pad:
                import torchvision.transforms as T
                scol_aug = T.Pad((pad, pad))(scol_aug)
            si = {("color", 0, 0): scol, ("color_aug", 0, 0): scol_aug, ("K_tgt", 0): K_tgt,
                  ("K_src", 0): K_src, ("inv_K_src", 0): inv_K_src, ("T_c2w", 0): T_c2w,
                  ("T_w2c", 0): torch.linalg.inv(T_c2w),
                  ("llava_feat", 0): ts.get(("llava_feat", 0), torch.zeros([39, 5120]))}
            for key in [("depth_sparse", 0), ("unidepth", 0, 0), ("scale_colmap", 0)]:
                if key in ts:
                    si[key] = ts[key]
            srcs[m] = si

        H = 216; frames_out = []
        for vi in traj:
            try:
                tfd = test_dataset.get_frame_data(seq_key=seq_key, frame_idx=int(vi), pose_data=seq_data, color_aug_fn=lambda x: x)
            except Exception:
                continue
            K_tgt_t, _, _, color_t, _, T_c2w_t, _, _ = tfd
            gt = color_t.unsqueeze(0).to(device)
            panels = []
            for m, mname in METHODS:
                ti = {("color", 1, 0): color_t, ("K_tgt", 1): K_tgt_t, ("T_c2w", 1): T_c2w_t, ("T_w2c", 1): torch.linalg.inv(T_c2w_t)}
                ci = prepare_catsplat_inputs(srcs[m], ti)
                with torch.no_grad():
                    rend = catsplat(ci)[("color_gauss", 1, 0)]
                if rend.shape[-2:] != gt.shape[-2:]:
                    rend = F.interpolate(rend, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                p = psnr_t(rend.clamp(0, 1), gt)
                img = E3.np_from_color(rend.clamp(0, 1)[0])
                img = cv2.resize(img, (int(img.shape[1] * H / img.shape[0]), H))
                panels.append(annotate(img, mname, f'{p:.1f} dB  (frame {vi})', eval_mark=(vi in tgts)))
            g = cv2.resize(E3.np_from_color(color_t), (int(color_t.shape[-1] * H / color_t.shape[-2]), H))
            panels.append(annotate(g, 'GT', f'frame {vi}', eval_mark=(vi in tgts)))
            frames_out.append(np.concatenate(panels, axis=1))
        if not frames_out:
            continue
        h, w = frames_out[0].shape[:2]
        vw = cv2.VideoWriter(f'{out}/{seq_key}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 6, (w, h))
        for fr in frames_out:
            vw.write(fr)
        vw.release()
        done += 1
        print(f'{seq_key}: {len(frames_out)} frames -> {out}/{seq_key}.mp4', flush=True)
        if want and done >= len(want):
            break
    print('DONE', done, 'videos')


if __name__ == '__main__':
    main()
