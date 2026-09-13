#!/usr/bin/env python3
"""
Infer3D (DAE) for RE10K: BLIND fisheye self-calibration by generative-prior naturalness.

This is the method. It searches camera hypotheses (perspective FOV x radial-distortion
severity `a` along the nominal lens profile) and scores each by how well the frozen DiffAE
prior AUTOENCODES the resulting undistorted image at a LOW diffusion step count (T=4). A
tight prior bottleneck reconstructs only geometrically-natural (correctly-undistorted)
images well, so the score peaks at the true camera — analysis-by-synthesis, no calibration
labels, no ground truth. Selection: coarse grid -> fine grid (mean localization) ->
per-image paired VOTE among the top cells (removes common-mode image-difficulty variance
along the FOV<->severity degeneracy valley). Writes the selected calibration and applies it
to every fisheye source (the "ours" undistortions fed to CATSplat).

Run:
  PYTHONPATH=. python experiments/re10k_fisheye/select_calib.py \
      --fish_dir $OUT/fisheye_src --fxf 190.4 \
      --out_json $OUT/ae_selected_calib.json --apply_out $OUT/ours_src
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, cv2, torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fisheye import fov2f, f2fov, undistort_np, psnr
from diffae_prior import load_diffae, device

K_SPEC = np.array([0.3872, -0.71595, 0.4026, 0.0])   # nominal lens theta-poly profile (datasheet shape)
S = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fish_dir', required=True, help='dir of exported fisheye source PNGs')
    ap.add_argument('--fxf', type=float, required=True, help='nominal fisheye focal (px) = circle_scale*(S/2)/(rad(out_fov)/2)')
    ap.add_argument('--n_img', type=int, default=24, help='images scored per candidate')
    ap.add_argument('--T', type=int, default=4, help='autoencode steps; LOW T = tight prior bottleneck = discriminative naturalness (T>=10 reconstructs anything and the signal inverts)')
    ap.add_argument('--fov_grid', default='60,68,76,84,92,100,108,116,124')
    ap.add_argument('--a_grid', default='0.0,0.35,0.7,1.05,1.4')
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--apply_out', default=None)
    args = ap.parse_args()

    model = load_diffae()
    tf = T.Compose([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    files = [f for f in sorted(glob.glob(f'{args.fish_dir}/*.png')) if not f.endswith(('_clean.png', '_gen.png'))]
    batch = [cv2.resize(cv2.imread(f), (S, S)) for f in files[:args.n_img]]

    def ae(bgr):
        x = tf(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        with torch.no_grad():
            cond = model.encode(x)
            xT = model.encode_stochastic(x, cond, T=args.T)
            rec = model.render(xT, cond, T=args.T).clamp(0, 1)
        r = rec.cpu().numpy()[0].transpose(1, 2, 0)[:, :, ::-1] * 255
        return psnr(bgr.astype(np.float64), r, bgr.max(2) > 2)

    def per_image(fov, a):
        vals = []
        for fish in batch:
            u, v = undistort_np(fish, fov2f(fov, S), args.fxf, tuple(K_SPEC * a), S, S)
            if v.mean() < 0.3:
                return None
            vals.append(ae(u))
        return np.array(vals)

    def grid_search(cells, tag):
        rows = []
        for fv, a in cells:
            vi = per_image(fv, a)
            if vi is None:
                print(f'[{tag}] fov {fv:.1f} a {a:.2f}: invalid', flush=True); continue
            rows.append(((fv, a), vi))
            print(f'[{tag}] fov {fv:.1f} a {a:.2f}: AE {vi.mean():.3f}', flush=True)
        return rows

    fovs = [float(x) for x in args.fov_grid.split(',')]
    avals = [float(x) for x in args.a_grid.split(',')]
    coarse = grid_search([(fv, a) for fv in fovs for a in avals], 'coarse')
    (fv, a), _ = max(coarse, key=lambda r: r[1].mean())
    print(f'coarse peak: fov {fv} a {a}', flush=True)
    fine = grid_search([(f2, round(a2, 3)) for f2 in np.arange(fv - 6, fv + 6.1, 2)
                        for a2 in np.arange(max(0, a - 0.35), a + 0.36, 0.1)], 'fine')
    top = sorted(fine, key=lambda r: -r[1].mean())[:8]
    M = np.stack([r[1] for r in top])                       # [cells, imgs]
    wins = (M.argmax(0)[None, :] == np.arange(len(top))[:, None]).sum(1)
    for (c, _), w in zip(top, wins):
        print(f'[vote] fov {c[0]:.1f} a {c[1]:.2f}: wins {int(w)}/{M.shape[1]}', flush=True)
    (fv2, a2), vi = top[int(np.argmax(wins))]
    calib = {'fov': float(fv2), 'a': float(a2), 'fxp': fov2f(float(fv2), S), 'fxf': args.fxf,
             'k': [float(x) for x in (K_SPEC * a2)], 'ae_score': float(vi.mean()),
             'votes': int(wins.max()), 'n_vote_imgs': int(M.shape[1]), 'n_img': len(batch)}
    print(f'SELECTED (vote): fov {fv2:.1f} a {a2:.2f} (AE {calib["ae_score"]:.3f}, wins {calib["votes"]}/{calib["n_vote_imgs"]})', flush=True)
    json.dump(calib, open(args.out_json, 'w'), indent=2)

    if args.apply_out:
        os.makedirs(args.apply_out, exist_ok=True)
        for f in files:
            seq = os.path.splitext(os.path.basename(f))[0]
            u, _ = undistort_np(cv2.resize(cv2.imread(f), (S, S)), calib['fxp'], calib['fxf'], calib['k'], S, S)
            cv2.imwrite(f'{args.apply_out}/{seq}.png', u)
        print(f'applied to {len(files)} -> {args.apply_out}', flush=True)


if __name__ == '__main__':
    main()
