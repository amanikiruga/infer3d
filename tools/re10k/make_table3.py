#!/usr/bin/env python3
"""Assemble Table 3 (RE10K sensor/FOV shift) from per-sequence eval metrics.

Every row is the mean over ALL evaluated sequences. No subset selection: nothing is
filtered or chosen to match the published value, which is printed only for reference.
Infer3D is additionally compared to each baseline as a *paired* per-sequence delta.

Input: table3_per_seq.json from eval_table3.py.
"""
import argparse
import json
import math
import statistics as st

PAPER = {  # method -> (psnr, ssim, lpips) from the Infer3D paper Table 3
    'clean':       (25.41, 0.840, 0.149),
    'fisheye':     (16.25, 0.610, 0.327),
    'equidistant': (19.24, 0.688, 0.253),
    'ours':        (23.33, 0.775, 0.198),
}
ROWLBL = {
    'clean':       ('In-dist.', 'CATSplat'),
    'fisheye':     ('OOD Fisheye', 'CATSplat direct'),
    'equidistant': ('OOD Fisheye', 'Equidistant (given TRUE FOV)'),
    'ours':        ('OOD Fisheye', 'Infer3D (DAE), blind'),
    'oracle':      ('OOD Fisheye', 'oracle undistort (ceiling)'),
}
ORDER = ['clean', 'fisheye', 'equidistant', 'ours', 'oracle']


def ci95(vals):
    return 1.96 * st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('per_seq_json')
    args = ap.parse_args()

    ps = json.load(open(args.per_seq_json))
    seqs = sorted(ps)
    present = list(next(iter(ps.values())))
    methods = [m for m in ORDER if m in present] + [m for m in present if m not in ORDER]
    print(f'{len(seqs)} sequences (full set, no selection)\n')

    print('| Setting | Method | PSNR | SSIM | LPIPS | paper (P/S/L) |')
    print('|---|---|---|---|---|---|')
    for m in methods:
        cells = []
        for k, fmt in (('psnr', 2), ('ssim', 3), ('lpips', 3)):
            v = [ps[s][m][k] for s in seqs]
            cells.append(f'{st.mean(v):.{fmt}f} ±{ci95(v):.{fmt}f}')
        p = PAPER.get(m)
        ref = f'{p[0]} / {p[1]} / {p[2]}' if p else '—'
        setting, method = ROWLBL.get(m, ('', m))
        print(f'| {setting} | {method} | ' + ' | '.join(cells) + f' | {ref} |')

    if 'ours' not in methods:
        return
    print('\nPaired per-sequence PSNR, Infer3D minus baseline:')
    for b in methods:
        if b == 'ours':
            continue
        d = [ps[s]['ours']['psnr'] - ps[s][b]['psnr'] for s in seqs]
        m, h = st.mean(d), ci95(d)
        wins = sum(1 for x in d if x > 0) / len(d)
        print(f'  vs {b:12s} {m:+7.3f} dB  95% CI [{m - h:+.2f}, {m + h:+.2f}]  '
              f'win {wins:4.0%}  n={len(d)}')


if __name__ == '__main__':
    main()
