"""Per-object breakdown for CO3D DiffAE ID: separate 'good but blurry' from 'failed'.
For each object: ours PSNR, splatter PSNR, gap, and ours high-freq content ratio vs
splatter (image spectrum, foreground) = sharpness. Flag failures (gap large)."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, glob, json, numpy as np
from PIL import Image
RES=f"{_EXT}/splatter-image-rebuttal/rebuttal/results/freq_co3d/hyd_diffae_id"
def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255.
def psnr(a,b): return -10*np.log10(np.mean((a-b)**2)+1e-12)
def hf(img):
    g=img.mean(-1); g=g-g.mean(); P=np.abs(np.fft.fftshift(np.fft.fft2(g)))**2
    H,W=P.shape; y,x=np.indices((H,W)); rr=np.sqrt((x-W//2)**2+(y-H//2)**2)/ (0.5*np.hypot(H,W))
    return P[rr>0.5].mean()   # high-freq band energy
rows=[]
for oid in sorted(os.listdir(f"{RES}/ours")):
    ro=sorted(glob.glob(f"{RES}/ours/{oid}/*.png")); rs=sorted(glob.glob(f"{RES}/splatter/{oid}/*.png"))
    ts=sorted(glob.glob(f"{RES}/gt/{oid}/targets/*.png"))
    po=np.mean([psnr(load(a),load(b)) for a,b in list(zip(ro,ts))[::5]])
    ps=np.mean([psnr(load(a),load(b)) for a,b in list(zip(rs,ts))[::5]])
    ho=np.mean([hf(load(a)) for a in ro[::5]]); hs=np.mean([hf(load(a)) for a in rs[::5]])
    rows.append((oid,po,ps,ps-po,ho/max(hs,1e-9)))
rows.sort(key=lambda r:r[3])
print(f"{'object':<20}{'ours':>7}{'splat':>7}{'gap':>7}{'sharp(o/s)':>12}")
for oid,po,ps,gap,sh in rows:
    flag=" <-- FAIL" if gap>4 else (" blurry" if sh<0.85 else "")
    print(f"{oid:<20}{po:>7.2f}{ps:>7.2f}{gap:>7.2f}{sh:>12.2f}{flag}")
g=np.array([r[3] for r in rows]); sh=np.array([r[4] for r in rows])
print(f"\nn={len(rows)}  mean gap {g.mean():.2f}  median gap {np.median(g):.2f}")
print(f"failures (gap>4): {(g>4).sum()}/{len(rows)}   good-but-blurry(sharp<0.85 & gap<=4): {((sh<0.85)&(g<=4)).sum()}")
print(f"mean sharpness ratio ours/splat: {sh.mean():.2f} (median {np.median(sh):.2f})")
