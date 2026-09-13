"""Disambiguate 'coarse identity' vs 'misregistration/shading' for the ID gap.
For each (render,gt) pair, report PSNR: raw, best over small 2D shifts (+/-3px),
and best over shift + per-channel affine gain/bias (removes exposure/shading).
If ours' gap closes much more than Splatter's under these nuisance transforms,
the low-freq excess is registration/shading, not genuine appearance/identity."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, glob, numpy as np
from PIL import Image
from scipy.ndimage import shift as ndshift
RES=f"{_EXT}/splatter-image-rebuttal/rebuttal/results"
def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255.
def psnr(a,b,m=None):
    e=(a-b)**2
    if m is not None: e=e[m]
    return -10*np.log10(e.mean()+1e-12)
def fg(gt): return gt.min(-1)<0.98
def best_shift(r,g,rng=3):
    best=psnr(r,g); bs=(0,0)
    for dy in range(-rng,rng+1):
        for dx in range(-rng,rng+1):
            rs=np.stack([ndshift(r[...,c],(dy,dx),order=1,mode='nearest') for c in range(3)],-1)
            p=psnr(rs,g)
            if p>best: best,bs=p,(dy,dx)
    return best,bs
def best_shift_affine(r,g,rng=3):
    best=-1
    for dy in range(-rng,rng+1):
        for dx in range(-rng,rng+1):
            rs=np.stack([ndshift(r[...,c],(dy,dx),order=1,mode='nearest') for c in range(3)],-1)
            # per-channel least-squares gain+bias to g
            rr=rs.copy()
            for c in range(3):
                x=rs[...,c].ravel(); y=g[...,c].ravel()
                A=np.vstack([x,np.ones_like(x)]).T
                gain,bias=np.linalg.lstsq(A,y,rcond=None)[0]
                rr[...,c]=np.clip(gain*rs[...,c]+bias,0,1)
            p=psnr(rr,g)
            if p>best: best=p
    return best
def run(method):
    md=f"{RES}/task_a_runs/freq_{method}_control"
    raw=[]; sh=[]; sa=[]
    for oid in sorted(os.listdir(md)):
        rs=sorted(glob.glob(f"{md}/{oid}/*.png")); ts=sorted(glob.glob(f"{RES}/task_a_inputs_control/{oid}/targets/*.png"))
        for rp,tp in list(zip(rs,ts))[::3]:   # every 3rd view for speed
            r,g=load(rp),load(tp)
            raw.append(psnr(r,g))
            bs,_=best_shift(r,g); sh.append(bs)
            sa.append(best_shift_affine(r,g))
    return np.mean(raw),np.mean(sh),np.mean(sa)
for m in ["ours","splatter"]:
    r,s,a=run(m)
    print(f"{m:9s}: raw {r:.2f} | +shift {s:.2f} (+{s-r:.2f}) | +shift+affine {a:.2f} (+{a-r:.2f})")
