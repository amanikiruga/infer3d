"""Is ours actually blurry? Two direct checks I should have run:
 1. IMAGE power spectrum (not error): P_ours(f), P_splatter(f), P_gt(f) radially
    averaged. Blur = high-frequency content attenuated -> ours rolls off below gt/splatter.
 2. Visual panel: gt | splatter | ours | amplified error maps, object crops.
Restricted to foreground so the flat background doesn't dominate."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, glob, numpy as np
from PIL import Image
RES=f"{_EXT}/splatter-image-rebuttal/rebuttal"
def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255.

def radial_power(img):
    g=img.mean(-1)
    g=g-g.mean()
    F=np.fft.fftshift(np.fft.fft2(g)); P=np.abs(F)**2
    H,W=P.shape; y,x=np.indices((H,W)); rr=np.sqrt((x-W//2)**2+(y-H//2)**2)
    nb=64; b=(rr/rr.max()*(nb-1)).astype(int)
    prof=np.bincount(b.ravel(),P.ravel(),minlength=nb)/np.maximum(np.bincount(b.ravel(),minlength=nb),1)
    return prof
def freqs(nb=64): return np.linspace(0,1,nb)*0.5

def collect(ours_dir, splat_dir, gt_dir, is_co3d):
    po=pg=ps=None; n=0
    oids=sorted(os.listdir(ours_dir))
    for oid in oids:
        rs=sorted(glob.glob(f"{ours_dir}/{oid}/*.png"))
        ss=sorted(glob.glob(f"{splat_dir}/{oid}/*.png"))
        if is_co3d: ts=sorted(glob.glob(f"{gt_dir}/{oid}/targets/*.png"))
        else: ts=sorted(glob.glob(f"{gt_dir}/{oid}/targets/*.png"))
        for rp,sp,tp in list(zip(rs,ss,ts))[::4]:
            r,s,g=load(rp),load(sp),load(tp)
            ao=radial_power(r); asp=radial_power(s); ag=radial_power(g)
            po=ao if po is None else po+ao; ps=asp if ps is None else ps+asp; pg=ag if pg is None else pg+ag; n+=1
    return po/n, ps/n, pg/n, n

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

for tag,ours,splat,gt,is_co3d in [
    ("cars_SG", f"{RES}/results/task_a_runs/freq_ours_control", f"{RES}/results/task_a_runs/freq_splatter_control", f"{RES}/results/task_a_inputs_control", False),
    ("co3d_DiffAE", f"{RES}/results/freq_co3d/hyd_diffae_id/ours", f"{RES}/results/freq_co3d/hyd_diffae_id/splatter", f"{RES}/results/freq_co3d/hyd_diffae_id/gt", True),
]:
    po,ps,pg,n=collect(ours,splat,gt,is_co3d)
    f=freqs()
    hi=f>0.25
    print(f"[{tag}] n={n}  high-freq(>0.25) mean image power:  gt={pg[hi].mean():.3g}  splatter={ps[hi].mean():.3g}  ours={po[hi].mean():.3g}")
    print(f"        ratio to GT at high freq:  splatter/gt={ps[hi].mean()/pg[hi].mean():.2f}  ours/gt={po[hi].mean()/pg[hi].mean():.2f}   (<1 = blurrier than GT)")
    print(f"        ours/splatter at high freq = {po[hi].mean()/ps[hi].mean():.2f}   (<1 = ours blurrier than splatter)")
    plt.figure(figsize=(6,4))
    plt.semilogy(f,pg,label="GT",color='k',lw=2)
    plt.semilogy(f,ps,label="Splatter",color='steelblue',lw=2)
    plt.semilogy(f,po,label="Infer3D (ours)",color='seagreen',lw=2)
    plt.xlabel("spatial frequency (cyc/px)"); plt.ylabel("image power (radial)")
    plt.title(f"Image content spectrum — {tag}\n(ours below gt/splatter at high f = ours is blurry)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"/tmp/claude-67408/imgspec_{tag}.png",dpi=130)
    print(f"        wrote imgspec_{tag}.png")
