"""Build qualitative assets for the honest ID-gap report:
 - typical-object montage (small gap) and failure montage (large gap)
 - per-object gap distribution (DiffAE + SG) showing the heavy tail
 - orbit videos (ours | splatter | GT) stitched from existing rendered views
All CPU, from PNGs already on disk. Outputs -> report_assets/.
"""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import os, glob, json, numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import imageio.v2 as imageio

RES = f"{_EXT}/splatter-image-rebuttal/rebuttal/results"
OUT = f"{_EXT}/splatter-image-rebuttal/rebuttal/frequency/report_assets"
os.makedirs(OUT, exist_ok=True)
def load(p): return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)/255.
def to8(x): return np.clip(x*255,0,255).astype(np.uint8)
def psnr(a,b): return -10*np.log10(np.mean((a-b)**2)+1e-12)

DA = f"{RES}/freq_co3d/hyd_diffae_id"      # ours / splatter / gt(/targets)
def views(kind, oid): return sorted(glob.glob(f"{DA}/{kind}/{oid}/*.png"))
def gtviews(oid): return sorted(glob.glob(f"{DA}/gt/{oid}/targets/*.png"))

def montage(oids, name, vi=30):
    rows=[]
    for oid in oids:
        ro,rs,ts=views("ours",oid),views("splatter",oid),gtviews(oid)
        j=min(vi,len(ro)-1)
        g,s,r=load(ts[j]),load(rs[j]),load(ro[j])
        eo=np.clip(np.abs(r-g)*4,0,1); es=np.clip(np.abs(s-g)*4,0,1)
        gap=psnr(r,g)-psnr(s,g)
        strip=to8(np.concatenate([g,s,r,es,eo],1))
        # label bar
        rows.append((strip,oid,gap))
    h=rows[0][0].shape[0]; w=rows[0][0].shape[1]
    canvas=np.concatenate([r[0] for r in rows],0)
    Image.fromarray(canvas).save(f"{OUT}/montage_{name}.png")
    print(f"montage_{name}.png:", [(o, round(g,1)) for _,o,g in rows])

# typical (small gap) and failures (large gap) — from perobj breakdown
montage(["497_71368_139133","519_74488_144698","411_55970_107894"], "typical")
montage(["304_31873_60475","417_57593_110775","421_58385_112527"], "failure")

# gap distribution DiffAE + SG (per-object, sorted) — heavy tail
def per_obj_gaps(tag):
    base=f"{RES}/freq_co3d/{tag}"
    gaps=[]
    for oid in sorted(os.listdir(f"{base}/ours")):
        ro=views("ours",oid) if tag=="hyd_diffae_id" else sorted(glob.glob(f"{base}/ours/{oid}/*.png"))
        rs=sorted(glob.glob(f"{base}/splatter/{oid}/*.png"))
        ts=sorted(glob.glob(f"{base}/gt/{oid}/targets/*.png"))
        po=np.mean([psnr(load(a),load(b)) for a,b in list(zip(ro,ts))[::8]])
        ps=np.mean([psnr(load(a),load(b)) for a,b in list(zip(rs,ts))[::8]])
        gaps.append(ps-po)
    return np.array(sorted(gaps))
gd=per_obj_gaps("hyd_diffae_id"); gs=per_obj_gaps("hyd_sg_id")
plt.figure(figsize=(7,4))
plt.bar(np.arange(len(gd))-0.2, gd, width=0.4, label=f"DiffAE (median {np.median(gd):.1f}, mean {gd.mean():.1f})", color="seagreen")
plt.bar(np.arange(len(gs))+0.2, gs, width=0.4, label=f"StyleGAN (median {np.median(gs):.1f}, mean {gs.mean():.1f})", color="steelblue")
plt.axhline(4, color="crimson", ls="--", lw=1, label="failure threshold (4 dB)")
plt.xlabel("object (sorted by gap)"); plt.ylabel("PSNR gap  Splatter − Infer3D (dB)")
plt.title("ID gap is heavy-tailed: a minority of failures drives the mean\n(CO3D hydrants, n=18)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/gap_distribution.png", dpi=140)
print("gap_distribution.png  DiffAE fails>4:", int((gd>4).sum()), " SG fails>4:", int((gs>4).sum()))

# orbit videos: ours | splatter | gt, stitched from existing views
def orbit(oid, name, stride=2, fps=12):
    ro,rs,ts=views("ours",oid),views("splatter",oid),gtviews(oid)
    n=min(len(ro),len(rs),len(ts)); frames=[]
    for j in range(0,n,stride):
        g,s,r=load(ts[j]),load(rs[j]),load(ro[j])
        frames.append(to8(np.concatenate([g,s,r],1)))
    imageio.mimsave(f"{OUT}/orbit_{name}.mp4", frames, fps=fps, codec="libx264",
                    output_params=["-pix_fmt","yuv420p"])
    print(f"orbit_{name}.mp4 ({len(frames)} frames) [GT | Splatter | Ours]")
orbit("411_55970_107894","typical")   # good
orbit("304_31873_60475","failure")    # collapse
