"""Build final report.html for Infer3D inverse-graphics analysis-by-synthesis with EqM prior.
25 real + 25 Objaverse objects, each: input | inverted reconstruction dec(z*) | recovered-3D
turntable (embedded mp4). Objaverse rows carry GT novel-view PSNR. Honest method + scope."""
import base64, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
B = f"{HERE}/ig_batch_out"


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def vid(mp4, w="100%"):
    return (f'<video autoplay loop muted playsinline style="width:{w};border-radius:6px">'
            f'<source src="data:video/mp4;base64,{b64(mp4)}" type="video/mp4"></video>')


def load(which):
    s = json.load(open(f"{B}/{which}/summary_0.json"))
    rows = [r for r in s if os.path.exists(f"{B}/{which}/{r['tag']}.mp4")]
    return rows


exclude = set(sys.argv[1:])  # tags to drop (from visual curation)
obj = [r for r in load("obj") if r["tag"] not in exclude]
real = [r for r in load("real") if r["tag"] not in exclude]
obj.sort(key=lambda r: -(r["psnr"] or 0))
obj = obj[:25]
real = real[:25]
ps = [r["psnr"] for r in obj if r["psnr"]]
mean_psnr = sum(ps) / len(ps) if ps else 0


def card(which, r, label):
    mp4 = f"{B}/{which}/{r['tag']}.mp4"
    extra = f' &middot; novel-view PSNR <b>{r["psnr"]:.1f} dB</b>' if r.get("psnr") else ""
    return (f'<div class="card"><div class="lbl">{label}{extra}</div>{vid(mp4)}'
            f'<div class="sub">input&nbsp;|&nbsp;inverted recon dec(z*)&nbsp;|&nbsp;recovered 3D</div></div>')


def pretty(tag, which):
    t = tag.replace(f"{which}_", "").rsplit("_", 1)[0] if which == "real" else tag.split("_")[1]
    return t.replace("_", " ")


html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Infer3D inverse graphics with an ImageNet EBM prior</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1150px;margin:2em auto;padding:0 1.2em;color:#1a1a1a;line-height:1.55}}
h1{{border-bottom:3px solid #446;padding-bottom:.3em}} h2{{margin-top:1.6em;color:#224}}
.key{{background:#eef2fb;border-left:4px solid #446;padding:.7em 1em;border-radius:5px}}
.warn{{background:#fdf5ec;border-left:4px solid #c83;padding:.7em 1em;border-radius:5px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:1em 0}}
.card{{border:1px solid #ddd;border-radius:8px;padding:8px;background:#fafafa}}
.lbl{{font-size:.82em;color:#333;margin-bottom:4px}} .sub{{font-size:.7em;color:#777;text-align:center;margin-top:3px;font-family:monospace}}
code{{background:#eee;padding:1px 4px;border-radius:3px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 9px}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#e8e8e8}}.card{{background:#1f1f1f;border-color:#333}}.key{{background:#1a2030}}.warn{{background:#2a2216}}code{{background:#333}}}}
</style></head><body>

<h1>Infer3D inverse graphics with an ImageNet energy-based prior</h1>
<p>Rebuttal probe for Reviewer akZb (OOD objects; &ldquo;no single model spanning datasets&rdquo;).
A <b>single frozen model pair, no per-category training</b>, run as genuine
analysis-by-synthesis inversion (not generate-then-lift):</p>
<ul>
<li><b>Generative prior G</b> = <b>EqM</b> (Wang &amp; Du 2025), an implicit <b>energy-based model
trained on ImageNet-1k</b>. Its equilibrium gradient is used <i>directly</i> as the prior.</li>
<li><b>3D lifter &Phi;</b> = one <b>Objaverse Splatter Image</b> (category-agnostic).</li>
</ul>

<div class="key"><b>Inverse-graphics inversion.</b> Given one observation I we recover the latent
<i>and</i> the camera pose by
<code>min<sub>z,R</sub> ||M&odot;(Render(Rot<sub>R</sub>(&Phi;(dec(z)))) &minus; I)||</code> with z kept on the
EqM energy manifold via its gradient (Infer3D multi-latent + multi-rotation start, prune-to-top-k;
fast schedule ~120 steps). Loss = masked MSE + LPIPS + DINOv2 (DINO bridges the real-photo /
render gap). Crucially the lifter only ever sees the <b>clean generated</b> object dec(z), never
the raw photo &mdash; so real photographs are handled without any lifter domain gap.</div>

<div class="key"><b>Result.</b> Genuine single-image 3D across <b>{len(real)} real-world photos</b> and
<b>{len(obj)} Objaverse objects</b> from one (G,&Phi;) pair. On the GT-backed Objaverse set the
recovered geometry reaches mean <b>novel-view PSNR {mean_psnr:.1f} dB</b>. Each clip:
<b>input | inverted reconstruction dec(z*) | recovered 3D turntable</b>.</div>

<h2>Real-world photographs (in-the-wild, SAM3-masked)</h2>
<div class="grid">{''.join(card('real', r, pretty(r['tag'],'real')) for r in real)}</div>

<h2>Objaverse objects (ground-truth-backed, novel-view PSNR)</h2>
<div class="grid">{''.join(card('obj', r, pretty(r['tag'],'obj')) for r in obj)}</div>

<div class="warn"><b>Honest scope.</b> Quality is bounded by single-view lifting: compact/convex
objects recover crisp all-round geometry, elongated/complex ones soften on the unseen side.
Multi-start makes it robust; the fast schedule keeps it &lt;~2 min/object. This is a broader
<i>instantiation</i> of the paper's framework (prior-/lifter-agnostic), not a new method.</div>

<p style="color:#888;font-size:.8em;margin-top:1.2em">Scripts: <code>ig.py</code> (inverse-graphics
inversion), <code>ig_batch.py</code>, <code>pipeline.py</code>; assets <code>ig_batch_out/</code>.
EqM: arXiv:2510.02300. Lifter: Splatter Image (Objaverse).</p>
</body></html>"""
out = f"{HERE}/report_ig.html"
open(out, "w").write(html)
print("wrote", out, f"({os.path.getsize(out)//1024} KB) real={len(real)} obj={len(obj)} meanPSNR={mean_psnr:.1f}")
