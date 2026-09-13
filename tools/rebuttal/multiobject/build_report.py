"""Self-contained HTML report for the compositionality result (images + orbit videos
base64-embedded). Open rebuttal/multiobject/report.html in a browser."""
import base64, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img(p, w="100%"):
    return f'<img style="width:{w};border:1px solid #ccc;border-radius:4px" src="data:image/png;base64,{b64(p)}"/>'


def vid(p, w="48%"):
    return (f'<video controls loop autoplay muted playsinline style="width:{w};border:1px solid #ccc;'
            f'border-radius:4px;margin:4px" src="data:video/mp4;base64,{b64(p)}"></video>')


orbits = sorted(glob.glob(f"{HERE}/videos/*_orbit.mp4"))
vids = "\n".join(vid(o) for o in orbits)

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Infer3D — multi-object (2-chair) compositionality</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2em auto;padding:0 1.2em;color:#1a1a1a;line-height:1.5}}
h1{{border-bottom:3px solid #2a6;padding-bottom:.3em}} h2{{margin-top:1.7em;color:#164}}
.cap{{color:#555;font-size:.9em;margin:.3em 0 1.3em}} .key{{background:#f2f8f4;border-left:4px solid #2a6;padding:.7em 1em;border-radius:4px}}
table{{border-collapse:collapse;margin:1em 0}} td,th{{border:1px solid #ccc;padding:5px 10px;text-align:center}}
code{{background:#eee;padding:1px 4px;border-radius:3px}} .lbl{{font-size:.82em;color:#444;font-family:monospace}}
</style></head><body>
<h1>Multi-object compositionality: 2-chair scenes from a single image</h1>
<p class="cap">Models trained ONLY on single SRN chairs (DiffAE prior + Splatter-Image lifter). Segment each
chair (SAM) &rarr; per-object Infer3D analysis-by-synthesis &rarr; compose. 20 scenes, held-out views.
Reviewer akZb Q3 ("unseen number of objects").</p>

<div class="key">The per-object reconstructions are sound; the end-to-end number is limited by single-view
<b>placement</b>. With a per-object 7-DoF registration (the analog of the paper's ICP-Chamfer alignment,
and less than the known pose FINV assumes), <b>cross-validated</b> (fit on half the views, report the
disjoint half), held-out PSNR is <b>18.2 / SSIM 0.80 / LPIPS 0.19</b>, fit-vs-held-out gap only 0.24 dB
(generalizes &rArr; not overfitting; 10/20 scenes &ge;18).</p></div>

<table>
<tr><th>setting</th><th>PSNR&uarr;</th><th>SSIM&uarr;</th><th>LPIPS&darr;</th></tr>
<tr><td>feed-forward Splatter on 2-chair image</td><td>13.4</td><td>&mdash;</td><td>0.31</td></tr>
<tr><td>Infer3D end-to-end (heuristic composition)</td><td>14.0</td><td>0.72</td><td>0.27</td></tr>
<tr><td><b>Infer3D + per-object registration (held-out)</b></td><td><b>18.2</b></td><td><b>0.80</b></td><td><b>0.19</b></td></tr>
</table>

<h2>1. Orbit videos — GT vs Infer3D (registered), all scene views</h2>
<p class="lbl">each clip: left = ground truth, right = ours; frame label marks input / fit / <b>held-out</b> views.
Scenes span strong (0009), mid (0000/0008), and hardest (0006/0011).</p>
<div style="display:flex;flex-wrap:wrap;justify-content:center">{vids}</div>
<p class="cap">Two distinct, correctly-placed chairs track the ground-truth orbit &mdash; not a fused blob.
Softness is the 128&sup2; single-view-lifted prior (shared with the single-object results), not a failure of composition.</p>

<h2>2. Feed-forward vs Infer3D (why feed-forward can't do this)</h2>
<p class="lbl">rows = scenes; columns = GT | anchored feed-forward | Infer3D. Novel views 1/10/19.</p>
{img(f"{HERE}/baseline_eval/montage_gt_ff_ours.png")}
<p class="cap">A single-chair-trained feed-forward lifter fed a 2-chair image fuses both chairs into one mass at
novel views; Infer3D composes two separate objects.</p>

<h2>3. Held-out registered reconstructions (stills)</h2>
<p class="lbl">rows = scenes (0008, 0011, 0009, 0006); columns = GT | ours. Views NOT used to fit the registration.</p>
{img(f"{HERE}/placement_crossval/crossval_heldout_montage.png", "60%")}

<p class="cap" style="margin-top:2em;border-top:1px solid #ccc;padding-top:1em">Honest scoping: a fully
<i>automatic</i> single-view placement (no registration) stays ~14 &mdash; input-view fitting overfits and the
monocular depth ratio is too weak; single-image multi-object layout is a separate open sub-problem. Numbers
and scripts: <code>rebuttal/multiobject/</code> (<code>placement_crossval.py</code>, <code>oracle_align*.py</code>,
<code>make_crossval_videos.py</code>).</p>
</body></html>"""
out = f"{HERE}/report.html"
with open(out, "w") as f:
    f.write(html)
print("wrote", out, f"({os.path.getsize(out)//1024} KB)")
