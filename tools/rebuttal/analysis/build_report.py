"""Assemble a self-contained HTML report (images+videos base64-embedded) for the
in-distribution-gap analysis. Honest, visually-grounded findings. Output: report.html."""
import base64, os
HERE = os.path.dirname(os.path.abspath(__file__))
A = f"{HERE}/report_assets"

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def img(path, w="100%"):
    ext = "png"
    return f'<img style="width:{w};border:1px solid #ccc;border-radius:4px" src="data:image/{ext};base64,{b64(path)}"/>'

def vid(path, w="70%"):
    return (f'<video controls loop muted style="width:{w};border:1px solid #ccc;border-radius:4px">'
            f'<source src="data:video/mp4;base64,{b64(path)}" type="video/mp4"></video>')

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Infer3D — in-distribution gap analysis</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2em auto;padding:0 1.2em;background:#15181c;color:#d8dde3;line-height:1.5}}
h1{{border-bottom:3px solid #3ba776;padding-bottom:.3em;color:#eef2f5}} h2{{margin-top:1.8em;color:#6fcf9c}}
a{{color:#6fcf9c}}
.cap{{color:#9aa3ad;font-size:.9em;margin:.3em 0 1.4em}} .key{{background:#1b2a22;border-left:4px solid #3ba776;padding:.8em 1em;border-radius:4px}}
.warn{{background:#2b1f1a;border-left:4px solid #e6805a;padding:.8em 1em;border-radius:4px}}
table{{border-collapse:collapse;margin:1em 0}} td,th{{border:1px solid #333;padding:5px 10px;text-align:center}}
code{{background:#262b31;color:#d8dde3;padding:1px 4px;border-radius:3px}} .lbl{{font-size:.82em;color:#9aa3ad;font-family:monospace}}
</style></head><body>

<h1>Infer3D: what the in-distribution gap actually is</h1>
<p class="cap">Rebuttal analysis for reviewer XsZX / AC priority 2 ("resolve the ID trade-off, analyze its
cause"). Every render reproduced from the paper's own metric/render code; ID cells match the paper
(DiffAE ours 20.6 / Splatter 23.0 / StyleGAN ours 17.8). CO3D hydrants, n=18.</p>

<div class="warn"><b>Methodology note (why this report is qualitative-first).</b> An initial
purely-numerical pass (averaged error power spectrum) suggested "high frequencies matched, gap is
low-frequency." Looking at the actual images <b>refuted</b> that — the error spectrum is dominated by
the shared object silhouette and by GT texture both methods miss. The pixels, per-object breakdown,
and image-content spectra below are the truth.</div>

<h2>Finding: the gap is inversion-failure-driven, not a uniform loss</h2>
<div class="key">On objects where 2D-prior inversion <b>succeeds</b>, Infer3D ≈ the feed-forward Splatter
lifter (median DiffAE gap 1.7 dB). The gap is driven by a <b>minority of objects where inversion
collapses</b> to a wrong mode (blurry blob). The encoder-initialized <b>DiffAE</b> makes this rare
(4/18); latent-only <b>StyleGAN</b> is broadly weaker (14/18 &gt; 4 dB). These failures are exactly
the high-reconstruction-loss cases the OOD detector flags (Table 6) and the adaptive router (Table 4)
sends to the feed-forward lifter — so the ID penalty is neutralized in deployment.</div>

<h2>1. The gap is heavy-tailed / failure-driven</h2>
{img(f"{A}/gap_distribution.png")}
<p class="cap">Per-object PSNR gap (Splatter − Infer3D), sorted. DiffAE: median 1.7 dB, a tail of 4
failures &gt;4 dB drives the mean (2.6). StyleGAN: broadly weaker (median 5.2). The mean overstates
the typical gap.</p>

<h2>2. Typical objects — Infer3D ≈ Splatter</h2>
<p class="lbl">columns: GT &nbsp;|&nbsp; Splatter &nbsp;|&nbsp; Infer3D (ours) &nbsp;|&nbsp; 4×|Splatter−GT| &nbsp;|&nbsp; 4×|Ours−GT|</p>
{img(f"{A}/montage_typical.png")}
<p class="cap">On successful objects both methods are close (and both are soft vs the real photo — that
softness is the 128² Gaussian-splat render pipeline, shared by both, not specific to ours). Error maps
are comparable, dominated by the silhouette/edge.</p>

<h2>3. Failure objects — Infer3D collapses (drives the gap)</h2>
<p class="lbl">columns: GT &nbsp;|&nbsp; Splatter &nbsp;|&nbsp; Infer3D (ours) &nbsp;|&nbsp; 4×|Splatter−GT| &nbsp;|&nbsp; 4×|Ours−GT|</p>
{img(f"{A}/montage_failure.png")}
<p class="cap">On hard objects (e.g. unusual white hydrants) the generator cannot invert the input and
optimization lands in a wrong mode → a blurry blob, while the feed-forward lifter still reconstructs a
recognizable object. These 4 objects drive DiffAE's mean gap.</p>

<h2>4. Orbit videos (ours vs Splatter vs GT, all novel views)</h2>
<p class="lbl">GT &nbsp;|&nbsp; Splatter &nbsp;|&nbsp; Infer3D (ours) — typical object</p>
{vid(f"{A}/orbit_typical.mp4")}
<p class="lbl" style="margin-top:1em">GT &nbsp;|&nbsp; Splatter &nbsp;|&nbsp; Infer3D (ours) — failure object (collapse)</p>
{vid(f"{A}/orbit_failure.mp4")}

<h2>5. What it is NOT (ruled out)</h2>
<table>
<tr><th>hypothesis</th><th>test</th><th>result</th></tr>
<tr><td>uniform fine-detail blur</td><td>image-content spectrum, per-object sharpness</td><td>on successes ours≈Splatter sharpness; not uniform blur</td></tr>
<tr><td>registration / shading offset</td><td>per-image shift + per-channel gain/bias search</td><td>closes only ~0.2 dB of the 3 dB gap</td></tr>
<tr><td>"high-freq matched" (error spectrum)</td><td>compare to image spectrum + pixels</td><td>artifact: silhouette-dominated; refuted by images</td></tr>
</table>

<h2>6. Optimization dynamics & cost (AC priority 3 / akZb / uH4P)</h2>
{img(f"{A.replace('/report_assets','')}/dynamics_id_vs_ood.png")}
<p class="cap">From logged runs (n=43 ID/43 OOD), no new compute. At feed-forward initialization ID loss
is 0.030 vs OOD 0.132 (4.3×) — the failures/OOD are separable from the initial loss at 0.43 s
(AUROC 0.97), which is what makes the adaptive router cheap. Full run 783 iters / 6 stages, peak VRAM
37.5 GB; amortized 191 s/sample at a 10%-OOD stream (Table 4).</p>

<p class="cap" style="margin-top:2em;border-top:1px solid #ccc;padding-top:1em">Assets:
<code>report_assets/</code> (montages, gap distribution, orbit mp4s), <code>FINDINGS.md</code>,
<code>DYNAMICS.md</code>, <code>RESPONSE_ID_AND_DYNAMICS.md</code>. Scripts: <code>render_id_pair*.py</code>,
<code>frequency_gap.py</code>, <code>make_report_assets.py</code>, <code>compare_variants.py</code>.</p>
</body></html>"""

out = f"{HERE}/report.html"
with open(out, "w") as f:
    f.write(html)
print("wrote", out, f"({os.path.getsize(out)//1024} KB)")
