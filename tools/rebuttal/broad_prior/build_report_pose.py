"""Report for OPTION 1 (pose reasoning) verified positives: 5 objects where, from an UNNATURAL
top-down observation, the EqM+lifter inversion recovers a coherent natural 3D at a pose that
reproduces the observation. Verification = rendering the recovered 3D back at the input pose
(last column) reproduces the top-down input, quantified by reprojection IoU."""
import base64, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
FIVE = ["teapot_4c30437c", "teapot_a5def4de", "teapot_7fdad1ee", "teapot_ba49973a", "teapot_27cff371"]
S = {r["tag"]: r for r in json.load(open(f"{HERE}/pose4b_out/summary_0.json")) + json.load(open(f"{HERE}/pose4b_out/summary_14.json"))}


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def card(tag):
    r = S[tag]
    return (f'<div class="card"><div class="lbl">teapot &middot; obs elevation {abs(r["elev"]):.0f}&deg; '
            f'&middot; reprojection IoU <b>{r["reproj_iou"]:.2f}</b></div>'
            f'<video autoplay loop muted playsinline style="width:100%;border-radius:6px">'
            f'<source src="data:video/mp4;base64,{b64(f"{HERE}/pose4b_out/{tag}.mp4")}" type="video/mp4"></video>'
            f'<div class="sub">top-down input | EqM natural hypothesis | feed-forward 3D | Infer3D 3D | '
            f'<b>recovered 3D @ input pose</b> (reproduces col 1)</div></div>')


import numpy as np
mri = float(np.mean([S[t]["reproj_iou"] for t in FIVE]))
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Infer3D pose reasoning (EqM) — 5 positives</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1150px;margin:2em auto;padding:0 1.2em;color:#1a1a1a;line-height:1.55}}
h1{{border-bottom:3px solid #446;padding-bottom:.3em}} .key{{background:#eef2fb;border-left:4px solid #446;padding:.7em 1em;border-radius:5px}}
.warn{{background:#fdf5ec;border-left:4px solid #c83;padding:.7em 1em;border-radius:5px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:8px;margin:12px 0;background:#fafafa}}
.lbl{{font-size:.85em;color:#333;margin-bottom:4px}} .sub{{font-size:.7em;color:#777;text-align:center;margin-top:3px;font-family:monospace}}
code{{background:#eee;padding:1px 4px;border-radius:3px}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#e8e8e8}}.card{{background:#1f1f1f;border-color:#333}}.key{{background:#1a2030}}.warn{{background:#2a2216}}code{{background:#333}}}}
</style></head><body>
<h1>Option 1 &mdash; pose reasoning from an unnatural view (EqM + lifter), 5 verified positives</h1>
<p>Observation = an <b>unnatural top-down</b> Objaverse render. The EqM ImageNet prior proposes a
<b>natural</b> teapot; a brute-force pose search (rotation frozen while scale+translation fit, so it
cannot drift to a side view) recovers the transform that explains the observation. Feed-forward
lifting of the top-down view collapses.</p>
<div class="key"><b>Verification (the last column).</b> Rendering the recovered 3D back at the INPUT
camera reproduces the top-down observation &mdash; matching orientation, spout and handle
positions &mdash; at mean <b>reprojection IoU {mri:.2f}</b> across these 5 objects. The recovered 3D
(column 4) is a coherent natural teapot; the feed-forward lift of the top-down view (column 3) is a
flat/degenerate shell.</div>
{''.join(card(t) for t in FIVE)}
<div class="warn"><b>Honest scope.</b> This works for objects with a <b>distinctive silhouette</b>
(teapot spout+handle make the top-down pose observable). It fails for near-symmetric objects (a
top-down mug/vase/bottle is silhouette-ambiguous) and the recovered <i>appearance</i> is a generic
EqM instance, not the specific object (so novel-view pixel match is limited; the unseen back is soft,
a single-view-lifting limit). The verifiable claim is pose recovery (reprojection) + coherent natural
3D, which these 5 satisfy.</div>
<p style="color:#888;font-size:.8em">Scripts: <code>ig_pose4.py</code> (brute-force pose), <code>ig.py</code>,
<code>pipeline.py</code>. Videos: <code>pose4b_out/</code>. EqM: arXiv:2510.02300.</p>
</body></html>"""
open(f"{HERE}/report_pose.html", "w").write(html)
print("wrote report_pose.html, 5 positives, mean reproj IoU", round(mri, 3))
