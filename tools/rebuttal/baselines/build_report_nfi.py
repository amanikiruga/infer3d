"""NFI-only view of the rebuttal report -> report/index_baseline_nfi.html.
Reuses the SAME videos + per-object scores as index.html; just filters to NFI vs ours
across all settings/tasks (nothing new generated). For sharing with an NFI-focused reader."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, json, os
WT = f"{_EXT}/splatter-image-rebuttal"
R = f"{WT}/rebuttal/results"; REP = f"{WT}/rebuttal/report"
SC = json.load(open(f"{R}/per_object_scores.json"))
SC_CTRL = json.load(open(f"{R}/per_object_scores_ctrl.json")) if os.path.exists(f"{R}/per_object_scores_ctrl.json") else {}
AGG = {"ours_so3": 19.2, "ours_ctrl": 21.4, "nfi_own": 14.3, "nfi_oracle": 17.7, "nfi_ctrl": 19.85}
CD = {"ours": 0.0091, "nfi": 0.0037}
RCD = {"ours": 0.131, "nfi": 0.491}


def nvs_ood():
    out = []
    for f in sorted(glob.glob(f"{REP}/videos/nvs/nfi_*.mp4")):
        oid = os.path.basename(f)[4:-4]; sc = SC.get(oid, {})
        cap = " · ".join([f"ours {sc.get('ours','')}", f"NFI own {sc.get('nfi_own','')}",
                          f"NFI oracle {sc.get('nfi_oracle','')}"])
        out.append(f'  <div class="v"><video src="videos/nvs/{os.path.basename(f)}" controls loop muted playsinline preload="metadata"></video>'
                   f'<div class="cap"><code>{oid[:8]}</code> — PSNR @ GT cameras: {cap}</div></div>')
    return out


def nvs_ctrl():
    out = []
    for f in sorted(glob.glob(f"{REP}/videos/nvs_ctrl/nfi_*.mp4")):
        oid = os.path.basename(f)[4:-4]; sc = SC_CTRL.get(oid, {})
        cap = " · ".join([f"ours {sc.get('ours_ctrl','')}", f"NFI {sc.get('nfi_ctrl','')}"])
        out.append(f'  <div class="v"><video src="videos/nvs_ctrl/{os.path.basename(f)}" controls loop muted playsinline preload="metadata"></video>'
                   f'<div class="cap"><code>{oid[:8]}</code> — ID-pose PSNR @ GT cameras: {cap}</div></div>')
    return out


def rc_orbit():
    out = []
    for f in sorted(glob.glob(f"{REP}/videos/by_method/nfi_rc_*.mp4")):
        oid = os.path.basename(f)[len("nfi_rc_"):-4]
        out.append(f'  <div class="v"><video src="videos/by_method/{os.path.basename(f)}" controls loop muted playsinline preload="metadata"></video>'
                   f'<div class="cap">RealCars <code>{oid}</code> — input ‖ OURS ‖ NFI (orbit; no NVS, Chamfer task)</div></div>')
    return out


H = ["""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Infer3D vs nerf-from-image (NFI) — full comparison</title><style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaed;--mut:#9aa3af;--line:#262b34;--accent:#5b9dff;--good:#39b57a;--bad:#e5678a}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d23;--mut:#5b636e;--line:#e3e7ec;--accent:#2f6fe0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1640px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:21px;margin:40px 0 10px;border-bottom:2px solid var(--accent);padding-bottom:6px}
h3{font-size:16px;margin:20px 0 8px;color:var(--accent)}p{color:var(--ink)}.mut{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}th,td{border:1px solid var(--line);padding:7px 10px;text-align:center}
th{background:rgba(127,127,127,.08)}td.l,th.l{text-align:left}.best{color:var(--good);font-weight:700}.worst{color:var(--bad)}
.grid{display:grid;grid-template-columns:1fr;gap:20px;margin:14px 0}@media(min-width:820px){.grid{grid-template-columns:1fr 1fr}}
video{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#000}.v .cap{font-size:13.5px;color:var(--mut);margin-top:5px}
code{background:rgba(127,127,127,.12);padding:1px 5px;border-radius:4px;font-size:13px}.pill{display:inline-block;font-size:12px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--mut);margin-left:6px}
details{margin:8px 0}summary{cursor:pointer;color:var(--accent);font-weight:600}
</style></head><body><div class="wrap">"""]
H.append('<h1>Infer3D vs. nerf-from-image (NFI) <span class="pill">NFI-only comparison</span></h1>')
H.append('<p class="mut">A focused extract of the rebuttal evidence: only <b>NFI (Bootstrapped Radiance-Field '
 'Inversion)</b> vs <b>Infer3D (ours)</b>, across all settings and both tasks. Same videos, scores, and '
 'protocol as the full report. NFI is the direct analysis-by-synthesis / NeRF-inversion analog to ours '
 '(encoder + PnP pose + latent inversion of a per-category 3D-aware GAN), and the only baseline with a '
 'dataset-aligned pose frame — hence the only one we can score with calibrated NVS.</p>')

H.append('<h2>Novel-view synthesis — Infer3D vs NFI</h2>'
 '<table>'
 '<tr><th class="l">Setting (SO(3) ShapeNet cars, PSNR ↑)</th><th>NFI</th><th>Infer3D (ours)</th></tr>'
 '<tr><td class="l">In-distribution (canonical pose)</td><td>19.85</td><td class="best">20.7</td></tr>'
 '<tr><td class="l">SO(3) OOD — method’s own estimated pose</td><td class="worst">14.3</td><td class="best">19.2</td></tr>'
 '<tr><td class="l">SO(3) OOD — oracle GT pose</td><td>17.7</td><td class="mut">— (recovers pose itself)</td></tr>'
 '<tr><td class="l"><b>Degradation, in-distribution → SO(3)</b></td><td class="worst"><b>−5.5 dB</b></td><td class="best"><b>−1.5 dB</b></td></tr>'
 '</table>'
 '<p class="mut">PSNR over 25 objects, <b>identical evaluation harness</b> (same scorer and target cameras for both '
 'methods). <b>In-distribution the two are on par</b> (interleaved per object). <b>Under SO(3) OOD, NFI collapses '
 '5.5&nbsp;dB while ours loses only 1.5&nbsp;dB.</b> Handed the <b>oracle GT pose</b> — the pose NFI fails to estimate — '
 'NFI recovers only to 17.7, still below ours, so its failure is <i>both</i> pose estimation <i>and</i> OOD '
 'reconstruction; ours needs no oracle. (Pose-removed shape Chamfer omitted: a fairer mesh-based CD for ours is pending.)</p>')

H.append('<h2>In-distribution (canonical pose) — NVS (scored @ GT cameras)</h2>'
 '<p class="mut">The homefield setting (19.85 vs 20.7). Panels: <code>input ‖ OURS ‖ NFI ‖ GT</code> at the '
 'control cameras — the two are on par and interleave per object.</p>')
iv = nvs_ctrl()
H.append(f'<details open><summary>In-distribution — {len(iv)} objects (per-object scores)</summary><div class="grid">')
H.append("\n".join(iv)); H.append('</div></details>')

H.append('<h2>SO(3) OOD — NVS (scored @ GT cameras)</h2>'
 '<p class="mut">Each clip is built from the <b>actual scored renders</b>: every panel is the same GT target '
 'camera per frame, so the video <b>is</b> the PSNR. Panels: <code>input ‖ OURS ‖ NFI own-pose ‖ NFI oracle-pose ‖ GT</code> '
 '— watch the oracle panel snap toward GT while own-pose sits at the wrong viewpoint.</p>')
nv = nvs_ood()
H.append(f'<details open><summary>SO(3) OOD — {len(nv)} objects (per-object scores)</summary><div class="grid">')
H.append("\n".join(nv)); H.append('</div></details>')

H.append('<h2>RealCars (synthetic→real)</h2>'
 '<p class="mut">RealCars is single-view + LiDAR pseudo-GT → <b>no NVS metric</b> (this task is scored by '
 'ICP-Chamfer; the fair mesh-based CD for ours is pending, so numbers are omitted here). Clips are orbit '
 'visualizations only: <code>input ‖ OURS ‖ NFI</code>. NFI’s pixel/latent inversion locks onto real texture the '
 'synthetic prior can’t represent — watch it collapse under real appearance, while ours’ DINOv2 semantic objective holds.</p>')
rv = rc_orbit()
H.append(f'<details open><summary>RealCars — {len(rv)} scenes (orbit)</summary><div class="grid">')
H.append("\n".join(rv)); H.append('</div></details>')

H.append('<p class="mut" style="margin-top:30px">Extract of the full rebuttal report (NFI only). Same sandbox, '
 'scripts in <code>rebuttal/baselines/</code>, numbers in <code>rebuttal/results/</code>.</p>')
H.append("""<script>
// Play videos only while they are in view; pause off-screen (saves CPU with many clips).
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting) { v.play().catch(()=>{}); }
    else { v.pause(); }
  }
}, { threshold: 0.25 });
document.querySelectorAll('video').forEach(v => {
  v.muted = true; v.loop = true; v.playsInline = true;
  // reveal videos inside collapsed <details> when opened
  io.observe(v);
});
document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', () => {
  d.querySelectorAll('video').forEach(v => { if (d.open) io.observe(v); });
}));
</script>""")
H.append('</div></body></html>')
open(f"{REP}/index_baseline_nfi.html", "w").write("\n".join(H))
print(f"wrote index_baseline_nfi.html : OOD {len(nv)} · ID {len(iv)} · RealCars {len(rv)}")
