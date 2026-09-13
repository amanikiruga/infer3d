"""Generate report/index.html grouped BY BASELINE (user layout): each baseline in its own
section with its NVS numbers + per-object videos (input|ours|baseline|GT) and per-object
scores. Self-contained, theme-aware, relative-path mp4s."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, json, os
WT = f"{_EXT}/splatter-image-rebuttal"
R = f"{WT}/rebuttal/results"; REP = f"{WT}/rebuttal/report"
SC = json.load(open(f"{R}/per_object_scores.json"))

# ---- aggregates (validated) ----
# all shared-harness (same scorer/cameras for every method) so numbers are apples-to-apples:
# ours own-pose 19.2 / control 20.7; NFI own 14.3 / oracle 17.7 / control 19.85. (ours' own-eval
# gives 20.0/21.4 on its native harness — we report the shared-harness values for fair comparison.)
AGG = {"ours_so3": 19.2, "ours_ctrl": 20.7, "nfi_own": 14.3, "nfi_oracle": 17.7, "nfi_ctrl": 19.85,
       "eg3d_own": 12.3, "finv_own": 12.3}
CD = {"ours": 0.0091, "nfi": 0.0037, "eg3d": 0.0127, "pigan": 0.0075, "finv": 0.0119}
# in-distribution (control) Chamfer — same 25 GT cars, homefield shape. Every method degrades
# only gracefully OOD -> the big NVS collapse is pose, not shape.
CD_CTRL = {"ours": 0.0077, "nfi": 0.0010, "eg3d": 0.0086, "pigan": 0.0058, "finv": 0.0096}
# homefield reconstruction fidelity (each method IN ITS OWN DOMAIN, on real data):
#  eg3d/finv = input-view recon PSNR on the 25 real control cars; pigan_real = input-view recon
#  PSNR on real CARLA images (its training domain). Shows the inverters fit real in-domain input.
HF = {"eg3d_recon": 21.5, "finv_recon": 22.1, "pigan_real_recon": 26.1}
# RealCars cd_pred2gt mean (pose+scale removed; see caveat). ours/Splatter/SF3D/LGM from paper pipeline.
RCD = {"pigan": 0.088, "ours": 0.131, "eg3d": 0.154, "finv": 0.169, "splatter": 0.219,
       "sf3d": 0.242, "lgm": 0.405, "nfi": 0.491}

BASE = [
 ("nfi", "NFI — nerf-from-image (Bootstrapped Radiance-Field Inversion)", "nfi_own",
  "The direct analysis-by-synthesis / NeRF-inversion analog to ours (encoder + PnP pose + latent opt). "
  "The ONLY baseline with a dataset-aligned pose frame, so it is the one we can evaluate with calibrated "
  "NVS. Own-pose NVS collapses under SO(3) (14.3); handed the <b>oracle GT pose</b> it recovers only to 17.7 "
  "— still below ours' own-pose 19–20, because OOD inputs also degrade its reconstruction."),
 ("eg3d", "3D-GAN-Inversion — EG3D + PTI (Ko et al., WACV'23)", "eg3d_own",
  "Triplane-GAN inversion with pose optimization + PTI generator fine-tuning. GAN inversion has no "
  "dataset-aligned pose estimator, so own-pose NVS collapses hardest (12.3). Recovers plausible generic "
  "shape (see Chamfer) but a wrong-identity car at an uncontrolled pose."),
 ("finv", "FINV-SV — multi-start particle inversion + pruning (the AC's cited method)", "finv_own",
  "FINV (3DV'24) instantiated single-view: N=16 latent particles → optimize each (latent+pose) → prune "
  "to 4 by input-consistency → refine → prune to 1 → PTI. <b>Result: FINV-SV (12.3) ≈ plain EG3D+PTI "
  "(12.3)</b> — the multi-start pruning machinery the AC cites does NOT close the OOD gap. Direct answer: "
  "the gain is not the particle scheme, it is our design."),
 ("pigan", "pi-GAN (SIREN 3D-GAN, CARLA prior)", None,
  "Latent-only inversion with a fixed frontal camera and a CARLA prior. Cannot estimate an arbitrary input "
  "pose, so it has no calibrated NVS; its free-orbit render is a diffuse blob — the genuine CARLA→ShapeNet / "
  "→real domain-gap collapse (the same geometry its Chamfer is computed on)."),
]


def vids_nvs(key):
    """NVS comparison videos built from the ACTUAL SCORED renders (at the 24 GT target
    cameras). Frame i = GT camera i; every panel is the same viewpoint = what PSNR measured."""
    out = []
    for f in sorted(glob.glob(f"{REP}/videos/nvs/{key}_*.mp4")):
        oid = os.path.basename(f)[len(key) + 1:-4]; sc = SC.get(oid, {})
        parts = []
        if "ours" in sc:
            parts.append(f"ours {sc['ours']}")
        sk = {"nfi": "nfi_own", "eg3d": "eg3d_own", "finv": "finv_own"}.get(key)
        if sk and sk in sc:
            parts.append(f"{key} own {sc[sk]}")
        if key == "nfi" and "nfi_oracle" in sc:
            parts.append(f"nfi oracle {sc['nfi_oracle']}")
        cap = " · ".join(parts)
        out.append(f'  <div class="v"><video src="videos/nvs/{os.path.basename(f)}" controls loop muted playsinline preload="metadata"></video>'
                   f'<div class="cap"><code>{oid[:8]}</code> — PSNR @ GT cameras: {cap}</div></div>')
    return out


SC_CTRL = json.load(open(f"{R}/per_object_scores_ctrl.json")) if os.path.exists(f"{R}/per_object_scores_ctrl.json") else {}


def vids_nvs_ctrl(key):
    """In-distribution (control) NVS videos, scored @ control GT cameras (input|ours|NFI|GT)."""
    out = []
    for f in sorted(glob.glob(f"{REP}/videos/nvs_ctrl/{key}_*.mp4")):
        oid = os.path.basename(f)[len(key) + 1:-4]; sc = SC_CTRL.get(oid, {})
        parts = []
        if "ours_ctrl" in sc:
            parts.append(f"ours {sc['ours_ctrl']}")
        if "nfi_ctrl" in sc:
            parts.append(f"nfi {sc['nfi_ctrl']}")
        cap = " · ".join(parts) if parts else "in-distribution (ID pose)"
        out.append(f'  <div class="v"><video src="videos/nvs_ctrl/{os.path.basename(f)}" controls loop muted playsinline preload="metadata"></video>'
                   f'<div class="cap"><code>{oid[:8]}</code> — ID-pose PSNR @ GT cameras: {cap}</div></div>')
    return out


def vids(key, task):
    fs = sorted(glob.glob(f"{REP}/videos/by_method/{key}_{task}_*.mp4"))
    out = []
    for f in fs:
        oid = os.path.basename(f)[len(key) + len(task) + 2:-4]
        sc = SC.get(oid, {})
        cap_parts = []
        if "ours" in sc:
            cap_parts.append(f"ours {sc['ours']}")
        sk = {"nfi": "nfi_own", "eg3d": "eg3d_own", "finv": "finv_own"}.get(key)
        if sk and sk in sc:
            cap_parts.append(f"{key} {sc[sk]}")
        if key == "nfi" and "nfi_oracle" in sc:
            cap_parts.append(f"nfi-oracle {sc['nfi_oracle']}")
        cap = " · ".join(cap_parts)
        rel = f"videos/by_method/{os.path.basename(f)}"
        out.append(f'  <div class="v"><video src="{rel}" controls loop muted playsinline preload="metadata"></video>'
                   f'<div class="cap"><code>{oid[:8]}</code> — PSNR: {cap}</div></div>')
    return out


H = []
H.append("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Infer3D rebuttal — NVS, FINV-SV, and per-baseline comparison</title><style>
:root{--bg:#0f1115;--card:#171a21;--ink:#e8eaed;--mut:#9aa3af;--line:#262b34;--accent:#5b9dff;--good:#39b57a;--bad:#e5678a}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d23;--mut:#5b636e;--line:#e3e7ec;--accent:#2f6fe0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:21px;margin:40px 0 10px;border-bottom:2px solid var(--accent);padding-bottom:6px}
h3{font-size:16px;margin:20px 0 8px;color:var(--accent)}p{color:var(--ink)}.mut{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;margin:14px 0}
.q{border-left:3px solid var(--accent);padding:10px 16px;background:var(--card);border-radius:8px;color:var(--mut);font-style:italic}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}th,td{border:1px solid var(--line);padding:7px 10px;text-align:center}
th{background:rgba(127,127,127,.08)}td.l,th.l{text-align:left}.best{color:var(--good);font-weight:700}.worst{color:var(--bad)}
.grid{display:grid;grid-template-columns:1fr;gap:12px;margin:12px 0}@media(min-width:760px){.grid{grid-template-columns:1fr 1fr}}
video{width:100%;border-radius:8px;border:1px solid var(--line);background:#000}.v .cap{font-size:12.5px;color:var(--mut);margin-top:3px}
code{background:rgba(127,127,127,.12);padding:1px 5px;border-radius:4px;font-size:13px}.pill{display:inline-block;font-size:12px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);color:var(--mut);margin-left:6px}
details{margin:8px 0}summary{cursor:pointer;color:var(--accent);font-weight:600}
</style></head><body><div class="wrap">""")

H.append('<h1>Infer3D vs. optimization-based reconstruction <span class="pill">rebuttal — NVS + FINV + per-baseline</span></h1>')
H.append('<div class="q">"…whether this particular instantiation advances what optimization-based reconstruction already offers … the experiments cannot currently separate the contribution of this specific design from the generic benefit of optimizing at test time … and notably FINV (3DV’24), which also employs multi-start particle inversion with pruning."</div>')

H.append('<div class="card"><h3 style="margin-top:0">Answer, with calibrated NVS</h3><p>'
 'We evaluate every optimization baseline on <b>SO(3) viewpoint OOD (ShapeNet cars)</b> with the same '
 'pose-sensitive NVS metric as our paper. Two independent controls separate generic test-time optimization '
 'from our design: (1) <b>oracle-pose NVS</b> — hand the method the GT camera it fails to estimate; '
 '(2) <b>pose-removed Chamfer</b> — shape only. Findings: generic inversion collapses on OOD pose '
 '(own-pose NVS 12–14 vs ours 19–20); even <b>granted the oracle GT pose</b>, the strongest baseline (NFI) '
 'reaches only 17.7, still below ours’ own-pose; and <b>FINV-SV’s multi-start pruning lands exactly where '
 'plain EG3D+PTI does (12.3)</b>. The gain is the specific instantiation, not the act of optimizing at test time.'
 '</p></div>')

# headline NVS table
H.append('<h2>Headline — SO(3) NVS PSNR ↑ (25 cars, shared harness)</h2><table>')
H.append('<tr><th class="l">setting</th><th>Infer3D (ours)</th><th>NFI</th><th>EG3D+PTI</th><th>FINV-SV</th></tr>')
H.append(f'<tr><td class="l">own (estimated) pose</td><td class="best">{AGG["ours_so3"]}</td><td>{AGG["nfi_own"]}</td><td class="worst">{AGG["eg3d_own"]}</td><td class="worst">{AGG["finv_own"]}</td></tr>')
H.append(f'<tr><td class="l"><b>oracle GT pose</b></td><td class="mut">— (solves pose itself)</td><td><b>{AGG["nfi_oracle"]}</b></td><td class="mut">calibration unreliable*</td><td class="mut">—</td></tr>')
H.append(f'<tr><td class="l">control / ID pose</td><td>{AGG["ours_ctrl"]}</td><td>{AGG["nfi_ctrl"]}</td><td class="mut">—</td><td class="mut">—</td></tr>')
H.append(f'<tr><td class="l">pose-removed shape CD ↓</td><td>{CD["ours"]}</td><td>{CD["nfi"]}</td><td>{CD["eg3d"]}</td><td>{CD["finv"]}</td></tr>')
H.append('</table><p class="cap mut">NVS renders at each method’s <i>estimated</i> input pose (no alignment). Oracle hands the '
 'GT pose (fair: GT poses only, never target images). *EG3D’s pose estimate on the control set is inconsistent, so its '
 'oracle-frame calibration is unreliable — we do not report an EG3D oracle number. Chamfer is ICP-aligned (pose removed): '
 'baselines are competitive on shape, so the NVS gap is pose, not shape.</p>')

# per-group OOD/ID counts + metric summary
GRPSUM = {
 "nfi":  "<b>25 OOD SO(3) objects</b> shown below (own-pose NVS avg <b>14.3</b> → oracle-pose avg <b>17.7</b>) · "
         "<b>25 in-distribution objects</b> (NVS <b>19.85</b>, ID clips below).",
 "eg3d": "<b>25 OOD SO(3) objects</b> shown below (own-pose NVS avg <b>12.3</b>) · no reliable oracle (see table*) · "
         "in-distribution not run for EG3D (GAN-inversion has no dataset-aligned pose frame).",
 "finv": "<b>25 OOD SO(3) objects</b> shown below (own-pose NVS avg <b>12.3</b>, ≈ EG3D+PTI) · in-distribution not run.",
 "pigan":"<b>25 OOD SO(3) objects</b> shown below (blob — no calibrated NVS; assumes frontal). CARLA-domain sanity separate.",
}
H.append('<h2>Per-baseline comparison — SO(3) cars</h2>'
 '<div class="card" style="border-left:3px solid var(--good)"><p style="margin:0"><b>Two video types — read the label on each block:</b><br>'
 '• <b>NVS (scored)</b>: rendered at the <b>24 GT target cameras</b> and PSNR’d against the GT images — '
 'frame <i>i</i> is the same camera in every panel, so this <b>IS what the metric measured</b> '
 '(<code>input ‖ OURS ‖ baseline ‖ GT</code>; NFI also shows own-pose vs oracle-pose).<br>'
 '• <b>Orbit (visualization only)</b>: a free 360° turntable, <b>not</b> camera-aligned and <b>not</b> the metric — '
 'used only where no calibrated NVS exists (pi-GAN, and all RealCars).</p></div>')
for key, title, sk, desc in BASE:
    H.append(f'<h3>{title}</h3><p>{desc}</p>')
    H.append(f'<p class="mut" style="margin:2px 0 8px">Counts: {GRPSUM.get(key,"")}</p>')
    nv = vids_nvs(key) if key in ("nfi", "eg3d", "finv") else []
    if nv:
        H.append(f'<details open><summary>NVS (scored @ GT cameras) — {len(nv)} OOD objects · frame = GT view, IS the metric</summary><div class="grid">')
        H.append("\n".join(nv)); H.append('</div></details>')
    else:
        vv = vids(key, "so3")
        if vv:
            H.append(f'<details open><summary>Orbit (visualization only — no calibrated NVS) — {len(vv)} OOD objects</summary><div class="grid">')
            H.append("\n".join(vv)); H.append('</div></details>')
        if key == "pigan" and os.path.exists(f"{REP}/videos/pigan_carla/carla_samples.png"):
            H.append('<details><summary>CARLA-domain sanity — pi-GAN in its OWN domain (random samples)</summary>'
                     '<p class="mut">Random latents from the released CARLA generator render as clean cars — so the '
                     'ShapeNet/RealCars blobs are a genuine CARLA→target domain gap, not a broken model.</p>'
                     '<img src="videos/pigan_carla/carla_samples.png" style="width:100%;border-radius:8px;border:1px solid var(--line)">'
                     '<div class="grid"><div class="v"><video src="videos/pigan_carla/carla_orbit.mp4" controls loop muted playsinline></video>'
                     '<div class="cap">CARLA sample, 360° orbit (in-domain — clean)</div></div></div></details>')
    # in-distribution / control NVS videos (NFI + ours)
    iv = vids_nvs_ctrl(key) if key == "nfi" else []
    if iv:
        H.append(f'<details><summary>In-distribution (control) NVS — {len(iv)} objects · scored @ GT cameras (ours ~20.7 · NFI 19.85)</summary><div class="grid">')
        H.append("\n".join(iv)); H.append('</div></details>')

# realcars
H.append('<h2>Per-baseline comparison — RealCars (synthetic→real)</h2>'
 '<p class="mut">RealCars is single-view + LiDAR pseudo-GT — there are <b>no multi-view GT novel views</b>, so '
 '<b>there is no NVS metric here</b>; this task is scored by ICP-Chamfer (shape). The clips are '
 '<b>orbit visualizations only</b> (<code>input ‖ OURS ‖ baseline</code>, free turntable, not camera-aligned).</p>')
for key, title, sk, desc in BASE:
    vv = vids(key, "rc")
    if vv:
        H.append(f'<h3>{title}</h3><details><summary>{len(vv)} RealCars scenes</summary><div class="grid">')
        H.append("\n".join(vv)); H.append('</div></details>')

# ---- Homefield (in-distribution) section ----
H.append('<h2>Homefield — each method in its own in-distribution setting</h2>'
 '<p>Claim: <b>in the setting each baseline was built for, it works and ours is comparable</b>; the gap opens '
 'only OOD. We give every baseline — not just NFI — a real homefield: NVS where the method admits it, '
 'reconstruction fidelity + pose-removed shape where it does not.</p>')

# (a) NVS homefield — NFI vs ours (only pose-supervised method admits calibrated NVS)
H.append('<h3>(a) NVS — NFI vs ours (the pose-supervised methods)</h3>')
H.append('<table><tr><th class="l">setting (SO(3) cars)</th><th>NFI</th><th>Infer3D (ours)</th></tr>'
 f'<tr><td class="l">in-distribution / control pose</td><td>{AGG["nfi_ctrl"]}</td><td class="best">{AGG["ours_ctrl"]}</td></tr>'
 f'<tr><td class="l">OOD SO(3) pose</td><td class="worst">{AGG["nfi_own"]}</td><td class="best">{AGG["ours_so3"]}</td></tr>'
 '</table><p class="cap mut">NVS PSNR, 25 objects, shared harness. In-distribution the two are comparable and '
 'interleaved per-object (NFI occasionally edges ours — sharp NeRF vs ours’ StyleGAN prior; per-object clips '
 'in the NFI group’s “In-distribution” block). Under OOD, ours drops 1–2 dB while NFI drops ~5.5 dB.</p>')

# (b) reconstruction fidelity homefield — GAN methods, on REAL in-domain data
H.append('<h3>(b) Reconstruction fidelity — the GAN-inversion methods, on real in-domain data</h3>')
H.append('<p>The GAN-inversion baselines have <b>no calibrated NVS even in-distribution</b>: EG3D+PTI’s own-pose '
 f'control NVS is {AGG["eg3d_own"]} (≈ its OOD {AGG["eg3d_own"]}) — near-symmetric cars make unposed GAN '
 'inversion <i>gauge-limited</i>, not pose-limited (confirmed three ways: y-up frame, z-up frame, and its '
 'scattered per-object pose estimates). So their honest homefield is <b>input-view reconstruction</b> (does '
 'the inverter fit real in-domain input?) plus pose-removed shape (below):</p>')
H.append('<table><tr><th class="l">real in-domain input-view recon (PSNR ↑)</th><th>value</th></tr>'
 f'<tr><td class="l">EG3D+PTI — 25 real control cars</td><td class="best">{HF["eg3d_recon"]}</td></tr>'
 f'<tr><td class="l">FINV-SV — 25 real control cars</td><td class="best">{HF["finv_recon"]}</td></tr>'
 f'<tr><td class="l">pi-GAN — 15 real CARLA images (its training domain)</td><td class="best">{HF["pigan_real_recon"]}</td></tr>'
 '</table><p class="cap mut">On its own domain each inverter fits the real <i>input view</i> cleanly (e.g. pi-GAN 26 dB on '
 'CARLA) — vs the diffuse blobs it produces on ShapeNet/RealCars. This isolates a genuine <b>domain gap</b> '
 'from a broken model. (We deliberately invert real images, not generator samples: self-reconstruction would '
 'only prove the optimizer works on its own manifold.) <b>Caveat, in ours’ favour:</b> EG3D and FINV fit the '
 'input view but their <i>novel-view</i> 3D of real cars still degrades even in-distribution (EG3D can’t render '
 'a valid novel view at all — see §a; FINV’s orbit blurs), so this recon number is generous to them.</p>')
# EG3D + FINV input-view recon montages side by side
_montages = [("eg3d_recon_montage.png", "EG3D+PTI · 6 real control cars"),
             ("finv_recon_montage.png", "FINV-SV · 6 real control cars")]
_present = [(f, c) for f, c in _montages if os.path.exists(f"{REP}/videos/homefield_real/img/{f}")]
if _present:
    H.append('<p class="mut">Input-view reconstruction on REAL in-distribution cars (real input | recon) — '
     'the honest homefield visual for the GAN-inversion methods (both fit the input; see caveat above):</p>')
    H.append('<div class="grid">')
    for f, c in _present:
        H.append(f'  <div class="v" style="max-width:280px">'
                 f'<img src="videos/homefield_real/img/{f}" style="width:100%;border-radius:6px">'
                 f'<div class="cap">{c}</div></div>')
    H.append('</div>')
H.append('<p class="mut">pi-GAN homefield — inverting REAL CARLA images (input | novel-view orbit). Clean cars '
 'in its own domain; CARLA has no per-image camera labels, so no novel-view GT / NVS metric exists in-domain '
 '(the paper reports none either) — these orbits are qualitative:</p>')
pr = sorted(glob.glob(f"{REP}/videos/homefield_real/pigan_r*.mp4"))
if pr:
    H.append('<div class="grid">')
    for f in pr:
        H.append(f'  <div class="v"><video src="videos/homefield_real/{os.path.basename(f)}" controls loop muted playsinline preload="metadata"></video>'
                 f'<div class="cap">real CARLA — input | novel views</div></div>')
    H.append('</div>')

# ---- Chamfer (pose-removed shape) section ----
H.append('<h2>Pose-removed shape — ICP-aligned Chamfer ↓</h2>'
 '<p>ICP alignment removes global pose, so this measures <b>shape only</b> — the control that shows the SO(3) '
 'NVS gap is a <i>pose</i> failure, not a shape failure: baselines are competitive here.</p>')
H.append('<table><tr><th class="l">SO(3) cars (cd_sym)</th><th>ours</th><th>NFI</th><th>EG3D+PTI</th><th>pi-GAN</th><th>FINV-SV</th></tr>'
 f'<tr><td class="l">in-distribution (homefield)</td><td>{CD_CTRL["ours"]}</td><td class="best">{CD_CTRL["nfi"]}</td><td>{CD_CTRL["eg3d"]}</td><td>{CD_CTRL["pigan"]}</td><td>{CD_CTRL["finv"]}</td></tr>'
 f'<tr><td class="l">OOD SO(3)</td><td>{CD["ours"]}</td><td class="best">{CD["nfi"]}</td><td>{CD["eg3d"]}</td><td>{CD["pigan"]}</td><td>{CD["finv"]}</td></tr>'
 '</table><p class="cap mut">Lower = better shape; all 5 methods on the same 25 GT cars. Two things at once: '
 '(i) <b>in each method’s homefield the shape is good and ours is comparable</b>; (ii) every method degrades '
 'only <b>gracefully</b> from in-distribution → OOD on shape (e.g. EG3D 0.0086→0.0127), while its NVS collapses '
 '12–14 dB — so the SO(3) NVS gap is <b>pose recovery, not shape</b>. Baselines match or beat ours on pose-removed '
 'shape (NFI’s clean SDF mesh even beats our Gaussian-centre cloud).</p>')
H.append('<table><tr><th class="l">RealCars (cd pred→gt, m²)</th><th>pi-GAN*</th><th>ours</th><th>EG3D*</th><th>FINV-SV*</th><th>Splatter</th><th>SF3D</th><th>LGM</th><th>NFI</th></tr>'
 '<tr><td class="l">Chamfer ↓</td><td>0.088</td><td>0.131</td><td>0.154</td><td>0.169</td><td>0.219</td><td>0.242</td><td>0.405</td><td class="worst">0.491</td></tr>'
 '</table><p class="cap mut"><b>*Fairness caveat:</b> the RealCars align step removes pose+scale and rewards emitting a '
 'complete generic car, so full-mesh GAN priors (pi-GAN, EG3D, FINV) score low by representation, not by recovering '
 'the correct car. The <b>discriminative</b> result is NFI’s genuine collapse (0.491, worse than feed-forward Splatter '
 '0.219): its pixel/latent inversion locks onto real texture the synthetic prior cannot represent — the same '
 'representation family as ours, so this is real geometry breakage. Among comparable single-view methods ours (0.131) '
 'beats Splatter (0.219) &gt; LGM (0.405).</p>')

H.append('<p class="mut" style="margin-top:30px">Isolated rebuttal sandbox. Scripts in <code>rebuttal/baselines/</code>, '
 'numbers in <code>rebuttal/results/</code>, design log in <code>rebuttal/JOURNAL.md</code>. '
 'Per-object scores: <code>rebuttal/results/per_object_scores.json</code>.</p>')
H.append('</div></body></html>')
open(f"{REP}/index.html", "w").write("\n".join(H))
print("wrote report/index.html")
for key, *_ in [(b[0],) for b in BASE]:
    print(f"  {key}: so3={len(vids(key,'so3'))} rc={len(vids(key,'rc'))}")
