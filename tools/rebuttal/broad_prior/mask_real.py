"""SAM3-mask + center each real photo into a 128px white-bg object-centric input
(what the Objaverse lifter expects). Writes real_images/proc/<name>.png + overlay for QA."""
from infer3d import config as _cfg
_EXT = _cfg.EXTERN_ROOT          # cluster-independent root (see infer3d/config.py)

import glob, json, os, sys
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

R = f"{_EXT}/splatter-image-rebuttal/rebuttal/broad_prior/real_images"
os.makedirs(f"{R}/proc", exist_ok=True)
man = json.load(open(f"{R}/manifest.json"))
sam = P.SAM3()
ok = []
for f in sorted(glob.glob(f"{R}/*.jpg") + glob.glob(f"{R}/*.png") + glob.glob(f"{R}/*.jpeg")):
    name = os.path.splitext(os.path.basename(f))[0]
    ent = man.get(os.path.basename(f), {})
    noun = ent.get("object_noun") if isinstance(ent, dict) else None
    if noun is None:
        noun = name.rsplit("_", 1)[0].replace("cowboyhat", "cowboy hat").replace("firehydrant", "fire hydrant")
    img = np.asarray(Image.open(f).convert("RGB"))
    mk = sam.mask(img, noun, threshold=0.3)
    if mk is None or mk.mean() < 0.005:
        print(f"  {name}: SAM3 FAIL ({noun})"); continue
    t = torch.from_numpy(img).float().permute(2, 0, 1).to(P.DEV) / 255.
    u = P.center_on_white(t, mk, out=128, margin=0.12)
    Image.fromarray((u.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(f"{R}/proc/{name}.png")
    ok.append(name)
    print(f"  {name}: ok ({noun}) fg={100*mk.mean():.0f}%")
json.dump(ok, open(f"{R}/proc/ok.json", "w"))
print(f"\n{len(ok)} masked inputs -> {R}/proc/")
