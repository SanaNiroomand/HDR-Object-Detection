"""Box geometry + resolution analysis, correctly normalized per-image."""
import os, json, collections
import numpy as np

ROOT = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database"
ANN = os.path.join(ROOT, "annotations")
OUT = os.path.dirname(os.path.abspath(__file__))

splits = {}
for name, fn in [("train", "instances_train2020.json"), ("test", "instances_test2020.json")]:
    with open(os.path.join(ANN, fn), encoding="utf-8") as f:
        splits[name] = json.load(f)
cats = {c["id"]: c["name"] for c in splits["train"]["categories"]}


def sec(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


# --------------------------------------------------- resolution per split
sec("A. RESOLUTION BY SPLIT")
allres = collections.Counter()
per_split_res = {}
for name, d in splits.items():
    c = collections.Counter((im["width"], im["height"]) for im in d["images"])
    per_split_res[name] = c
    allres.update(c)
print(f"{'resolution':<16}{'MP':>7}{'orient':>10}{'train':>8}{'test':>8}{'total':>8}")
print("-" * 60)
for (w, h), n in allres.most_common():
    o = "portrait" if h > w else "landscape"
    print(f"{f'{w}x{h}':<16}{w*h/1e6:>7.1f}{o:>10}{per_split_res['train'].get((w,h),0):>8}"
          f"{per_split_res['test'].get((w,h),0):>8}{n:>8}")
print("-" * 60)
print(f"{len(allres)} distinct resolutions")
port = sum(n for (w, h), n in allres.items() if h > w)
print(f"portrait images: {port} ({port/4080:.1%})")
mps = np.array([w * h / 1e6 for (w, h), n in allres.items() for _ in range(n)])
print(f"megapixels: min={mps.min():.1f}  median={np.median(mps):.1f}  max={mps.max():.1f}  "
      f"ratio max/min={mps.max()/mps.min():.1f}x")
ars = sorted({round(w / h, 3) for (w, h) in allres})
print(f"aspect ratios present: {ars}")

# --------------------------------------------------- boxes, per-image normalized
sec("B. BOX GEOMETRY (normalized by each image's OWN resolution)")
recs = []
for name, d in splits.items():
    dims = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    for a in d["annotations"]:
        W, H = dims[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        recs.append((name, cats[a["category_id"]], x, y, bw, bh, W, H))

bw = np.array([r[4] for r in recs], float)
bh = np.array([r[5] for r in recs], float)
X = np.array([r[2] for r in recs], float)
Y = np.array([r[3] for r in recs], float)
W = np.array([r[6] for r in recs], float)
H = np.array([r[7] for r in recs], float)

area_frac = (bw * bh) / (W * H)
# relative size = sqrt of area fraction -> "object occupies this fraction of image side"
rel = np.sqrt(area_frac)


def q(a, label, fmt="%.2f"):
    ps = np.percentile(a, [0, 1, 25, 50, 75, 99, 100])
    print(f"   {label:<16}" + "".join((fmt % v).rjust(10) for v in ps) + (f"{fmt % a.mean():>11}"))


print(f"   {'':<16}{'min':>10}{'p1':>10}{'p25':>10}{'p50':>10}{'p75':>10}{'p99':>10}{'max':>10}{'mean':>11}")
q(bw, "width px", "%.0f")
q(bh, "height px", "%.0f")
q(area_frac * 100, "% of image", "%.3f")
q(rel * 100, "rel. size %", "%.2f")
q(bw / np.maximum(bh, 1e-9), "aspect w/h", "%.2f")

sec("C. ANNOTATION VALIDITY (per-image bounds)")
x2, y2 = X + bw, Y + bh
oob_l = X < 0
oob_t = Y < 0
oob_r = x2 > W
oob_b = y2 > H
any_oob = oob_l | oob_t | oob_r | oob_b
n = len(recs)
print(f"total boxes: {n}")
print(f"  x < 0            : {int(oob_l.sum())}")
print(f"  y < 0            : {int(oob_t.sum())}")
print(f"  x+w > width      : {int(oob_r.sum())}")
print(f"  y+h > height     : {int(oob_b.sum())}")
print(f"  ANY out of bounds: {int(any_oob.sum())} ({any_oob.mean():.2%})")
if any_oob.sum():
    ex = np.where(oob_r | oob_b)[0][:5]
    print("  overflow magnitude (px beyond edge):")
    ovr = np.maximum(0, x2 - W); ovb = np.maximum(0, y2 - H)
    ov = np.maximum(ovr, ovb)[any_oob]
    print(f"     max={ov.max():.0f}  median={np.median(ov):.0f}  "
          f"boxes overflowing by >1px: {int((ov>1).sum())}  by >10px: {int((ov>10).sum())}")
    print("  examples:")
    for i in ex[:5]:
        r = recs[i]
        print(f"     {r[0]:<6}{r[1]:<12} bbox=({r[2]},{r[3]},{r[4]},{r[5]}) -> x2={r[2]+r[4]} y2={r[3]+r[5]}  img={r[6]}x{r[7]}")

print(f"\n  boxes larger than their image: {int((area_frac > 1.0).sum())}")
print(f"  boxes with area_frac > 0.99  : {int((area_frac > 0.99).sum())}")
print(f"  zero/negative w or h         : {int(((bw <= 0) | (bh <= 0)).sum())}")
tiny = (bw < 8) | (bh < 8)
print(f"  very thin boxes (<8px side)  : {int(tiny.sum())}")

# area field vs w*h
sec("D. COCO 'area' FIELD vs w*h  (and segmentation sanity)")
for name, d in splits.items():
    a_field = np.array([a["area"] for a in d["annotations"]], float)
    a_calc = np.array([a["bbox"][2] * a["bbox"][3] for a in d["annotations"]], float)
    rel_err = np.abs(a_field - a_calc) / np.maximum(a_calc, 1)
    print(f"{name}: area==w*h for {int((rel_err<1e-6).sum())}/{len(a_field)} annotations "
          f"(max rel. diff {rel_err.max():.3f})")
    segs = [a["segmentation"] for a in d["annotations"]]
    npts = collections.Counter(len(s[0]) // 2 if s and s[0] else 0 for s in segs)
    print(f"   segmentation polygon vertex counts: {dict(npts)}  -> "
          f"{'all are axis-aligned box rectangles (no real masks)' if set(npts)=={4} else 'mixed'}")

# --------------------------------------------------- size buckets, scale-aware
sec("E. OBJECT SCALE  (relative, comparable across resolutions)")
buckets = [(0, 0.01, "tiny   (<1% of side)"), (0.01, 0.05, "small  (1-5%)"),
           (0.05, 0.15, "medium (5-15%)"), (0.15, 0.4, "large  (15-40%)"),
           (0.4, 10, "huge   (>40%)")]
for lo, hi, lab in buckets:
    m = (rel >= lo) & (rel < hi)
    print(f"   {lab:<24}{int(m.sum()):>8}  {m.mean():>7.1%}")

print("\n   median relative size by class (top classes):")
byc = collections.defaultdict(list)
for i, r in enumerate(recs):
    byc[r[1]].append(rel[i])
for cl in sorted(byc, key=lambda c: -len(byc[c]))[:10]:
    v = np.array(byc[cl])
    print(f"     {cl:<14}n={len(v):<7} median={np.median(v)*100:>6.2f}%  p25={np.percentile(v,25)*100:>6.2f}%  p75={np.percentile(v,75)*100:>6.2f}%")

# --------------------------------------------------- class co-occurrence / per image
sec("F. IMAGES PER CLASS (how many images contain >=1 instance)")
img_has = collections.defaultdict(set)
for name, d in splits.items():
    for a in d["annotations"]:
        img_has[cats[a["category_id"]]].add((name, a["image_id"]))
tot_imgs = 4080
print(f"{'class':<14}{'images':>8}{'% imgs':>9}{'inst/img':>10}")
print("-" * 42)
inst = collections.Counter()
for name, d in splits.items():
    for a in d["annotations"]:
        inst[cats[a["category_id"]]] += 1
for cl in sorted(img_has, key=lambda c: -len(img_has[c])):
    k = len(img_has[cl])
    print(f"{cl:<14}{k:>8}{k/tot_imgs:>8.1%}{inst[cl]/k:>10.2f}")
