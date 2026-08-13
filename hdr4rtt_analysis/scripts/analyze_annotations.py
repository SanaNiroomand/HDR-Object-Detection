"""Structural analysis of the HDR4RTT dataset: COCO annotations + YOLO-style label files."""
import os, json, glob, collections
import numpy as np

ROOT = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database"
ANN = os.path.join(ROOT, "annotations")
OUT = os.path.dirname(os.path.abspath(__file__))


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


splits = {}
for name, fn in [("train", "instances_train2020.json"), ("test", "instances_test2020.json")]:
    with open(os.path.join(ANN, fn), "r", encoding="utf-8") as f:
        splits[name] = json.load(f)

# ---------------------------------------------------------------- top level
sec("1. COCO FILE STRUCTURE")
for name, d in splits.items():
    print(f"{name:6s} top-level keys: {list(d.keys())}")
    print(f"       images={len(d['images'])}  annotations={len(d['annotations'])}  categories={len(d['categories'])}")
    print(f"       sample image record: {d['images'][0]}")
    print(f"       sample annot record: {d['annotations'][0]}")

cats = {c["id"]: c["name"] for c in splits["train"]["categories"]}
cats_test = {c["id"]: c["name"] for c in splits["test"]["categories"]}
print(f"\ncategory sets identical across splits: {cats == cats_test}")
print(f"{len(cats)} categories: {list(cats.values())}")

# ---------------------------------------------------------------- splits
sec("2. SPLITS & IMAGE INVENTORY")
img_files = sorted(glob.glob(os.path.join(ROOT, "images", "*.exr")))
lbl_files = sorted(glob.glob(os.path.join(ROOT, "labels", "*.txt")))
img_stems = {os.path.splitext(os.path.basename(p))[0] for p in img_files}
lbl_stems = {os.path.splitext(os.path.basename(p))[0] for p in lbl_files}
print(f"images/ .exr files : {len(img_stems)}")
print(f"labels/ .txt files : {len(lbl_stems)}")
print(f"images without label: {len(img_stems - lbl_stems)}")
print(f"labels without image: {len(lbl_stems - img_stems)}")

split_stems = {}
for name, d in splits.items():
    stems = {os.path.splitext(os.path.basename(im["file_name"]))[0] for im in d["images"]}
    split_stems[name] = stems
    print(f"\n{name}: {len(d['images'])} images  ({len(stems)} unique stems)")
    print(f"   present in images/ : {len(stems & img_stems)}   missing: {len(stems - img_stems)}")
    if stems - img_stems:
        print(f"   e.g. missing: {sorted(stems - img_stems)[:5]}")

ov = split_stems["train"] & split_stems["test"]
print(f"\ntrain/test overlap: {len(ov)} images" + (f"  e.g. {sorted(ov)[:5]}" if ov else "  (clean)"))
covered = split_stems["train"] | split_stems["test"]
print(f"union of splits: {len(covered)}   images/ not in any split: {len(img_stems - covered)}")
tot = len(split_stems['train']) + len(split_stems['test'])
print(f"train:test ratio = {len(split_stems['train'])/tot:.1%} : {len(split_stems['test'])/tot:.1%}")

# ---------------------------------------------------------------- resolution
sec("3. IMAGE RESOLUTION (from COCO records)")
res = collections.Counter()
for name, d in splits.items():
    for im in d["images"]:
        res[(im["width"], im["height"])] += 1
for (w, h), n in res.most_common():
    print(f"   {w} x {h}   {n} images")

# ---------------------------------------------------------------- classes
sec("4. CLASS DISTRIBUTION (COCO annotations)")
per_split = {}
for name, d in splits.items():
    c = collections.Counter(cats[a["category_id"]] for a in d["annotations"])
    per_split[name] = c

allc = collections.Counter()
for c in per_split.values():
    allc.update(c)

print(f"{'class':<14}{'train':>9}{'test':>9}{'total':>9}{'% total':>9}{'test%':>8}")
print("-" * 58)
gtot = sum(allc.values())
for cl, n in allc.most_common():
    tr, te = per_split["train"].get(cl, 0), per_split["test"].get(cl, 0)
    tepct = te / n * 100 if n else 0
    print(f"{cl:<14}{tr:>9}{te:>9}{n:>9}{n/gtot*100:>8.2f}%{tepct:>7.1f}%")
print("-" * 58)
print(f"{'TOTAL':<14}{sum(per_split['train'].values()):>9}{sum(per_split['test'].values()):>9}{gtot:>9}")

unused = [c for c in cats.values() if c not in allc]
print(f"\ncategories with zero annotations: {unused if unused else 'none'}")
print(f"imbalance ratio (max/min class): {max(allc.values())/min(allc.values()):.1f}x")

# ---------------------------------------------------------------- boxes
sec("5. BOUNDING BOX GEOMETRY (COCO, all splits)")
W = H = None
for (w, h), n in res.most_common(1):
    W, H = w, h

rows = []
for name, d in splits.items():
    for a in d["annotations"]:
        x, y, bw, bh = a["bbox"]
        rows.append((name, cats[a["category_id"]], x, y, bw, bh, a.get("area", bw * bh), a.get("iscrowd", 0)))

bw = np.array([r[4] for r in rows], dtype=float)
bh = np.array([r[5] for r in rows], dtype=float)
area = bw * bh
frac = area / (W * H)


def q(a, name, fmt="%.1f"):
    ps = np.percentile(a, [0, 1, 25, 50, 75, 99, 100])
    print(f"   {name:<12}" + "  ".join(fmt % v for v in ps) + f"   mean={fmt % a.mean()}")


print(f"   {'':<12}{'min':>8}{'p1':>10}{'p25':>10}{'p50':>10}{'p75':>10}{'p99':>10}{'max':>10}")
q(bw, "width px")
q(bh, "height px")
q(np.sqrt(area), "sqrt(area)")
q(frac * 100, "% of image", "%.3f")
q(bw / np.maximum(bh, 1e-9), "aspect w/h", "%.2f")

# COCO-style size buckets (on 1920x1080; COCO thresholds are 32^2 / 96^2 on ~640px imgs,
# report both raw COCO thresholds and area-fraction buckets)
small = int((area < 32 ** 2).sum())
med = int(((area >= 32 ** 2) & (area < 96 ** 2)).sum())
large = int((area >= 96 ** 2).sum())
n = len(area)
print(f"\n   COCO size buckets (absolute px):  small<32^2: {small} ({small/n:.1%})   "
      f"medium: {med} ({med/n:.1%})   large>=96^2: {large} ({large/n:.1%})")

print(f"\n   crowd annotations (iscrowd=1): {sum(r[7] for r in rows)}")
degen = int(((bw <= 0) | (bh <= 0)).sum())
print(f"   degenerate boxes (w or h <= 0): {degen}")
oob = sum(1 for r in rows if r[2] < 0 or r[3] < 0 or r[2] + r[4] > W + 1 or r[3] + r[5] > H + 1)
print(f"   boxes extending outside image bounds: {oob}")

# ---------------------------------------------------------------- per image
sec("6. OBJECTS PER IMAGE")
for name, d in splits.items():
    per_img = collections.Counter(a["image_id"] for a in d["annotations"])
    counts = np.array([per_img.get(im["id"], 0) for im in d["images"]])
    print(f"{name:6s} mean={counts.mean():.2f}  median={np.median(counts):.0f}  "
          f"min={counts.min()}  max={counts.max()}  empty images={int((counts==0).sum())}")

# ---------------------------------------------------------------- labels/
sec("7. labels/ TXT FILES  (format check vs COCO)")
lbl_classes = collections.Counter()
nbox = 0
bad = []
lbl_per_file = {}
for p in lbl_files:
    stem = os.path.splitext(os.path.basename(p))[0]
    k = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                bad.append((stem, line))
                continue
            lbl_classes[parts[0]] += 1
            nbox += 1
            k += 1
    lbl_per_file[stem] = k

print(f"total boxes in labels/: {nbox}   (COCO total: {gtot})   match: {nbox == gtot}")
print(f"malformed lines: {len(bad)}")
print(f"empty label files: {sum(1 for v in lbl_per_file.values() if v == 0)}")
print(f"\nclass names in labels/ ({len(lbl_classes)}): ")
for cl, k in lbl_classes.most_common():
    print(f"   {cl:<14}{k:>8}   (COCO: {allc.get(cl, 0)})")

names_match = set(lbl_classes) == set(c for c in allc)
print(f"\nlabel class names == COCO class names with annotations: {names_match}")

# verify coordinate convention on one file
sample = "hdr_00000"
coco_boxes = []
for name, d in splits.items():
    id_by_stem = {os.path.splitext(os.path.basename(im["file_name"]))[0]: im["id"] for im in d["images"]}
    if sample in id_by_stem:
        iid = id_by_stem[sample]
        coco_boxes = [(cats[a["category_id"]], a["bbox"]) for a in d["annotations"] if a["image_id"] == iid]
        print(f"\n{sample} found in split '{name}'")
        break
print(f"COCO bboxes (x,y,w,h) for {sample}:")
for c, b in coco_boxes:
    print(f"   {c:<14}{[round(v,1) for v in b]}   -> xyxy {[round(b[0],1), round(b[1],1), round(b[0]+b[2],1), round(b[1]+b[3],1)]}")
print(f"labels/{sample}.txt:")
with open(os.path.join(ROOT, "labels", sample + ".txt")) as f:
    for line in f:
        print("   " + line.rstrip())

json.dump(
    {
        "n_images": len(img_stems),
        "splits": {k: len(v) for k, v in split_stems.items()},
        "overlap": len(ov),
        "categories": list(cats.values()),
        "class_counts_train": dict(per_split["train"]),
        "class_counts_test": dict(per_split["test"]),
        "total_boxes": gtot,
        "resolution": {f"{w}x{h}": n for (w, h), n in res.most_common()},
    },
    open(os.path.join(OUT, "annotation_summary.json"), "w"),
    indent=2,
)
print("\n[saved annotation_summary.json]")
