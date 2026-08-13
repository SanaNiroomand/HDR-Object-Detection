#!/usr/bin/env python3
"""
build_rod_annotations.py

Turns HDR4RTT's COCO annotations into the exact format RAOD's COCORawDataset
expects, and builds a leakage-free train/test split.

RAOD's annotation convention, taken from scripts/preprocess_anno.py (their own
script, not a guess):

    image['height'] = 1280          # hardcoded
    image['width']  = 1280          # hardcoded
    min_x = int(min_x * (1280/2880))    # anisotropic: x by width ratio
    min_y = int(min_y * (1280/1856))    #              y by height ratio
    file_name = <stem>.npy.gz
    classname_to_id = {'Pedestrian':1,'Car':2,'Cyclist':3,'Tram':4,'Truck':5}
    boxes with w<16 or h<16 (in 1280-space) are dropped

The anisotropic box scaling exists because preprocess_raw.py squashes every
image to 1280x1280 with cv2.resize WITHOUT preserving aspect ratio. Boxes must
be squashed the same way or they will not line up. Then coco_raw.py computes
r = min(1280/1280, 1280/1280) = 1.0 and leaves both alone. We reproduce that
contract exactly, per-image, using each HDR4RTT image's own dimensions.

Class mapping. RAOD's head has 5 outputs (verified from all three checkpoints in
pre-trained/: head.cls_preds.*.weight has shape (5, 64, 1, 1)). The 6-class line
in preprocess_anno.py is commented out -- Tricycle was dropped. Of HDR4RTT's 20
VOC classes only two map honestly onto ROD's traffic classes:

    person -> Pedestrian (id 1)
    car    -> Car        (id 2)

That keeps 21,399 of 32,964 boxes (64.9%) across 3,818 of 4,080 images.
Deliberately NOT mapped, and why:
    bicycle -> Cyclist   'Cyclist' is a person riding, not the bicycle object.
    bus/truck            HDR4RTT has 7 bus instances total; not worth the noise.
    train   -> Tram      8 instances, and a railway train is not a city tram.
All 5 ROD categories are still declared in the output json so that
coco_raw.py's `class_ids.index(category_id)` yields the same head index the
model was trained with (sorted([1,2,3,4,5]) -> Pedestrian=0, Car=1).

Splits produced:
    original  -- HDR4RTT's own train/test assignment, kept as-is. Comparable
                 with earlier numbers, but leaky: 100% of S3's test frames have
                 a training frame within +-3 of them in the source video.
    seqsafe   -- S3's video frames are grouped into contiguous blocks and whole
                 blocks are assigned to one side, with a guard band of dropped
                 frames at every block boundary so no test frame is temporally
                 adjacent to a train frame. S1 and S2 keep their original
                 assignment (S2 is independent bracketed stills; S1 has no
                 metadata to detect sequences with -- see caveat below).

CAVEAT on S1: 1,289 images carry no EXR metadata at all, so there is no frame
numbering to group by. If S1 is also video-derived, 'seqsafe' still contains
leakage from that source. Detecting it would need image-similarity hashing.
"""
import os
import json
import argparse
import collections
import numpy as np

ROD_CLASSES = {"Pedestrian": 1, "Car": 2, "Cyclist": 3, "Tram": 4, "Truck": 5}
HDR4RTT_TO_ROD = {"person": "Pedestrian", "car": "Car"}
TARGET = 1280


def load_hdr4rtt(root):
    """Returns per-stem records: dims, split, boxes (original pixel xywh + class)."""
    recs = {}
    for split, fn in [("train", "instances_train2020.json"),
                      ("test", "instances_test2020.json")]:
        with open(os.path.join(root, "annotations", fn), encoding="utf-8") as f:
            d = json.load(f)
        cats = {c["id"]: c["name"] for c in d["categories"]}
        byid = {}
        for im in d["images"]:
            stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
            byid[im["id"]] = stem
            recs[stem] = {"stem": stem, "w": im["width"], "h": im["height"],
                          "split": split, "boxes": []}
        for a in d["annotations"]:
            name = cats[a["category_id"]]
            if name not in HDR4RTT_TO_ROD:
                continue
            recs[byid[a["image_id"]]]["boxes"].append((a["bbox"], HDR4RTT_TO_ROD[name]))
    return recs


def seqsafe_split(recs, sources, frame_nums, n_blocks, guard, test_frac, seed):
    """Assign whole contiguous runs of S3 video frames to one side of the split.

    S3 frames are numbered 1..1499 in the original video. We cut that range into
    n_blocks contiguous blocks, assign blocks to test until test_frac is met, and
    then DROP `guard` frames on each side of every train/test boundary so that no
    surviving test frame sits within `guard` frames of a surviving train frame.
    Dropped frames are excluded from both splits rather than silently reassigned.
    """
    s3 = sorted([s for s in recs if sources.get(s) == "S3" and s in frame_nums],
                key=lambda s: frame_nums[s])
    if not s3:
        return {}, set()
    nums = np.array([frame_nums[s] for s in s3])
    lo, hi = nums.min(), nums.max()
    edges = np.linspace(lo, hi + 1, n_blocks + 1)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_blocks)
    block_split, n_test, target = {}, 0, test_frac * len(s3)
    for b in order:
        cnt = int(((nums >= edges[b]) & (nums < edges[b + 1])).sum())
        if n_test < target:
            block_split[int(b)] = "test"
            n_test += cnt
        else:
            block_split[int(b)] = "train"

    block_of = np.digitize(nums, edges) - 1
    block_of = np.clip(block_of, 0, n_blocks - 1)
    assign = {s: block_split[int(b)] for s, b in zip(s3, block_of)}

    # Guard band: drop frames whose number is within `guard` of any frame
    # assigned to the other side.
    num_to_stem = {frame_nums[s]: s for s in s3}
    dropped = set()
    for s in s3:
        n = frame_nums[s]
        mine = assign[s]
        for d in range(-guard, guard + 1):
            other = num_to_stem.get(n + d)
            if other is not None and assign[other] != mine:
                dropped.add(s)
                break
    return assign, dropped


def to_coco(recs, stems, min_box):
    """Emit RAOD-format COCO. Boxes squashed anisotropically into 1280x1280."""
    images, annotations = [], []
    ann_id = 0
    n_small = 0
    for img_id, stem in enumerate(sorted(stems)):
        r = recs[stem]
        images.append({"height": TARGET, "width": TARGET, "id": img_id,
                       "file_name": stem + ".npy.gz"})
        sx, sy = TARGET / r["w"], TARGET / r["h"]
        for (bx, by, bw, bh), cname in r["boxes"]:
            x, y = bx * sx, by * sy
            w, h = bw * sx, bh * sy
            # clamp into frame (47 boxes in HDR4RTT overhang the image edge)
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(TARGET), x + w), min(float(TARGET), y + h)
            w, h = x1 - x0, y1 - y0
            if w < min_box or h < min_box:
                n_small += 1
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": ROD_CLASSES[cname],
                "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2), "iscrowd": 0,
                "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
            })
            ann_id += 1
    cats = [{"id": v, "name": k} for k, v in sorted(ROD_CLASSES.items(), key=lambda kv: kv[1])]
    return {"info": "", "license": [""], "images": images,
            "annotations": annotations, "categories": cats}, n_small


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database")
    p.add_argument("--stats_csv", default=r"D:\Codes\HDR\Sana\hdr4rtt_analysis\hdr_stats_sources.csv",
                   help="from the dataset analysis; supplies source group + original frame number")
    p.add_argument("--out_dir", default=r"D:\Data\HDR\hdr4rtt_rod\annotations")
    p.add_argument("--min_box", type=float, default=16.0,
                   help="drop boxes smaller than this in 1280-space (RAOD uses 16)")
    p.add_argument("--n_blocks", type=int, default=24, help="contiguous S3 video blocks")
    p.add_argument("--guard", type=int, default=3,
                   help="drop frames within this many frames of a split boundary")
    p.add_argument("--test_frac", type=float, default=0.21)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recs = load_hdr4rtt(args.root)
    print(f"loaded {len(recs)} images from HDR4RTT annotations")

    # source group + original video frame number, from the analysis CSV
    import csv
    sources, frame_nums = {}, {}
    with open(args.stats_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = row["stem"]
            sources[st] = row["source"].split(":")[0]
            fn = (row.get("FILE_NAME") or "").strip()
            if sources[st] == "S3" and fn:
                digits = "".join(ch for ch in os.path.basename(fn) if ch.isdigit())
                if digits:
                    frame_nums[st] = int(digits)
    print(f"source groups: {collections.Counter(sources.values())}")
    print(f"S3 frames with a number: {len(frame_nums)}")

    keep = {s for s, r in recs.items() if r["boxes"]}
    print(f"images with >=1 person/car box: {len(keep)}  (dropped {len(recs)-len(keep)} empty)")

    # ---- original split
    splits = {"original": {"train": {s for s in keep if recs[s]["split"] == "train"},
                           "test": {s for s in keep if recs[s]["split"] == "test"}}}

    # ---- sequence-safe split
    assign, dropped = seqsafe_split(recs, sources, frame_nums,
                                    args.n_blocks, args.guard, args.test_frac, args.seed)
    tr, te = set(), set()
    for s in keep:
        if sources.get(s) == "S3" and s in assign:
            if s in dropped:
                continue
            (tr if assign[s] == "train" else te).add(s)
        else:
            (tr if recs[s]["split"] == "train" else te).add(s)
    splits["seqsafe"] = {"train": tr, "test": te}
    print(f"\nS3 guard band dropped {len(dropped)} frames at block boundaries")

    # ---- verify the seqsafe split really is leakage-free
    tr_nums = {frame_nums[s] for s in tr if s in frame_nums}
    te_nums = {frame_nums[s] for s in te if s in frame_nums}
    viol = sum(1 for n in te_nums for d in range(-args.guard, args.guard + 1)
               if d and (n + d) in tr_nums)
    print(f"seqsafe leakage check: test frames within +-{args.guard} of a train frame = {viol}")
    orig_tr = {frame_nums[s] for s in splits['original']['train'] if s in frame_nums}
    orig_te = {frame_nums[s] for s in splits['original']['test'] if s in frame_nums}
    oviol = sum(1 for n in orig_te if any((n + d) in orig_tr for d in (-1, 1)))
    print(f"original split for comparison: {oviol}/{len(orig_te)} test frames "
          f"have a train frame at +-1")

    # ---- write
    print()
    manifest = {}
    for split_name, parts in splits.items():
        for part, stems in parts.items():
            coco, n_small = to_coco(recs, stems, args.min_box)
            out = os.path.join(args.out_dir, f"hdr4rtt_rod_{split_name}_{part}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(coco, f)
            nb = len(coco["annotations"])
            per_c = collections.Counter(a["category_id"] for a in coco["annotations"])
            print(f"{split_name:9} {part:5}  {len(stems):5} imgs  {nb:6} boxes  "
                  f"(Pedestrian {per_c[1]}, Car {per_c[2]})  dropped<{args.min_box:g}px: {n_small}")
            manifest[f"{split_name}_{part}"] = {"images": len(stems), "boxes": nb,
                                                "json": os.path.basename(out)}

    # file list for the converter: every image referenced by any split
    allstems = sorted(set().union(*[p for s in splits.values() for p in s.values()]))
    lst = os.path.join(args.out_dir, "images_to_convert.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write("\n".join(allstems))
    print(f"\n{len(allstems)} distinct images to convert -> {lst}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"splits": manifest, "min_box": args.min_box, "guard": args.guard,
                   "n_blocks": args.n_blocks, "class_map": HDR4RTT_TO_ROD,
                   "rod_classes": ROD_CLASSES, "n_images": len(allstems)}, f, indent=1)
    print(f"wrote {args.out_dir}\\manifest.json")


if __name__ == "__main__":
    main()
