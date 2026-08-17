#!/usr/bin/env python3
"""
build_voc20_annotations.py

COCO annotations keeping ALL 20 Pascal VOC classes, for the fixed-detector
experiment. The earlier build_rod_annotations.py kept only person and car,
because RAOD's pretrained head has 5 traffic classes. Here the detector is
trained from a COCO-pretrained backbone with a fresh head, so there is no reason
to discard 35% of the boxes -- and the thesis tables this experiment is meant to
sit beside report over 20 classes.

Same geometry as everything else: boxes squashed anisotropically into 1280x1280
to match the non-aspect-preserving resize the converters apply.

Two classes carry ZERO test instances (cow, cat). COCO evaluation excludes
classes with no ground truth from the average, so they simply do not contribute;
they are still declared so category ids stay stable across splits.
"""
import os
import json
import argparse
import collections

import pandas as pd

TARGET = 1280


def load(root):
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
            recs[stem] = {"w": im["width"], "h": im["height"], "split": split, "boxes": []}
        for a in d["annotations"]:
            recs[byid[a["image_id"]]]["boxes"].append((a["bbox"], cats[a["category_id"]]))
    names = sorted({n for r in recs.values() for _, n in r["boxes"]})
    return recs, names


def to_coco(recs, stems, name_to_id, min_box):
    images, anns, ann_id, dropped = [], [], 0, 0
    for img_id, stem in enumerate(sorted(stems)):
        r = recs[stem]
        images.append({"height": TARGET, "width": TARGET, "id": img_id,
                       "file_name": stem + ".png"})
        sx, sy = TARGET / r["w"], TARGET / r["h"]
        for (bx, by, bw, bh), cname in r["boxes"]:
            x0, y0 = max(0.0, bx * sx), max(0.0, by * sy)
            x1 = min(float(TARGET), (bx + bw) * sx)
            y1 = min(float(TARGET), (by + bh) * sy)
            w, h = x1 - x0, y1 - y0
            if w < min_box or h < min_box:
                dropped += 1
                continue
            anns.append({"id": ann_id, "image_id": img_id,
                         "category_id": name_to_id[cname],
                         "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0})
            ann_id += 1
    cats = [{"id": v, "name": k} for k, v in sorted(name_to_id.items(), key=lambda kv: kv[1])]
    return {"info": "", "license": [""], "images": images,
            "annotations": anns, "categories": cats}, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database")
    p.add_argument("--stats_csv", default=r"D:\Codes\HDR\Sana\hdr4rtt_analysis\hdr_stats_sources.csv")
    p.add_argument("--out_dir", default=r"D:\Data\HDR\hdr4rtt_voc20\annotations")
    p.add_argument("--min_box", type=float, default=8.0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    recs, names = load(args.root)
    name_to_id = {n: i + 1 for i, n in enumerate(names)}
    print(f"{len(recs)} images, {len(names)} classes: {names}\n")

    # reuse the leakage-free split membership decided by build_rod_annotations.py
    seq_dir = r"D:\Data\HDR\hdr4rtt_rod\annotations"
    members = {}
    for part in ["train", "test"]:
        with open(os.path.join(seq_dir, f"hdr4rtt_rod_seqsafe_{part}.json"), encoding="utf-8") as f:
            d = json.load(f)
        members[part] = {os.path.splitext(os.path.splitext(im["file_name"])[0])[0]
                         for im in d["images"]}
    # that file dropped images with no person/car box; re-add them here, assigning
    # each to whichever side its source-group counterpart went to
    assigned = members["train"] | members["test"]
    extra = collections.Counter()
    for stem, r in recs.items():
        if stem in assigned or not r["boxes"]:
            continue
        members[r["split"]].add(stem)
        extra[r["split"]] += 1
    print(f"re-added images that had no person/car box: {dict(extra)}")

    manifest = {}
    for part in ["train", "test"]:
        stems = [s for s in members[part] if recs[s]["boxes"]]
        coco, dropped = to_coco(recs, stems, name_to_id, args.min_box)
        out = os.path.join(args.out_dir, f"hdr4rtt_voc20_{part}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        cnt = collections.Counter(a["category_id"] for a in coco["annotations"])
        top = ", ".join(f"{n}={cnt[name_to_id[n]]}" for n in names if cnt[name_to_id[n]])
        print(f"\n{part}: {len(coco['images'])} images, {len(coco['annotations'])} boxes "
              f"(dropped <{args.min_box:g}px: {dropped})")
        print(f"   {top}")
        zero = [n for n in names if not cnt[name_to_id[n]]]
        if zero:
            print(f"   classes with ZERO instances: {zero}")
        manifest[part] = {"images": len(coco["images"]), "boxes": len(coco["annotations"]),
                          "json": os.path.basename(out)}

    stems_all = sorted(members["train"] | members["test"])
    lst = os.path.join(args.out_dir, "images_to_convert.txt")
    with open(lst, "w", encoding="utf-8") as f:
        f.write("\n".join(s for s in stems_all if recs[s]["boxes"]))
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": name_to_id, "min_box": args.min_box, "splits": manifest,
                   "split_source": "seqsafe (leakage-free)"}, f, indent=1)
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
