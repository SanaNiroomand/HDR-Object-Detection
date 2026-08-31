#!/usr/bin/env python3
"""
build_rod_dedup_annotations.py -- 2-class (person/car) annotations in RAOD's
format, restricted to the images that survive near-duplicate removal.

The headline table was measured on a split that still contained the
near-duplicate S1 frames, so its absolute values are inflated the same way the
tone-mapping table was before it was re-run. This rebuilds that split on the
deduplicated image set so the two tables sit on the same footing.

Same class mapping as build_rod_annotations.py: person -> Pedestrian (1),
car -> Car (2), with all five ROD categories declared so
`class_ids.index(category_id)` yields the head index RAOD was trained with. Same
geometry too: boxes squashed anisotropically into 1280x1280, matching the
non-aspect-preserving resize both converters apply.

One annotation set serves both arms. RAOD reads the .npy.gz images and the
torchvision arm reads the .png ones; both loaders swap the extension themselves,
so the boxes and image ids are shared and the two arms stay comparable.
"""
import os
import csv
import json
import argparse
import collections

ROD_CLASSES = {"Pedestrian": 1, "Car": 2, "Cyclist": 3, "Tram": 4, "Truck": 5}
HDR4RTT_TO_ROD = {"person": "Pedestrian", "car": "Car"}
TARGET = 1280


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default=r"D:\Data\HDR\dedup_report_0.92.csv")
    p.add_argument("--root", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database")
    p.add_argument("--out_dir", default=r"D:\Data\HDR\hdr4rtt_rod\annotations")
    p.add_argument("--prefix", default="hdr4rtt_rod_dedup")
    p.add_argument("--min_box", type=float, default=16.0)
    args = p.parse_args()

    kept = set()
    with open(args.report) as f:
        for row in csv.DictReader(f):
            if str(row["kept"]) in ("1", "True", "true"):
                kept.add(os.path.splitext(row["filename"])[0])
    print(f"{len(kept)} images survive deduplication")

    recs = {}
    for split, fn in [("train", "instances_train2020.json"),
                      ("test", "instances_test2020.json")]:
        with open(os.path.join(args.root, "annotations", fn), encoding="utf-8") as f:
            d = json.load(f)
        cats = {c["id"]: c["name"] for c in d["categories"]}
        byid = {}
        for im in d["images"]:
            stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
            byid[im["id"]] = stem
            recs[stem] = {"w": im["width"], "h": im["height"], "split": split, "boxes": []}
        for a in d["annotations"]:
            nm = cats[a["category_id"]]
            if nm in HDR4RTT_TO_ROD:
                recs[byid[a["image_id"]]]["boxes"].append((a["bbox"], HDR4RTT_TO_ROD[nm]))

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {}
    for split in ["train", "test"]:
        stems = sorted(s for s in kept
                       if s in recs and recs[s]["split"] == split and recs[s]["boxes"])
        images, anns, ann_id, dropped = [], [], 0, 0
        for img_id, stem in enumerate(stems):
            r = recs[stem]
            images.append({"height": TARGET, "width": TARGET, "id": img_id,
                           "file_name": stem + ".npy.gz"})
            sx, sy = TARGET / r["w"], TARGET / r["h"]
            for (bx, by, bw, bh), cname in r["boxes"]:
                x0, y0 = max(0.0, bx * sx), max(0.0, by * sy)
                x1 = min(float(TARGET), (bx + bw) * sx)
                y1 = min(float(TARGET), (by + bh) * sy)
                w, h = x1 - x0, y1 - y0
                if w < args.min_box or h < args.min_box:
                    dropped += 1
                    continue
                anns.append({"id": ann_id, "image_id": img_id,
                             "category_id": ROD_CLASSES[cname],
                             "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                             "area": round(w * h, 2), "iscrowd": 0,
                             "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]]})
                ann_id += 1
        coco = {"info": "", "license": [""],
                "categories": [{"id": v, "name": k}
                               for k, v in sorted(ROD_CLASSES.items(), key=lambda kv: kv[1])],
                "images": images, "annotations": anns}
        out = os.path.join(args.out_dir, f"{args.prefix}_{split}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        per_c = collections.Counter(a["category_id"] for a in anns)
        print(f"{split:5}: {len(images):4} images  {len(anns):5} boxes  "
              f"(Pedestrian {per_c[1]}, Car {per_c[2]})  dropped<{args.min_box:g}px: {dropped}")
        manifest[split] = {"images": len(images), "boxes": len(anns),
                           "json": os.path.basename(out)}

    with open(os.path.join(args.out_dir, f"{args.prefix}_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"report": args.report, "n_kept": len(kept), "splits": manifest,
                   "class_map": HDR4RTT_TO_ROD, "rod_classes": ROD_CLASSES,
                   "note": "own deduplication, NOT Kocdemir's split"}, f, indent=1)
    print(f"\nwrote {args.out_dir}\\{args.prefix}_*.json")


if __name__ == "__main__":
    main()
