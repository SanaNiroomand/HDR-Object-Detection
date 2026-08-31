#!/usr/bin/env python3
"""
build_deduped_split.py -- 20-class annotations restricted to the images that
survive near-duplicate removal.

Why this exists: scores on the full test set are inflated by near-duplicate
frames. Measured with the best arm (Reinhard) per source:

    S1  video / rendered        mAP 90.1   AP50 99.2   <- not a real result
    S3  HDR video               mAP 63.3   AP50 86.5
    S2  bracketed photographs   mAP 24.7   AP50 43.5   <- the only clean source

Kocdemir removed near-identical frames outright, 4,080 images down to 1,871.
That list was not available, so this uses Ulas's dedupe_hdr_frames.py, which
compares each frame to the last KEPT frame by SSIM on a tone-mapped copy.

Threshold choice, decided from retention per source rather than by targeting a
count:

    threshold   kept    S1 kept    S2 kept    S3 kept
      0.92      3145      28%        100%       99%     <- used here
      0.60      2822       7%         98%       98%
      0.30      2582       2%         89%       94%

0.92 removes what is actually duplicated and leaves the rest alone. The more
aggressive settings reach a count closer to Kocdemir's 1,871, but only by
deleting S2 images -- independent photographs taken across 218 separate days,
which contain no duplicates to remove. Losing those would trade one distortion
for another.

The result is NOT Kocdemir's split and must not be presented as such. It is this
project's own deduplication, and the surviving count differs from his.

S1 collapsing to 28% is itself the finding: 1,289 images contain only a few
hundred distinct scenes, which is why S1 scored 99.2 and why the headline number
was untrustworthy.
"""
import os
import csv
import json
import argparse
import collections

import pandas as pd

TARGET = 1280


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default=r"D:\Data\HDR\dedup_report_0.92.csv",
                   help="report.csv from dedupe_hdr_frames.py")
    p.add_argument("--root", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database")
    p.add_argument("--stats_csv", default=r"D:\Codes\HDR\Sana\hdr4rtt_analysis\hdr_stats_sources.csv")
    p.add_argument("--out_dir", default=r"D:\Data\HDR\hdr4rtt_voc20\annotations")
    p.add_argument("--prefix", default="hdr4rtt_voc20_dedup")
    p.add_argument("--min_box", type=float, default=8.0)
    args = p.parse_args()

    kept = set()
    with open(args.report) as f:
        for row in csv.DictReader(f):
            if str(row["kept"]) in ("1", "True", "true"):
                kept.add(os.path.splitext(row["filename"])[0])
    print(f"{len(kept)} images survive deduplication")

    src = {r.stem: r.source.split(":")[0]
           for r in pd.read_csv(args.stats_csv).itertuples()}
    by_src = collections.Counter(src.get(s) for s in kept)
    print(f"  by source: {dict(by_src)}")

    # load originals, keep each surviving image on the side it was already on
    recs, names = {}, set()
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
            names.add(nm)
            recs[byid[a["image_id"]]]["boxes"].append((a["bbox"], nm))

    names = sorted(names)
    name_to_id = {n: i + 1 for i, n in enumerate(names)}

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {}
    for split in ["train", "test"]:
        stems = sorted(s for s in kept
                       if s in recs and recs[s]["split"] == split and recs[s]["boxes"])
        images, anns, ann_id, dropped = [], [], 0, 0
        for img_id, stem in enumerate(stems):
            r = recs[stem]
            images.append({"height": TARGET, "width": TARGET, "id": img_id,
                           "file_name": stem + ".png"})
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
                             "category_id": name_to_id[cname],
                             "bbox": [round(x0, 2), round(y0, 2), round(w, 2), round(h, 2)],
                             "area": round(w * h, 2), "iscrowd": 0})
                ann_id += 1
        coco = {"info": "", "license": [""],
                "categories": [{"id": v, "name": k}
                               for k, v in sorted(name_to_id.items(), key=lambda kv: kv[1])],
                "images": images, "annotations": anns}
        out = os.path.join(args.out_dir, f"{args.prefix}_{split}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(coco, f)
        per_src = collections.Counter(src.get(s) for s in stems)
        print(f"{split:5}: {len(images):4} images  {len(anns):5} boxes  "
              f"(dropped <{args.min_box:g}px: {dropped})  sources {dict(per_src)}")
        manifest[split] = {"images": len(images), "boxes": len(anns),
                           "json": os.path.basename(out), "by_source": dict(per_src)}

        # per-source subsets, so the clean source can be reported on its own
        for g in ["S1", "S2", "S3"]:
            keep_ids = {im["id"] for im in images
                        if src.get(os.path.splitext(im["file_name"])[0]) == g}
            sub = {"info": "", "license": [""], "categories": coco["categories"],
                   "images": [im for im in images if im["id"] in keep_ids],
                   "annotations": [a for a in anns if a["image_id"] in keep_ids]}
            with open(os.path.join(args.out_dir, f"{args.prefix}_{split}_{g}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(sub, f)

    with open(os.path.join(args.out_dir, f"{args.prefix}_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"report": args.report, "n_kept_images": len(kept),
                   "kept_by_source": dict(by_src), "classes": name_to_id,
                   "splits": manifest,
                   "note": "own deduplication, NOT Kocdemir's split"}, f, indent=1)
    print(f"\nwrote {args.out_dir}\\{args.prefix}_*.json")


if __name__ == "__main__":
    main()
