#!/usr/bin/env python3
"""
make_source_subsets.py -- split each test annotation file by source group.

Needed to measure the train/test leakage cleanly. Duplicate frames only exist in
S3 (the continuous video); S1 and S2 have no known frame adjacency. So the
leakage effect must be measured S3-against-S3, otherwise the comparison is
confounded by content -- S3 is traffic-heavy video while S2 is bracketed stills.
"""
import os
import json
import argparse
import collections

import pandas as pd

ANN = r"D:\Data\HDR\hdr4rtt_rod\annotations"
STATS = r"D:\Codes\HDR\Sana\hdr4rtt_analysis\hdr_stats_sources.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann_dir", default=ANN)
    p.add_argument("--stats_csv", default=STATS)
    args = p.parse_args()

    df = pd.read_csv(args.stats_csv)
    src = {r.stem: r.source.split(":")[0] for r in df.itertuples()}

    for split in ["seqsafe", "original"]:
        path = os.path.join(args.ann_dir, f"hdr4rtt_rod_{split}_test.json")
        d = json.load(open(path, encoding="utf-8"))
        by_img = {im["id"]: os.path.splitext(os.path.splitext(im["file_name"])[0])[0]
                  for im in d["images"]}
        for group in ["S1", "S2", "S3"]:
            keep = {i for i, s in by_img.items() if src.get(s) == group}
            sub = {
                "info": d.get("info", ""), "license": d.get("license", [""]),
                "categories": d["categories"],
                "images": [im for im in d["images"] if im["id"] in keep],
                "annotations": [a for a in d["annotations"] if a["image_id"] in keep],
            }
            out = os.path.join(args.ann_dir, f"hdr4rtt_rod_{split}_test_{group}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(sub, f)
            per_c = collections.Counter(a["category_id"] for a in sub["annotations"])
            print(f"{split:9} {group}: {len(sub['images']):4} images  "
                  f"{len(sub['annotations']):5} boxes  "
                  f"(Pedestrian {per_c[1]}, Car {per_c[2]})")
        print()


if __name__ == "__main__":
    main()
