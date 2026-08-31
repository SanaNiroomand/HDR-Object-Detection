#!/usr/bin/env python3
"""
analyze_failures_controlled.py -- separate the effect of brightness and dynamic
range from the effect of object size.

The raw slices from analyze_failures.py look like they contradict the HDR
premise: objects in deep shadow are missed LESS often than normally-lit ones
(24% against 44%), and low-dynamic-range scenes are missed MORE often than
high-dynamic-range ones (55% against 20%).

Both are suspect, because object size dominates everything (80% missed under 2%
of the image side, 8% missed above 20%) and size is almost certainly correlated
with the other two. A person standing in a dark foreground is both close to the
camera and in shadow. So the shadow slice may just be measuring "big".

This holds size fixed and asks whether brightness and dynamic range still matter
inside each size band. If they flatten out, size was the whole story.
"""
import argparse
import os

import numpy as np
import pandas as pd


def cross(df, row_col, row_edges, row_labels, title, min_n=25):
    """Miss rate by `row_col` within each size band."""
    size_edges = [5, 10, 20]
    size_labels = ["<5%", "5-10%", "10-20%", ">20%"]
    df = df[np.isfinite(df[row_col])].copy()
    df["_size"] = np.digitize(df["size"], size_edges)
    df["_row"] = np.digitize(df[row_col], row_edges)

    print(f"\n{title}")
    print("  miss rate, size held fixed across each row")
    header = f"    {'':<24}" + "".join(f"{s:>12}" for s in size_labels)
    print(header)
    print("    " + "-" * (24 + 12 * len(size_labels)))
    for i, lab in enumerate(row_labels):
        cells = []
        for j in range(len(size_labels)):
            sel = (df["_row"] == i) & (df["_size"] == j)
            cells.append(f"{df.loc[sel,'missed'].mean():11.0%}" if sel.sum() >= min_n
                         else f"{'-':>11}")
        n = int((df["_row"] == i).sum())
        print(f"    {lab:<24}" + "".join(cells) + f"   (n={n})")

    # is the trend within a size band real, or flat?
    print("    spread within each size band (max - min over rows):")
    for j, s in enumerate(size_labels):
        vals = []
        for i in range(len(row_labels)):
            sel = (df["_row"] == i) & (df["_size"] == j)
            if sel.sum() >= min_n:
                vals.append(df.loc[sel, "missed"].mean())
        if len(vals) >= 2:
            print(f"      {s:<8} {max(vals)-min(vals):5.0%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=r"D:\Codes\HDR\Sana\hdr4rtt_rod\failures\retinanet_reinhard_dedup_gt_boxes.csv")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"{len(df)} ground-truth boxes, {df['missed'].mean():.1%} missed overall")

    # --- is size confounded with the other two? ---
    print("\n" + "=" * 78)
    print("IS SIZE CONFOUNDED WITH BRIGHTNESS AND DYNAMIC RANGE?")
    print("=" * 78)
    for col, lab in [("rel_lum", "brightness relative to scene"),
                     ("box_dr", "dynamic range inside the box"),
                     ("scene_dr", "dynamic range of the scene")]:
        d = df[np.isfinite(df[col]) & np.isfinite(df["size"])]
        r = np.corrcoef(np.log10(np.maximum(d["size"], 1e-3)),
                        np.log10(np.maximum(d[col], 1e-6)) if col != "scene_dr" else d[col])[0, 1]
        print(f"  size vs {lab:<32} r = {r:+.2f}")
    print("\n  A strong correlation means the raw slice for that variable is partly")
    print("  just measuring object size.")

    print("\n" + "=" * 78)
    print("HOLDING SIZE FIXED")
    print("=" * 78)
    cross(df, "rel_lum", [0.25, 0.5, 1.0, 2.0, 5.0],
          ["<0.25x deep shadow", "0.25-0.5x", "0.5-1x", "1-2x", "2-5x", ">5x bright"],
          "BRIGHTNESS relative to the object's own scene")
    cross(df, "scene_dr", [3, 4, 5], ["<3 decades", "3-4", "4-5", ">5 decades"],
          "DYNAMIC RANGE of the whole scene")
    cross(df, "box_dr", [1, 2, 3], ["<1 decade", "1-2", "2-3", ">3 decades"],
          "DYNAMIC RANGE inside the object's box")

    # --- what actually predicts a miss, all together ---
    print("\n" + "=" * 78)
    print("RANKED BY HOW MUCH EACH ONE MOVES THE MISS RATE")
    print("=" * 78)
    rows = []
    for col, edges, lab in [("size", [5, 10, 20], "object size"),
                            ("rel_lum", [0.25, 0.5, 1, 2, 5], "brightness vs scene"),
                            ("box_dr", [1, 2, 3], "box dynamic range"),
                            ("scene_dr", [3, 4, 5], "scene dynamic range")]:
        d = df[np.isfinite(df[col])]
        idx = np.digitize(d[col], edges)
        vals = [d.loc[idx == i, "missed"].mean() for i in range(len(edges) + 1)
                if (idx == i).sum() >= 25]
        if len(vals) >= 2:
            rows.append((lab, max(vals) - min(vals)))
    for lab, spread in sorted(rows, key=lambda r: -r[1]):
        bar = "#" * int(spread * 60)
        print(f"  {lab:<22} {spread:5.0%}  {bar}")

    # --- the single worst image, for context ---
    print("\n" + "=" * 78)
    print("MOST-MISSED IMAGES: how many boxes do they hold?")
    print("=" * 78)
    g = df.groupby("stem").agg(n=("missed", "size"), miss=("missed", "sum"),
                               med_size=("size", "median"))
    g["rate"] = g["miss"] / g["n"]
    for stem, r in g.sort_values("miss", ascending=False).head(6).iterrows():
        print(f"  {stem}  {int(r['miss']):3}/{int(r['n']):3} missed  "
              f"median object size {r['med_size']:.1f}% of image side")
    print("\n  A high count with a small median size means the image is crowded with")
    print("  tiny objects, which the size row above already explains.")


if __name__ == "__main__":
    main()
