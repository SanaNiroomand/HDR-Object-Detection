#!/usr/bin/env python3
"""
anchor_fit.py -- how many ground-truth boxes can an anchor set even match?

RetinaNet marks an anchor as positive when it overlaps an object by IoU 0.5. An
object that no anchor comes within 0.5 of never becomes a positive example, so
it is close to unlearnable however good the features are. This measures that
ceiling before any training, which is cheap enough to run before committing a
GPU to a configuration.

IoU here is shape-only: anchor and box are taken as concentric. Anchors tile the
image densely, so centring is nearly free and this is the best case. Treat the
numbers as an upper bound on what the anchor set permits, not as a score.

torchvision's aspect_ratios are height/width, so ratio 2 is twice as tall as wide.

Two results this produced for HDR4RTT, both of which contradicted the obvious guess:

  - raising the input from 800 to 1280 and scaling the anchor sizes to match
    (the textbook move, to preserve relative coverage) drops small-object
    matching to zero: the smallest anchor goes 32px -> 51px, and no object under
    32^2 in area can reach IoU 0.5 against it.

  - the sizes were never the mismatch. 85% of these objects are taller than
    wide, median height/width 2.39, while the stock ratios stop at 2.0.
    Retuning the ratios alone lifts every size bucket at once.
"""
import json
import argparse
import itertools

import numpy as np

# torchvision's retinanet_resnet50_fpn defaults, one tuple per FPN level
DEFAULT_SIZES = ((32, 40, 50), (64, 80, 101), (128, 161, 203),
                 (256, 322, 406), (512, 645, 812))
DEFAULT_RATIOS = (0.5, 1.0, 2.0)


def best_iou(wh, sizes, ratios):
    """Max concentric IoU of each (w, h) against every anchor in the set."""
    s = np.array([v for lvl in sizes for v in lvl], dtype=np.float64)
    r = np.array(ratios, dtype=np.float64)
    aw = (s[:, None] / np.sqrt(r)[None, :]).ravel()
    ah = (s[:, None] * np.sqrt(r)[None, :]).ravel()
    w, h = wh[:, 0:1], wh[:, 1:2]
    inter = np.minimum(w, aw[None, :]) * np.minimum(h, ah[None, :])
    union = (w * h) + (aw * ah)[None, :] - inter
    return (inter / union).max(axis=1)


def matched(wh, sizes, ratios, thr=0.5):
    return float((best_iou(wh, sizes, ratios) >= thr).mean())


def row(name, wh, img_scale, size_mult, ratios, buckets=None):
    """img_scale: the detector resizes the image, and the boxes with it, before
    laying down anchors. A 1280 image at min_size 800 scales by 0.625."""
    sizes = tuple(tuple(v * size_mult for v in lvl) for lvl in DEFAULT_SIZES)
    b = wh * img_scale
    ok = best_iou(b, sizes, ratios) >= 0.5
    cells = ""
    if buckets is not None:
        area = b[:, 0] * b[:, 1]
        for m in [area < 32 ** 2, (area >= 32 ** 2) & (area < 96 ** 2), area >= 96 ** 2]:
            cells += f"{100 * ok[m].mean():7.1f}" if m.any() else "      -"
    print(f"  {name:<32}{100 * ok.mean():6.1f}{cells}")
    return ok.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", default=r"D:\Data\HDR\hdr4rtt_voc20\annotations"
                                    r"\hdr4rtt_voc20_dedup_train.json")
    p.add_argument("--search", action="store_true",
                   help="search for the best three ratios, holding the count at "
                        "three: the pretrained regression head is shaped by the "
                        "anchor count, so it cannot change")
    args = p.parse_args()

    d = json.load(open(args.ann, encoding="utf-8"))
    wh = np.array([a["bbox"][2:4] for a in d["annotations"]], dtype=np.float64)
    wh = wh[(wh[:, 0] > 0) & (wh[:, 1] > 0)]
    ar = wh[:, 1] / wh[:, 0]

    print(f"{len(wh)} boxes, in the 1280x1280 frame the annotations use")
    print(f"  height/width   p5 {np.percentile(ar, 5):.2f}   p25 {np.percentile(ar, 25):.2f}"
          f"   median {np.median(ar):.2f}   p75 {np.percentile(ar, 75):.2f}"
          f"   p95 {np.percentile(ar, 95):.2f}")
    print(f"  taller than wide: {100 * (ar > 1).mean():.0f}%   "
          f"stock ratios reach only {max(DEFAULT_RATIOS):g}\n")

    print("share of boxes reaching IoU 0.5 with some anchor (%)\n")
    print(f"  {'configuration':<32}{'all':>6}{'small':>7}{'medium':>7}{'large':>7}")
    print("  " + "-" * 57)
    row("baseline  800px, stock", wh, 800 / 1280, 1.0, DEFAULT_RATIOS, True)
    print()
    row("arm B    1280px, stock", wh, 1.0, 1.0, DEFAULT_RATIOS, True)
    row("  + sizes x1.6", wh, 1.0, 1.6, DEFAULT_RATIOS, True)
    print()
    row("  + ratios 1, 2.5, 5", wh, 1.0, 1.0, (1.0, 2.5, 5.0), True)
    row("  + ratios 0.5, 1.6, 6.4", wh, 1.0, 1.0, (0.5, 1.6, 6.4), True)
    print("\n  small/medium/large are the COCO area buckets, measured in the "
          "resized frame,\n  so the same object can change bucket between rows.")

    if args.search:
        grid = np.round(np.exp(np.linspace(np.log(0.4), np.log(8.0), 40)), 2)
        best, arg = 0.0, None
        for c in itertools.combinations(grid, 3):
            m = matched(wh, DEFAULT_SIZES, c)
            if m > best:
                best, arg = m, c
        print(f"\nbest three ratios found: {tuple(float(v) for v in arg)} "
              f"-> {100 * best:.1f}% matched")


if __name__ == "__main__":
    main()
