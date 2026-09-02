#!/usr/bin/env python3
"""
make_frontend_native.py -- regenerate one front end without squashing the aspect
ratio, and at the resolution the source actually has.

The existing pipeline forces every image to 1280x1280. That came from RAOD, whose
annotation script squashes boxes to match, so it was necessary there. The
torchvision arm inherited it for no reason: those detectors accept any input size.

Two things were being lost:

  aspect ratio   a 4208x3120 photo is stretched 1.35x horizontally
  resolution     that same photo drops from 13.1 MP to 1.64 MP

Here each image is scaled so its long side is min(--long_side, native long side),
keeping the aspect ratio. Nothing is upscaled: a 1280x720 source stays 1280x720
rather than being blown up to fill a target. Boxes are scaled by the same single
factor, so they stay aligned by construction.

Writes both the images and a matching pair of annotation files, since the two
have to change together.

Note this only matters alongside the detector's own resize limits. torchvision
rescales its input so the short side is min_size and the long side is at most
max_size; at the defaults of 800 and 1333 it would shrink these right back down.
See --min_size on train_frontend.py.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

from make_frontends import tonemap

cv2.setNumThreads(1)
EPS = 1e-6


def prepare_native(bgr, long_side):
    """Scale by one factor, aspect preserved, never upscaling. Returns the image
    and the factor, so boxes can be moved by exactly the same amount."""
    img = np.clip(bgr.astype(np.float32), 0, None)[:, :, ::-1]
    h, w = img.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    mr, mg, mb = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    if mr > EPS:
        img[:, :, 0] *= mg / mr
    if mb > EPS:
        img[:, :, 2] *= mg / mb
    return img, scale


def rebuild_annotations(src_json, out_json, sizes, min_box):
    """Same boxes, rescaled per image by that image's own factor."""
    d = json.load(open(src_json, encoding="utf-8"))
    keep_imgs, id_scale = [], {}
    for im in d["images"]:
        stem = os.path.splitext(os.path.splitext(im["file_name"])[0])[0]
        if stem not in sizes:
            continue
        W, H, sc = sizes[stem]
        id_scale[im["id"]] = (W, H, sc)
        keep_imgs.append({**im, "width": W, "height": H, "file_name": stem + ".png"})
    anns, dropped, ann_id = [], 0, 0
    for a in d["annotations"]:
        if a["image_id"] not in id_scale:
            continue
        W, H, sc = id_scale[a["image_id"]]
        # the source boxes live in the squashed 1280x1280 frame, so undo that
        # first, then apply this image's own factor
        x, y, w, h = a["bbox"]
        x, w = x / 1280.0, w / 1280.0
        y, hh = y / 1280.0, h / 1280.0
        x, w = x * W, w * W
        y, hh = y * H, hh * H
        if w < min_box or hh < min_box:
            dropped += 1
            continue
        anns.append({**a, "id": ann_id, "bbox": [round(x, 2), round(y, 2),
                                                 round(w, 2), round(hh, 2)],
                     "area": round(w * hh, 2)})
        ann_id += 1
    out = {"info": "", "license": [""], "categories": d["categories"],
           "images": keep_imgs, "annotations": anns}
    json.dump(out, open(out_json, "w", encoding="utf-8"))
    print(f"  {os.path.basename(out_json)}: {len(keep_imgs)} images, {len(anns)} boxes "
          f"(dropped <{min_box:g}px: {dropped})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="reinhard")
    p.add_argument("--exr_dir", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database\images")
    p.add_argument("--ann_dir", default=r"D:\Data\HDR\hdr4rtt_voc20\annotations")
    p.add_argument("--src_prefix", default="hdr4rtt_voc20_dedup")
    p.add_argument("--out_root", default=r"D:\Data\HDR\hdr4rtt_voc20\frontends_native")
    p.add_argument("--long_side", type=int, default=2048)
    p.add_argument("--percentile", type=float, default=99.9)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--min_box", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_root) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = set()
    for part in ["train", "test"]:
        d = json.load(open(os.path.join(args.ann_dir, f"{args.src_prefix}_{part}.json"),
                           encoding="utf-8"))
        for im in d["images"]:
            stems.add(os.path.splitext(os.path.splitext(im["file_name"])[0])[0])
    stems = sorted(stems)
    print(f"{len(stems)} images, arm={args.arm}, long side capped at {args.long_side}")

    sizes, fails, t0, done = {}, [], time.time(), 0

    def work(stem):
        bgr = cv2.imread(os.path.join(args.exr_dir, stem + ".exr"),
                         cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if bgr is None or bgr.ndim != 3:
            return stem, None
        img, sc = prepare_native(bgr[:, :, :3], args.long_side)
        out = tonemap(img, args.arm, args.percentile, args.gamma)
        u8 = (out * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(str(out_dir / (stem + ".png")), u8[:, :, ::-1],
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return stem, (u8.shape[1], u8.shape[0], sc)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for stem, r in ex.map(work, stems):
            done += 1
            if r is None:
                fails.append(stem)
            else:
                sizes[stem] = r
            if done % 500 == 0 or done == len(stems):
                el = time.time() - t0
                print(f"  {done}/{len(stems)}  {el:.0f}s", flush=True)

    wh = np.array([[v[0], v[1]] for v in sizes.values()])
    mp = (wh[:, 0] * wh[:, 1]) / 1e6
    print(f"\ndone in {time.time()-t0:.0f}s | {len(sizes)} images | failed {len(fails)}")
    print(f"  resolution: median {np.median(mp):.2f} MP, min {mp.min():.2f}, max {mp.max():.2f}")
    print(f"  against 1.64 MP for the squashed 1280x1280 version "
          f"({np.median(mp)/1.64:.1f}x more pixels at the median)")
    print(f"  distinct shapes: {len(set(map(tuple, wh)))}")

    print("\nannotations:")
    for part in ["train", "test"]:
        rebuild_annotations(
            os.path.join(args.ann_dir, f"{args.src_prefix}_{part}.json"),
            os.path.join(args.ann_dir, f"{args.src_prefix}_native_{part}.json"),
            sizes, args.min_box)

    json.dump({"arm": args.arm, "long_side": args.long_side, "n": len(sizes),
               "failed": fails, "aspect_preserved": True,
               "median_megapixels": float(np.median(mp))},
              open(out_dir / "manifest.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
