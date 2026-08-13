#!/usr/bin/env python3
"""
convert_hdr4rtt_to_rod.py

Converts HDR4RTT .exr files into the .npy.gz format RAOD's COCORawDataset reads.

WHAT CHANGED vs the previous converter, and why
-----------------------------------------------
The earlier convert_exr_to_raod.py stretched each image so it filled [0,255]
(via a per-population white point, or p99, or log, or histogram matching). That
is the wrong target for this model, and it is why none of those variants worked.

RAOD's model tone-maps internally. models/adaptive_module.py applies

    out = img ** (1.0 / gamma),   gamma learned in [7.0, 10.5]

plus a learned local piecewise curve, and models/yolox.py then does
clamp(x_tm, 0, 1) * 255. The dataloader feeds it [0,1]: coco_raw.py:141 does
`img = img / 255.0` after loading the .npy.gz. So the network is built to
receive DARK, LINEAR, UN-tone-mapped input and brighten it itself.

Measured on RAOD's own sample (scripts/official_output/day-00018.npy.gz):
    mean 3.017 in [0,255]  ->  0.0118 in [0,1]  ->  ^(1/7) = 0.53, mid-grey.

The old converter produced mean 96.2 in [0,255] = 0.377 in [0,1], 32x brighter
than anything the model saw in training; ^(1/7) = 0.87, washed toward white.
Every "use the full range" method fails the same way, which matches the note in
README9August.txt that no rescaling variant gave good results.

So this script does NO stretching. HDR4RTT's stored values already sit in
roughly the same numeric range as ROD's post-normalisation values -- measured
across all 4,080 files, the median pixel is 1.51 (S1), 3.73 (S2), 7.73 (S3)
against ROD's mean of 3.02. Passing them through unchanged lands all three
sources within ~2.5x of the model's operating point with no per-file tuning.

Pipeline, mirroring scripts/preprocess_raw.py step for step:
    1. read EXR (cv2 gives BGR float32)
    2. clamp negatives to 0   -- 41% of HDR4RTT images contain negative pixels,
                                 down to -872, from debayer/colour-conversion
                                 undershoot. Left in, they corrupt the AWB means.
    3. BGR -> RGB             -- preprocess_raw.py's AWB reads index 0 as R and
                                 index 2 as B, so the saved convention is RGB.
    4. resize to 1280x1280, aspect ratio NOT preserved -- preprocess_raw.py does
                                 exactly this (cv2.resize(im,(1280,1280))), and
                                 preprocess_anno.py squashes the boxes to match
                                 by scaling x and y independently. Our
                                 build_rod_annotations.py reproduces that.
    5. gray-world AWB: scale R and B so their means match G's
    6. clip to [0, white_point]  -- the analogue of preprocess_raw.py's
                                 clip(im, 0, BIT24-1); values above the clip are
                                 highlights, which the model's local tone-mapping
                                 branch and final clamp(0,1) absorb anyway
    7. save float32, gzipped

--gain exists only so the operating point can be swept if you want to test
sensitivity; the default of 1.0 is the measured match and needs no tuning.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import gzip
import time
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

cv2.setNumThreads(1)

TARGET = 1280
ROD_REFERENCE_MEAN = 3.017  # day-00018.npy.gz, in [0,255] units


def convert_one(img, white_point=255.0, gain=1.0, target=TARGET):
    """img: HxWx3 float32 BGR from cv2. Returns HxWx3 float32 RGB in [0,white_point]."""
    img = img.astype(np.float32)
    img = np.clip(img, 0, None)          # negatives -> 0, before they poison the AWB means
    img = img[:, :, ::-1].copy()         # BGR -> RGB (preprocess_raw.py's saved convention)
    img = cv2.resize(img, (target, target), interpolation=cv2.INTER_LINEAR)

    mean_r = img[:, :, 0].mean()
    mean_g = img[:, :, 1].mean()
    mean_b = img[:, :, 2].mean()
    if mean_r > 1e-8:
        img[:, :, 0] *= mean_g / mean_r
    if mean_b > 1e-8:
        img[:, :, 2] *= mean_g / mean_b

    if gain != 1.0:
        img *= gain
    return np.clip(img, 0, white_point).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database\images")
    p.add_argument("--output_dir", default=r"D:\Data\HDR\hdr4rtt_rod\images")
    p.add_argument("--file_list", default=r"D:\Data\HDR\hdr4rtt_rod\annotations\images_to_convert.txt",
                   help="one stem per line; omit to convert every .exr in input_dir")
    p.add_argument("--white_point", type=float, default=255.0)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--compresslevel", type=int, default=1,
                   help="gzip level; 1 is ~5x faster than 9 and only ~10% larger on float data")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry_run", action="store_true",
                   help="convert but do not write; report the resulting distribution "
                        "against ROD's reference so the operating point can be checked first")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.file_list and os.path.exists(args.file_list):
        stems = [l.strip() for l in open(args.file_list, encoding="utf-8") if l.strip()]
        files = [in_dir / (s + ".exr") for s in stems]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} listed files missing, e.g. {missing[:3]}")
    else:
        files = sorted(in_dir.glob("*.exr"))
    if args.limit:
        rng = np.random.default_rng(0)
        files = [files[i] for i in sorted(rng.choice(len(files), min(args.limit, len(files)),
                                                     replace=False))]
    print(f"{len(files)} files | white_point={args.white_point} gain={args.gain} "
          f"| {'DRY RUN' if args.dry_run else out_dir}")

    stats, failures = [], []
    t0 = time.time()
    done = 0

    def work(f):
        out_path = out_dir / (f.stem + ".npy.gz")
        if not args.dry_run and not args.overwrite and out_path.exists():
            return ("skip", f.stem, None)
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None or img.ndim != 3 or img.shape[2] < 3:
            return ("fail", f.stem, None)
        conv = convert_one(img[:, :, :3], args.white_point, args.gain)
        st = (float(conv.mean()), float(np.median(conv)), float(conv.max()),
              float((conv >= args.white_point).mean()))
        if not args.dry_run:
            with gzip.GzipFile(str(out_path), "w", compresslevel=args.compresslevel) as gz:
                np.save(file=gz, arr=conv)
        return ("ok", f.stem, st)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, stem, st in ex.map(work, files):
            done += 1
            if status == "fail":
                failures.append(stem)
            elif st is not None:
                stats.append(st)
            if done % 250 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  {done}/{len(files)}  {el:.0f}s  eta {el/done*(len(files)-done):.0f}s",
                      flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s | converted {len(stats)} | failed {len(failures)}")
    if failures:
        print("  failures:", failures[:10])
    if not stats:
        return

    a = np.array(stats)
    means, meds, maxs, satu = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    print("\n--- resulting distribution, in [0,255] units as stored ---")
    print(f"  per-image mean  : p5={np.percentile(means,5):.3f}  median={np.median(means):.3f}  "
          f"p95={np.percentile(means,95):.3f}")
    print(f"  per-image median: p5={np.percentile(meds,5):.4f}  median={np.median(meds):.4f}  "
          f"p95={np.percentile(meds,95):.3f}")
    print(f"  pixels at clip  : median={np.median(satu):.4%}  p95={np.percentile(satu,95):.3%}")
    print(f"\n  ROD reference mean = {ROD_REFERENCE_MEAN:.3f}")
    print(f"  ratio to ROD (median of per-image means) = "
          f"{np.median(means)/ROD_REFERENCE_MEAN:.2f}x")
    g = 7.0
    print(f"  after loader /255 and TMM ^(1/{g:g}): "
          f"typical image -> {(np.median(means)/255)**(1/g):.3f}  "
          f"(ROD reference -> {(ROD_REFERENCE_MEAN/255)**(1/g):.3f}; 0.5 is mid-grey)")

    if not args.dry_run:
        with open(out_dir / "conversion_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"input_dir": str(in_dir), "n": len(stats),
                       "white_point": args.white_point, "gain": args.gain,
                       "target": TARGET, "aspect_preserved": False,
                       "median_of_per_image_means": float(np.median(means)),
                       "rod_reference_mean": ROD_REFERENCE_MEAN,
                       "failures": failures}, f, indent=1)
        print(f"\nmanifest -> {out_dir / 'conversion_manifest.json'}")


if __name__ == "__main__":
    main()
