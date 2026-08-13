#!/usr/bin/env python3
"""
convert_hdr4rtt_to_ldr.py

Converts HDR4RTT .exr into ordinary 8-bit sRGB PNGs -- the "no learned
tone-mapping module" arm of the detector comparison.

This is row 2 of:
    1. tiny detector + RAOD's learned tone-mapping module   (done, mAP 0.349)
    2. big detector, ordinary tone mapping                  <- this script
    3. big detector + RAOD's module
Row 3 minus row 2 isolates what the learned module actually contributes once the
detector is no longer a 1M-parameter model.

Geometry is deliberately IDENTICAL to convert_hdr4rtt_to_rod.py: negatives
clamped, BGR->RGB, squashed to 1280x1280 without preserving aspect, gray-world
white balance. That way the same annotation files describe both versions and the
only difference between the arms is the pixels' tone curve, not the framing.

Tone-mapping operators (--tmo):
  gamma     per-image normalise by a high percentile, then sRGB-style gamma.
            The standard display transform and the honest "what you would do
            without a learned module" default.
  reinhard  global Reinhard: L/(1+L) after scaling to a target key. Classical,
            and one of the operators TMO-Det compares against.
  log       log1p compression normalised to the same percentile.

The percentile matters: HDR4RTT images are individually normalised so their
single brightest pixel sits on the ceiling (measured: ~1 pixel per image), so
normalising by the true max would be driven by that lone pixel. p99.5 ignores it.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

cv2.setNumThreads(1)
TARGET = 1280


def tonemap(img, tmo="gamma", pct=99.5, gamma=2.2, key=0.18):
    """img: HxWx3 float32 RGB, linear, non-negative. Returns uint8 sRGB."""
    ref = np.percentile(img, pct)
    if not np.isfinite(ref) or ref <= 1e-8:
        ref = max(float(img.max()), 1e-8)

    if tmo == "reinhard":
        lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        key_now = np.exp(np.mean(np.log(lum + 1e-6)))
        scaled = img * (key / max(key_now, 1e-8))
        out = scaled / (1.0 + scaled)
    elif tmo == "log":
        out = np.log1p(np.clip(img, 0, ref)) / np.log1p(ref)
    else:  # gamma
        out = np.clip(img / ref, 0, 1)

    out = np.clip(out, 0, 1) ** (1.0 / gamma)
    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def convert_one(bgr, tmo, pct, gamma):
    img = bgr.astype(np.float32)
    img = np.clip(img, 0, None)           # 41% of HDR4RTT images contain negatives
    img = img[:, :, ::-1].copy()          # BGR -> RGB
    img = cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)

    mr, mg, mb = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    if mr > 1e-8:
        img[:, :, 0] *= mg / mr
    if mb > 1e-8:
        img[:, :, 2] *= mg / mb

    return tonemap(img, tmo, pct, gamma)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database\images")
    p.add_argument("--output_dir", default=r"D:\Data\HDR\hdr4rtt_ldr\images")
    p.add_argument("--file_list", default=r"D:\Data\HDR\hdr4rtt_rod\annotations\images_to_convert.txt")
    p.add_argument("--tmo", default="gamma", choices=["gamma", "reinhard", "log"])
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = [l.strip() for l in open(args.file_list, encoding="utf-8") if l.strip()]
    files = [in_dir / (s + ".exr") for s in stems]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} listed files missing, e.g. {missing[:3]}")
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} files | tmo={args.tmo} p{args.percentile} gamma={args.gamma} -> {out_dir}")

    t0, done, fails, means = time.time(), 0, [], []

    def work(f):
        outp = out_dir / (f.stem + ".png")
        if outp.exists() and not args.overwrite:
            return ("skip", None)
        bgr = cv2.imread(str(f), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
            return ("fail", f.stem)
        u8 = convert_one(bgr[:, :, :3], args.tmo, args.percentile, args.gamma)
        cv2.imwrite(str(outp), u8[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return ("ok", float(u8.mean()))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for status, val in ex.map(work, files):
            done += 1
            if status == "fail":
                fails.append(val)
            elif status == "ok":
                means.append(val)
            if done % 500 == 0 or done == len(files):
                el = time.time() - t0
                print(f"  {done}/{len(files)}  {el:.0f}s  eta {el/done*(len(files)-done):.0f}s", flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s | converted {len(means)} | failed {len(fails)}")
    if means:
        a = np.array(means)
        print(f"  per-image mean brightness (0-255): p5={np.percentile(a,5):.1f} "
              f"median={np.median(a):.1f} p95={np.percentile(a,95):.1f}")
        print("  (a normal photo collection sits roughly 90-130; far outside that "
              "means the tone curve needs revisiting)")
    with open(out_dir / "conversion_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"tmo": args.tmo, "percentile": args.percentile, "gamma": args.gamma,
                   "n": len(means), "failed": fails, "target": TARGET,
                   "aspect_preserved": False}, f, indent=1)


if __name__ == "__main__":
    main()
