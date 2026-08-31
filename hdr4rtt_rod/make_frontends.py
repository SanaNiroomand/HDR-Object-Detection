#!/usr/bin/env python3
"""
make_frontends.py -- build one image set per tone-mapping front end, all from the
same EXR source and identical in every other respect.

This is the controlled experiment the advisor asked for: hold the detector fixed
and change only what sits in front of it. Kocdemir's thesis Tables 4.2 and 4.3
show the answer flips with the detector (the learned joint method beats every
classical operator with RetinaNet at 31.6 vs 31.3, and loses to four of them with
Faster R-CNN at 27.7 vs Fattal 29.5), so the detector cannot be left free to vary.

Front ends produced here:

  hdr        linear values normalised by the 99.9th percentile, clipped, NO tone
             curve. The "HDR" row of the thesis tables -- worst performer there
             (26.3 / 23.5). Written as float16 because forcing it through 8 bits
             would apply exactly the compression the arm is meant to omit.
  gamma      same normalisation then sRGB-style gamma. The "HDR with gamma" row.
  reinhard   GLOBAL Reinhard: L/(1+L) after scaling to a target key, one curve
             applied uniformly. NOT the local dodging-and-burning variant the
             same paper also defines -- the thesis lists both separately
             (CityScapes: local 33.2, global 32.7) but does not say which its
             HDR4RTT table used, so treat that row as possibly a different
             operator when comparing.
  durand     bilateral base/detail separation in log luminance, base compressed,
             detail preserved. A local operator, unlike the three above.
  log        log1p compression.

The four tone-mapped arms are written as 8-bit PNG, which is what a tone-mapping
operator actually produces -- compressing to a displayable image is the point of
one. Only the `hdr` arm keeps float precision. That difference is deliberate and
is the honest representation of each method, not an oversight.

RAOD's learned Adaptive_Module is the sixth front end. It is not produced here
because it is trained jointly with the detector rather than applied beforehand;
see train_frontend.py.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import gzip
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2

cv2.setNumThreads(1)
TARGET = 1280
EPS = 1e-6


def _luminance(rgb):
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def _recolour(rgb, lum_in, lum_out, saturation=1.0):
    """Re-apply colour after a luminance-domain operator: (channel / luminance)
    scaled by the new luminance. saturation defaults to 1.0 (plain ratio) --
    an exponent below 1 desaturates, and combined with the gray-world balance
    already applied it produced a visible colour cast."""
    ratio = rgb / np.maximum(lum_in, EPS)[:, :, None]
    ratio = np.clip(ratio, 0, None)
    if saturation != 1.0:
        ratio = ratio ** saturation
    return ratio * lum_out[:, :, None]


def tonemap(rgb, kind, pct=99.9, gamma=2.2, key=0.18, base_contrast=4.0):
    """rgb: HxWx3 float32 linear, non-negative. Returns float32 in [0,1]."""
    ref = float(np.percentile(rgb, pct))
    if not np.isfinite(ref) or ref <= EPS:
        ref = max(float(rgb.max()), EPS)

    if kind == "hdr":
        return np.clip(rgb / ref, 0, 1)                    # no tone curve at all

    if kind == "gamma":
        out = np.clip(rgb / ref, 0, 1)

    elif kind == "log":
        out = np.log1p(np.clip(rgb, 0, ref)) / np.log1p(ref)

    elif kind == "reinhard":
        lum = _luminance(rgb)
        key_now = np.exp(np.mean(np.log(lum + EPS)))
        scaled = rgb * (key / max(key_now, EPS))
        out = scaled / (1.0 + scaled)

    elif kind == "durand":
        # Durand & Dorsey: split log luminance into a bilateral-filtered base and
        # a detail residual, compress only the base so local detail survives.
        lum = np.maximum(_luminance(rgb), EPS)
        log_lum = np.log10(lum)

        # The bilateral filter is the expensive part. Durand's own method
        # computes it at reduced resolution and upsamples; doing the same here
        # cuts this arm from ~1.9 s/image to well under 0.2 s with no visible
        # difference, since the base layer is by construction low-frequency.
        h, w = log_lum.shape
        small = cv2.resize(log_lum, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        sigma_space = max(2.0, 0.02 * np.hypot(h // 4, w // 4))
        base_s = cv2.bilateralFilter(small.astype(np.float32), d=-1,
                                     sigmaColor=0.4, sigmaSpace=sigma_space)
        base = cv2.resize(base_s, (w, h), interpolation=cv2.INTER_LINEAR)

        detail = log_lum - base
        # compress the base so its range spans log10(base_contrast); detail is
        # passed through untouched, which is the whole point of the operator
        lo, hi = np.percentile(base, [0.5, 99.5])
        scale = np.log10(base_contrast) / max(float(hi - lo), EPS)
        new_log = (base - hi) * scale + detail
        # Normalise on the COMBINED base+detail result, not on the base alone.
        # Offsetting by the base's top ignores the detail layer, which then
        # pushes textured regions past 1 and washes the frame out (measured mean
        # 0.74). Using a high percentile of the combined signal exposes the image
        # on what is actually there.
        #
        # Consequence worth knowing: a small, very bright source is
        # high-frequency, so it lands in the DETAIL layer and escapes the base
        # compression entirely. On such scenes (a welding arc) Durand leaves the
        # rest of the frame dark. That is the operator's real behaviour, not an
        # implementation fault, and it is part of what this comparison measures.
        new_log = new_log - float(np.percentile(new_log, 99.0))
        new_lum = np.power(10.0, new_log)
        out = np.clip(_recolour(rgb, lum, new_lum), 0, 1)
    else:
        raise ValueError(kind)

    out = np.clip(out, 0, 1) ** (1.0 / gamma)
    return np.clip(out, 0, 1).astype(np.float32)


def prepare(bgr):
    """Shared geometry and white balance -- identical across every arm."""
    img = bgr.astype(np.float32)
    img = np.clip(img, 0, None)
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
    mr, mg, mb = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    if mr > EPS:
        img[:, :, 0] *= mg / mr
    if mb > EPS:
        img[:, :, 2] *= mg / mb
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exr_dir", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database\images")
    p.add_argument("--file_list", default=r"D:\Data\HDR\hdr4rtt_voc20\annotations\images_to_convert.txt")
    p.add_argument("--out_root", default=r"D:\Data\HDR\hdr4rtt_voc20\frontends")
    p.add_argument("--arms", default="gamma,reinhard,durand,log,hdr")
    p.add_argument("--percentile", type=float, default=99.9)
    p.add_argument("--gamma", type=float, default=2.2)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    stems = [l.strip() for l in open(args.file_list, encoding="utf-8") if l.strip()]
    if args.limit:
        rng = np.random.default_rng(0)
        stems = [stems[i] for i in sorted(rng.choice(len(stems), min(args.limit, len(stems)),
                                                     replace=False))]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"{len(stems)} images x {len(arms)} front ends: {arms}")

    for arm in arms:
        out_dir = Path(args.out_root) / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        is_float = (arm == "hdr")
        t0, done, fails, stats = time.time(), 0, [], []

        def work(stem):
            outp = out_dir / (stem + (".npy.gz" if is_float else ".png"))
            if outp.exists():
                return ("skip", None)
            bgr = cv2.imread(os.path.join(args.exr_dir, stem + ".exr"),
                             cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
                return ("fail", stem)
            img = prepare(bgr[:, :, :3])
            out = tonemap(img, arm, args.percentile, args.gamma)
            if is_float:
                with gzip.GzipFile(str(outp), "w", compresslevel=1) as gz:
                    np.save(file=gz, arr=out.astype(np.float16))
            else:
                u8 = (out * 255.0 + 0.5).astype(np.uint8)
                cv2.imwrite(str(outp), u8[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return ("ok", float(out.mean()))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for status, val in ex.map(work, stems):
                done += 1
                if status == "fail":
                    fails.append(val)
                elif status == "ok":
                    stats.append(val)
                if done % 1000 == 0 or done == len(stems):
                    el = time.time() - t0
                    print(f"  [{arm}] {done}/{len(stems)}  {el:.0f}s", flush=True)

        sz = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file()) / 1e9
        m = np.mean(stats) if stats else float("nan")
        print(f"  [{arm}] done in {time.time()-t0:.0f}s | mean pixel {m:.4f} "
              f"| {sz:.1f} GB | failed {len(fails)}\n")
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "percentile": args.percentile, "gamma": args.gamma,
                       "float": is_float, "n": len(stats), "failed": fails,
                       "mean_pixel": None if np.isnan(m) else float(m)}, f, indent=1)


if __name__ == "__main__":
    main()
