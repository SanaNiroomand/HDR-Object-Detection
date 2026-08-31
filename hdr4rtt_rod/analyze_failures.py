#!/usr/bin/env python3
"""
analyze_failures.py -- where does the detector fail, and what do those cases
have in common?

Produces three things:

1. An error breakdown per predicted box, in the style of TIDE:
     TP    correct class, IoU >= 0.5
     Cls   right place, wrong class
     Loc   right class, 0.1 <= IoU < 0.5
     Dupe  a second box on an already-matched object
     Bkg   IoU < 0.1 with anything
   plus every ground-truth box that nothing matched (Miss).

2. Miss rate sliced by properties that might explain it. The interesting ones
   for HDR are measured from the ORIGINAL EXR rather than the tone-mapped copy:
     - how bright the object is RELATIVE to its own scene (a person in shadow
       beside a window sits far below the scene median)
     - the dynamic range inside the object's own box
     - the dynamic range of the whole scene
   alongside the ordinary suspects: object size, class, source group.

3. The worst images written out with ground truth and predictions drawn, so the
   slices can be checked by eye rather than trusted from a table.

The point is to find what the failures share. A miss rate that climbs smoothly
with scene dynamic range would say something quite different from one that
depends only on object size.
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import gzip
import json
import argparse
import contextlib
import io as _io
import collections

import numpy as np
import torch
import cv2

cv2.setNumThreads(1)

EXR_DIR = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database\images"
STATS_CSV = r"D:\Codes\HDR\Sana\hdr4rtt_analysis\hdr_stats_sources.csv"
FLOAT_ARMS = {"hdr", "tmm"}
ARM_DIR = {"tmm": "hdr"}
TARGET = 1280


def iou_matrix(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(ar_a[:, None] + ar_b[None, :] - inter, 1e-9)).astype(np.float32)


def box_hdr_stats(stem, boxes):
    """Per-box statistics from the ORIGINAL linear HDR file.

    Returns, per box: median luminance inside the box divided by the image's
    median luminance (so <1 means the object sits darker than its own scene),
    and the dynamic range inside the box in decades.
    """
    p = os.path.join(EXR_DIR, stem + ".exr")
    bgr = cv2.imread(p, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if bgr is None:
        return [(np.nan, np.nan)] * len(boxes)
    img = np.clip(bgr[:, :, :3].astype(np.float32), 0, None)[:, :, ::-1]
    img = cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)
    lum = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    scene_med = max(float(np.median(lum)), 1e-9)
    out = []
    for x1, y1, x2, y2 in boxes:
        xa, ya = int(max(0, x1)), int(max(0, y1))
        xb, yb = int(min(TARGET, x2)), int(min(TARGET, y2))
        if xb - xa < 2 or yb - ya < 2:
            out.append((np.nan, np.nan)); continue
        crop = lum[ya:yb, xa:xb]
        rel = float(np.median(crop)) / scene_med
        pos = crop[crop > 0]
        dr = (np.log10(np.percentile(pos, 99.5) / max(np.percentile(pos, 0.5), 1e-9))
              if pos.size > 20 else np.nan)
        out.append((rel, float(dr)))
    return out


def bucket_report(name, values, missed, edges, labels):
    """Miss rate within each bucket of `values`."""
    values = np.asarray(values, float)
    missed = np.asarray(missed, bool)
    ok = np.isfinite(values)
    print(f"\n  {name}")
    idx = np.digitize(values[ok], edges)
    m = missed[ok]
    for i, lab in enumerate(labels):
        sel = idx == i
        if sel.sum() < 10:
            continue
        print(f"    {lab:<22} n={sel.sum():5}   missed {m[sel].mean():6.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="reinhard")
    p.add_argument("--arch", default="retinanet")
    p.add_argument("--tag", default="dedup")
    p.add_argument("--ann", default=r"D:\Data\HDR\hdr4rtt_voc20\annotations\hdr4rtt_voc20_dedup_test.json")
    p.add_argument("--frontend_root", default=r"D:\Data\HDR\hdr4rtt_voc20\frontends")
    p.add_argument("--run_root", default=r"D:\Codes\HDR\Sana\hdr4rtt_rod\frontend_runs")
    p.add_argument("--score_thr", type=float, default=0.3,
                   help="a detection below this is not something a user would act on")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--viz_n", type=int, default=8)
    p.add_argument("--out_dir", default=r"D:\Codes\HDR\Sana\hdr4rtt_rod\failures")
    args = p.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_frontend import build_detector, TmmWrapper
    from pycocotools.coco import COCO
    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_float = args.arm in FLOAT_ARMS
    img_dir = os.path.join(args.frontend_root, ARM_DIR.get(args.arm, args.arm))
    run = f"{args.arch}_{args.arm}" + (f"_{args.tag}" if args.tag else "")
    ckpt = os.path.join(args.run_root, run, "last.pth")

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(args.ann)
    names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    det = build_detector(args.arch, sd["n_classes"])
    model = TmmWrapper(det, init_ckpt=None, input_gain=sd.get("input_gain", 1.0)) \
        if args.arm == "tmm" else det
    model.load_state_dict(sd["model"]); model.eval().to(device)
    print(f"{run}, score threshold {args.score_thr}\n")

    src = {r.stem: r.source.split(":")[0] for r in pd.read_csv(STATS_CSV).itertuples()}
    scene_dr = {r.stem: r.dr_decades for r in pd.read_csv(STATS_CSV).itertuples()}

    ext = ".npy.gz" if is_float else ".png"
    ids = sorted(coco.getImgIds())
    gt_rows, det_rows, per_image = [], [], []

    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        tens, metas = [], []
        for img_id in chunk:
            info = coco.loadImgs(img_id)[0]
            stem = os.path.splitext(os.path.splitext(info["file_name"])[0])[0]
            fp = os.path.join(img_dir, stem + ext)
            if not os.path.exists(fp):
                continue
            if is_float:
                with gzip.GzipFile(fp, "r") as f:
                    rgb = np.load(f).astype(np.float32)
            else:
                rgb = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tens.append(torch.from_numpy(rgb).permute(2, 0, 1).to(device))
            metas.append((img_id, stem))
        if not tens:
            continue
        with torch.no_grad():
            outs = model(tens)

        for (img_id, stem), o in zip(metas, outs):
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False))
            g = np.array([[a["bbox"][0], a["bbox"][1],
                           a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                          for a in anns], np.float32).reshape(-1, 4)
            gl = np.array([a["category_id"] for a in anns], int)
            keep = o["scores"].cpu().numpy() >= args.score_thr
            d = o["boxes"].cpu().numpy()[keep].reshape(-1, 4)
            dl = o["labels"].cpu().numpy()[keep]
            ds = o["scores"].cpu().numpy()[keep]

            M = iou_matrix(d, g)
            gt_taken = np.zeros(len(g), bool)
            gt_hit = np.zeros(len(g), bool)
            order = np.argsort(-ds)
            for k in order:
                if M.shape[1] == 0:
                    det_rows.append((stem, "Bkg")); continue
                j = int(np.argmax(M[k]))
                v = M[k, j]
                if v >= 0.5 and dl[k] == gl[j]:
                    if gt_taken[j]:
                        det_rows.append((stem, "Dupe"))
                    else:
                        gt_taken[j] = gt_hit[j] = True
                        det_rows.append((stem, "TP"))
                elif v >= 0.5:
                    det_rows.append((stem, "Cls"))
                elif v >= 0.1 and dl[k] == gl[j]:
                    det_rows.append((stem, "Loc"))
                else:
                    det_rows.append((stem, "Bkg"))

            hdr = box_hdr_stats(stem, g) if len(g) else []
            for j in range(len(g)):
                w, h = g[j, 2] - g[j, 0], g[j, 3] - g[j, 1]
                gt_rows.append(dict(
                    stem=stem, cls=names.get(int(gl[j]), "?"), source=src.get(stem, "?"),
                    missed=not gt_hit[j], size=float(np.sqrt(w * h) / TARGET * 100),
                    rel_lum=hdr[j][0], box_dr=hdr[j][1],
                    scene_dr=scene_dr.get(stem, np.nan)))
            per_image.append((stem, int((~gt_hit).sum()), len(g)))
        if (i // args.batch) % 60 == 0:
            print(f"   {min(i+args.batch,len(ids))}/{len(ids)}", flush=True)

    df = pd.DataFrame(gt_rows)
    dd = pd.DataFrame(det_rows, columns=["stem", "kind"])
    df.to_csv(os.path.join(args.out_dir, f"{run}_gt_boxes.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"ERROR BREAKDOWN  ({len(dd)} predictions above {args.score_thr}, "
          f"{len(df)} ground-truth boxes)")
    print("=" * 70)
    for k, n in dd["kind"].value_counts().items():
        print(f"  {k:<6} {n:6}  {n/len(dd):6.1%}")
    print(f"  {'Miss':<6} {int(df['missed'].sum()):6}  "
          f"{df['missed'].mean():6.1%} of ground truth")

    print("\n" + "=" * 70)
    print("WHAT DO THE MISSES HAVE IN COMMON?")
    print("=" * 70)

    bucket_report("object size (% of image side)", df["size"], df["missed"],
                  [2, 5, 10, 20], ["<2 (tiny)", "2-5", "5-10", "10-20", ">20 (large)"])
    bucket_report("object brightness relative to its own scene", df["rel_lum"], df["missed"],
                  [0.25, 0.5, 1.0, 2.0, 5.0],
                  ["<0.25x (deep shadow)", "0.25-0.5x", "0.5-1x", "1-2x",
                   "2-5x", ">5x (bright)"])
    bucket_report("dynamic range inside the object's box (decades)", df["box_dr"],
                  df["missed"], [1, 2, 3], ["<1", "1-2", "2-3", ">3"])
    bucket_report("dynamic range of the whole scene (decades)", df["scene_dr"],
                  df["missed"], [3, 4, 5], ["<3", "3-4", "4-5", ">5"])

    print("\n  by source")
    for s, g in df.groupby("source"):
        print(f"    {s:<22} n={len(g):5}   missed {g['missed'].mean():6.1%}")

    print("\n  by class (10 or more instances)")
    for c, g in sorted(df.groupby("cls"), key=lambda kv: -kv[1]["missed"].mean()):
        if len(g) >= 10:
            print(f"    {c:<22} n={len(g):5}   missed {g['missed'].mean():6.1%}")

    worst = sorted(per_image, key=lambda r: (-r[1], -r[2]))[:args.viz_n]
    print("\n  worst images (most missed objects)")
    for stem, miss, tot in worst:
        print(f"    {stem}  missed {miss}/{tot}")
    with open(os.path.join(args.out_dir, f"{run}_worst.txt"), "w") as f:
        f.write("\n".join(s for s, _, _ in worst))
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
