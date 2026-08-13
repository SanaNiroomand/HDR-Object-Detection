#!/usr/bin/env python3
"""
run_single.py -- run RAOD on one .npy.gz and report detections.

Replaces eval_single_image.py, which had a real bug: it fed the model [0,255]
data. RAOD's dataloader divides by 255 first (coco_raw.py:141, inside
load_image), so the network expects [0,1]. Its own docstring conceded the
preprocessing was "NOT verified against this project's actual ValTransformRaw".

The consequence was severe and silent: with [0,255] input the tone-mapping
module computes x**(1/gamma) on values up to 255, every pixel saturates past
1.0, clamp(0,1) flattens the frame to white, and the model returns nothing.
That is why RAOD's OWN sample (scripts/official_output/day-00018.npy.gz)
produced zero detections at conf=0.001 in cfg_small_HDR4RTT/val_single_log.txt
-- the earlier "HDR4RTT doesn't work" conclusions were measured through this bug.

To stop that recurring, this script does not reimplement the preprocessing. It
calls RAOD's own functions:
    load  : gzip + np.load + /255.0     (exactly coco_raw.py load_image)
    resize: r = min(H/h, W/w)           (exactly coco_raw.py load_resized_img)
    pad   : preproc_raw                 (imported from yolox.data.data_augment)
For 1280x1280 inputs the resize and pad are both no-ops, which is why RAOD's
own data never exposed the 114 pad value.

--gain multiplies the loaded array before /255, so the operating point can be
swept without re-converting the dataset.
"""
import os
import sys
import gzip
import glob
import json
import argparse
import importlib.util

import numpy as np
import torch

RAOD_DIR = r"D:\Codes\HDR\Sana\RAOD\RAOD"


def load_exp(cfg_path):
    sys.path.insert(0, RAOD_DIR)          # so `import models` and `yolox` resolve
    os.chdir(RAOD_DIR)                    # cfg uses relative output_dir
    spec = importlib.util.spec_from_file_location("cfg_mod", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def load_npygz(path, gain=1.0, white_point=255.0):
    """coco_raw.py load_image, with an optional gain applied before the /255."""
    with gzip.GzipFile(path, "r") as f:
        arr = np.load(f)
    arr = arr.astype(np.float32)
    raw_stats = (float(arr.min()), float(arr.mean()), float(arr.max()))
    if gain != 1.0:
        arr = np.clip(arr * gain, 0, white_point)
    arr = arr / 255.0                      # <-- the step eval_single_image.py omitted
    return arr, raw_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npygz", required=True, help="file, or a directory/glob for many")
    p.add_argument("--ckpt", default=os.path.join(RAOD_DIR, "pre-trained", "best-day_night.pth"))
    p.add_argument("--cfg", default=os.path.join(RAOD_DIR, "cfg_small.py"))
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--nms", type=float, default=0.65)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="summary only, no per-box lines")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    exp = load_exp(args.cfg)
    from yolox.data.data_augment import preproc_raw
    from yolox.utils import postprocess
    from yolox.data.datasets.imx490_classes import COCO_CLASSES as ROD6

    # preprocess_anno.py's live mapping is 5 classes; Tricycle is commented out.
    names = ["Pedestrian", "Car", "Cyclist", "Tram", "Truck"]
    assert exp.num_classes == len(names), f"cfg says {exp.num_classes} classes"

    exp.test_conf, exp.nmsthre = args.conf, args.nms
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    print(f"loaded {os.path.basename(args.ckpt)} (epoch {ckpt.get('start_epoch')}), "
          f"{exp.num_classes} classes, test_size={exp.test_size}, gain={args.gain}")

    if os.path.isdir(args.npygz):
        files = sorted(glob.glob(os.path.join(args.npygz, "*.npy.gz")))
    elif any(c in args.npygz for c in "*?"):
        files = sorted(glob.glob(args.npygz))
    else:
        files = [args.npygz]
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} file(s)\n")

    results, n_with, tot = [], 0, 0
    for fp in files:
        arr, raw = load_npygz(fp, args.gain)
        h, w = arr.shape[:2]
        r = min(exp.test_size[0] / h, exp.test_size[1] / w)
        if r != 1:
            import cv2
            arr = cv2.resize(arr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        padded, _ = preproc_raw(arr, exp.test_size)
        t = torch.from_numpy(padded).unsqueeze(0).float().cuda()

        with torch.no_grad():
            out = postprocess(model(t), exp.num_classes, exp.test_conf, exp.nmsthre,
                              class_agnostic=False)[0]

        dets = []
        if out is not None:
            for d in out.cpu():
                x1, y1, x2, y2, oc, cc, ci = d[:7]
                dets.append({"cls": names[int(ci)], "score": float(oc * cc),
                             "bbox": [round(float(x1 / r), 1), round(float(y1 / r), 1),
                                      round(float(x2 / r), 1), round(float(y2 / r), 1)]})
        tot += len(dets)
        n_with += bool(dets)
        results.append({"file": os.path.basename(fp), "n": len(dets),
                        "raw_min": raw[0], "raw_mean": raw[1], "raw_max": raw[2],
                        "model_in_mean": float(t.mean()), "dets": dets})
        if not args.quiet:
            print(f"{os.path.basename(fp)}")
            print(f"   stored [0,255]: min={raw[0]:.3f} mean={raw[1]:.3f} max={raw[2]:.3f}"
                  f"   -> model input mean={float(t.mean()):.4f}")
            print(f"   {len(dets)} detection(s) above conf={args.conf}")
            for d in sorted(dets, key=lambda x: -x["score"])[:8]:
                print(f"      {d['cls']:<11} {d['score']:.3f}  {d['bbox']}")

    print(f"\n=== {len(files)} images | {tot} detections | "
          f"{n_with} image(s) with >=1 ({n_with/max(len(files),1):.0%}) | "
          f"{tot/max(len(files),1):.2f} det/img ===")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"gain": args.gain, "conf": args.conf, "ckpt": args.ckpt,
                       "results": results}, f, indent=1)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
