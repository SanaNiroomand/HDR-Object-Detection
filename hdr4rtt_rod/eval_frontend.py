#!/usr/bin/env python3
"""
eval_frontend.py -- COCO mAP for a detector trained by train_frontend.py.

Mirrors that script's data handling exactly, so an arm is evaluated on the same
representation it was trained on: 8-bit images for the classical operators,
float linear for `hdr`, and float linear through the learned module for `tmm`.

Reports the same summary the thesis tables use (COCO mAP and AP50) plus
per-class AP50, so results drop straight into the format of Table 4.2.
"""
import os
import gzip
import json
import copy
import argparse
import contextlib
import io as _io

import numpy as np
import torch
import cv2

FRONTEND_ROOT = r"D:\Data\HDR\hdr4rtt_voc20\frontends"
ANN_DIR = r"D:\Data\HDR\hdr4rtt_voc20\annotations"
FLOAT_ARMS = {"hdr", "tmm"}
ARM_DIR = {"tmm": "hdr"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["gamma", "reinhard", "durand", "log", "hdr", "tmm"])
    p.add_argument("--arch", default="retinanet", choices=["retinanet", "fasterrcnn"])
    p.add_argument("--ckpt", default=None)
    p.add_argument("--tag", default="", help="variant suffix used at training time")
    p.add_argument("--frontend_root", default=FRONTEND_ROOT)
    p.add_argument("--ann_dir", default=ANN_DIR)
    p.add_argument("--ann", default=None,
                   help="explicit annotation file; overrides --ann_dir. Lets a "
                        "per-source subset be scored rather than the whole test set.")
    p.add_argument("--run_root", default=r"D:\Codes\HDR\Sana\hdr4rtt_rod\frontend_runs")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--score_thr", type=float, default=0.01)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from train_frontend import build_detector, TmmWrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_float = args.arm in FLOAT_ARMS
    img_dir = os.path.join(args.frontend_root, ARM_DIR.get(args.arm, args.arm))
    run_name = f"{args.arch}_{args.arm}" + (f"_{args.tag}" if args.tag else "")
    ckpt = args.ckpt or os.path.join(args.run_root, run_name, "last.pth")

    ann = args.ann or os.path.join(args.ann_dir, "hdr4rtt_voc20_test.json")
    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(ann)

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    n_classes = sd.get("n_classes")
    _r = sd.get("anchor_ratios", "")
    det = build_detector(args.arch, n_classes,
                         sd.get("min_size", 800), sd.get("max_size", 1333),
                         sd.get("anchor_scale", 1.0),
                         [float(v) for v in _r.split(",")] if _r else None)
    if args.arm == "tmm":
        # input_gain is part of the trained configuration, not a free choice at
        # test time: evaluating at a different scale than training would measure
        # a model that was never trained.
        model = TmmWrapper(det, init_ckpt=None, input_gain=sd.get("input_gain", 1.0))
    else:
        model = det
    model.load_state_dict(sd["model"])
    model.eval().to(device)
    print(f"  detector resize: short side {sd.get('min_size', 800)}, long side at most {sd.get('max_size', 1333)}")
    print(f"arm={args.arm} arch={args.arch} epoch={sd.get('epoch')} classes={n_classes-1}"
          + (f" input_gain={sd.get('input_gain', 1.0)} "
             f"lr_mult={sd.get('tmm_lr_mult')} "
             f"init={'random' if sd.get('tmm_random_init') else 'RAOD'}"
             if args.arm == "tmm" else ""))

    ext = ".npy.gz" if is_float else ".png"
    present = []
    for img_id in sorted(coco.getImgIds()):
        stem = os.path.splitext(os.path.splitext(
            coco.loadImgs(img_id)[0]["file_name"])[0])[0]
        fp = os.path.join(img_dir, stem + ext)
        if os.path.exists(fp):
            present.append((img_id, fp))
    print(f"{len(present)} test images present in {img_dir}")

    gt = copy.deepcopy(coco.dataset)
    keep = {i for i, _ in present}
    gt["images"] = [im for im in gt["images"] if im["id"] in keep]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    with contextlib.redirect_stdout(_io.StringIO()):
        sub = COCO(); sub.dataset = gt; sub.createIndex()
    print(f"{len(gt['annotations'])} ground-truth boxes")

    dets = []
    for i in range(0, len(present), args.batch):
        chunk = present[i:i + args.batch]
        tens = []
        for _, fp in chunk:
            if is_float:
                with gzip.GzipFile(fp, "r") as f:
                    rgb = np.load(f).astype(np.float32)
            else:
                rgb = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tens.append(torch.from_numpy(rgb).permute(2, 0, 1).to(device))
        with torch.no_grad():
            outs = model(tens)
        for (img_id, _), o in zip(chunk, outs):
            for b, s, l in zip(o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(),
                               o["labels"].cpu().numpy()):
                if s < args.score_thr:
                    continue
                x1, y1, x2, y2 = map(float, b)
                dets.append({"image_id": int(img_id), "category_id": int(l),
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        if (i // args.batch) % 40 == 0:
            print(f"   {min(i+args.batch, len(present))}/{len(present)}", flush=True)

    print(f"{len(dets)} detections")
    if not dets:
        raise SystemExit("no detections")
    with contextlib.redirect_stdout(_io.StringIO()):
        dt = sub.loadRes(copy.deepcopy(dets))
        E = COCOeval(sub, dt, "bbox")
        E.evaluate(); E.accumulate()
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        E.summarize()
    print(buf.getvalue().rstrip())

    cat_ids = sorted(sub.getCatIds())
    prec = E.eval["precision"]
    per = {}
    for ci, cid in enumerate(cat_ids):
        pr = prec[0, :, ci, 0, 2]
        pr = pr[pr > -1]
        if pr.size:
            per[sub.loadCats([cid])[0]["name"]] = round(float(np.mean(pr)), 4)
    print("   per-class AP50: " + "  ".join(f"{k}={v:.3f}" for k, v in
                                            sorted(per.items(), key=lambda kv: -kv[1])))

    out = args.out_json or os.path.join(args.run_root, run_name, "eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"arm": args.arm, "arch": args.arch, "ckpt": ckpt,
                   "n_images": len(present), "n_gt": len(gt["annotations"]),
                   "mAP": float(E.stats[0]), "AP50": float(E.stats[1]),
                   "AP75": float(E.stats[2]), "AP_small": float(E.stats[3]),
                   "AP_med": float(E.stats[4]), "AP_large": float(E.stats[5]),
                   "AR100": float(E.stats[8]), "per_class_AP50": per,
                   "n_dets": len(dets)}, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
