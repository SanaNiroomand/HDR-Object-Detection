#!/usr/bin/env python3
"""
eval_torchvision.py -- score a torchvision detector on the tone-mapped (LDR)
HDR4RTT images, using the same annotations and splits as the RAOD arm.

Two modes:
  --weights coco    COCO-pretrained, zero-shot. Maps COCO person/car onto the
                    dataset's Pedestrian/Car categories. No training involved.
  --weights <path>  a checkpoint produced by train_torchvision.py.

Uses the existing hdr4rtt_rod_*_test*.json files unchanged: the LDR images share
the exact geometry of the .npy.gz ones (both squashed to 1280x1280 by the same
code path), so the boxes describe both without modification. Only the file
extension differs, which is rewritten on the fly.

Comparability note: mmdetection could not be used on this machine (mmcv needs
compiled CUDA ops; no CUDA toolkit or MSVC present, and its prebuilt wheels stop
at torch versions that predate this GPU's compute capability). torchvision's
detectors need no compilation and cover both a one-stage (RetinaNet, the family
TMO-Det used) and a two-stage (Faster R-CNN) architecture, which is what the
detector-architecture question actually needs.
"""
import os
import json
import copy
import argparse
import contextlib
import io as _io

import numpy as np
import torch
import cv2

COCO_PERSON, COCO_CAR = 1, 3
ROD_PEDESTRIAN, ROD_CAR = 1, 2
CLASS_NAMES = {1: "Pedestrian", 2: "Car"}


def build_model(kind, weights, num_classes=3, device="cuda"):
    import torchvision
    from torchvision.models.detection import (
        retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)

    coco = (weights == "coco")
    if kind == "retinanet":
        w = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if coco else None
        model = retinanet_resnet50_fpn_v2(weights=w,
                                          num_classes=91 if coco else num_classes)
    else:
        w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if coco else None
        model = fasterrcnn_resnet50_fpn_v2(weights=w,
                                           num_classes=91 if coco else num_classes)
    if not coco:
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
    return model.eval().to(device), coco


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--img_dir", default=r"D:\Data\HDR\hdr4rtt_ldr\images")
    p.add_argument("--arch", default="retinanet", choices=["retinanet", "fasterrcnn"])
    p.add_argument("--weights", default="coco")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--score_thr", type=float, default=0.01)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, is_coco = build_model(args.arch, args.weights, device=device)
    n_par = sum(x.numel() for x in model.parameters()) / 1e6
    print(f"{args.arch} | weights={args.weights} | {n_par:.1f}M params | {device}")

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(os.path.abspath(args.ann))

    present, missing = [], 0
    for img_id in sorted(coco.getImgIds()):
        fn = coco.loadImgs(img_id)[0]["file_name"]
        png = os.path.splitext(os.path.splitext(fn)[0])[0] + ".png"
        fp = os.path.join(args.img_dir, png)
        if os.path.exists(fp):
            present.append((img_id, fp))
        else:
            missing += 1
    print(f"{os.path.basename(args.ann)}: {len(present)} images present, {missing} missing")
    if not present:
        raise SystemExit("nothing to evaluate")

    gt = copy.deepcopy(coco.dataset)
    keep = {i for i, _ in present}
    gt["images"] = [im for im in gt["images"] if im["id"] in keep]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    print(f"ground truth: {len(gt['annotations'])} boxes")
    with contextlib.redirect_stdout(_io.StringIO()):
        sub = COCO(); sub.dataset = gt; sub.createIndex()

    dets = []
    for i in range(0, len(present), args.batch):
        chunk = present[i:i + args.batch]
        tens = []
        for _, fp in chunk:
            bgr = cv2.imread(fp)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tens.append(torch.from_numpy(rgb).permute(2, 0, 1).to(device))
        with torch.no_grad():
            outs = model(tens)
        for (img_id, _), o in zip(chunk, outs):
            for b, s, l in zip(o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(),
                               o["labels"].cpu().numpy()):
                if s < args.score_thr:
                    continue
                if is_coco:
                    cid = ROD_PEDESTRIAN if l == COCO_PERSON else (
                          ROD_CAR if l == COCO_CAR else None)
                    if cid is None:
                        continue
                else:
                    cid = int(l)          # trained head: 1=Pedestrian, 2=Car
                    if cid not in CLASS_NAMES:
                        continue
                x1, y1, x2, y2 = map(float, b)
                dets.append({"image_id": int(img_id), "category_id": cid,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
        if (i // args.batch) % 25 == 0:
            print(f"   {min(i+args.batch, len(present))}/{len(present)}", flush=True)

    print(f"\n{len(dets)} detections")
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
            per[sub.loadCats([cid])[0]["name"]] = float(np.mean(pr))
    print("   AP50 per class: " + "  ".join(f"{k}={v:.4f}" for k, v in per.items()))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({"arch": args.arch, "weights": args.weights, "ann": args.ann,
                       "n_images": len(present), "n_gt": len(gt["annotations"]),
                       "mAP": float(E.stats[0]), "AP50": float(E.stats[1]),
                       "AP_small": float(E.stats[3]), "AP_med": float(E.stats[4]),
                       "AP_large": float(E.stats[5]), "AR100": float(E.stats[8]),
                       "per_class_AP50": per, "n_dets": len(dets)}, f, indent=1)
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
