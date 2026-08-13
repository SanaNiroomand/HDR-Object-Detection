#!/usr/bin/env python3
"""
pick_tmo.py -- choose the tone-mapping operator for the LDR baseline arm, by
measurement rather than by eye.

Why this exists: row 2 of the detector comparison is "big detector, ordinary
tone mapping". If a deliberately poor tone map is chosen, row 2 loses by
construction and the whole comparison is rigged. TMO-Det avoids this by
comparing six operators and reporting the best (their Table 2: classical TMOs
range 28.2 to 31.3 mAP -- a 3-point spread, so the choice genuinely matters).

Training a detector per operator would cost hours each. Instead this scores each
candidate with a COCO-pretrained detector, zero-shot. That is a fair proxy
precisely because COCO-pretrained weights encode "what ordinary photographs look
like" -- the tone map that best matches that expectation is the one a normal
detector is best equipped to consume.

COCO label ids used: person=1, car=3. These are mapped onto the dataset's ROD
category ids (Pedestrian=1, Car=2) so the existing ground truth can be reused
unchanged.
"""
import os
# must precede `import cv2`: OpenCV initialises the EXR codec on first use and
# reads this flag then, so setting it later leaves the codec disabled.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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


def build_subset_gt(full_ann, stems, out_path):
    """COCO json restricted to `stems`, with file_name pointing at .png."""
    d = json.load(open(full_ann, encoding="utf-8"))
    keep_ids, images = set(), []
    for im in d["images"]:
        stem = os.path.splitext(os.path.splitext(im["file_name"])[0])[0]
        if stem in stems:
            keep_ids.add(im["id"])
            im = dict(im)
            im["file_name"] = stem + ".png"
            images.append(im)
    sub = {"info": "", "license": [""], "categories": d["categories"], "images": images,
           "annotations": [a for a in d["annotations"] if a["image_id"] in keep_ids]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sub, f)
    return sub


def evaluate(model, img_dir, gt_path, device, batch=4, score_thr=0.01):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(gt_path)
    ids = sorted(coco.getImgIds())
    dets = []
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        tens = []
        for img_id in chunk:
            fn = coco.loadImgs(img_id)[0]["file_name"]
            bgr = cv2.imread(os.path.join(img_dir, fn))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tens.append(torch.from_numpy(rgb).permute(2, 0, 1).to(device))
        with torch.no_grad():
            outs = model(tens)
        for img_id, o in zip(chunk, outs):
            boxes = o["boxes"].cpu().numpy()
            scores = o["scores"].cpu().numpy()
            labels = o["labels"].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                if s < score_thr:
                    continue
                if l == COCO_PERSON:
                    cid = ROD_PEDESTRIAN
                elif l == COCO_CAR:
                    cid = ROD_CAR
                else:
                    continue
                x1, y1, x2, y2 = map(float, b)
                dets.append({"image_id": int(img_id), "category_id": cid,
                             "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    if not dets:
        return {"mAP": 0.0, "AP50": 0.0, "per_class": {}, "n_dets": 0}
    with contextlib.redirect_stdout(_io.StringIO()):
        dt = coco.loadRes(copy.deepcopy(dets))
        E = COCOeval(coco, dt, "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    cat_ids = sorted(coco.getCatIds())
    prec = E.eval["precision"]
    per = {}
    for ci, cid in enumerate(cat_ids):
        pr = prec[0, :, ci, 0, 2]
        pr = pr[pr > -1]
        if pr.size:
            per[coco.loadCats([cid])[0]["name"]] = float(np.mean(pr))
    return {"mAP": float(E.stats[0]), "AP50": float(E.stats[1]),
            "per_class": per, "n_dets": len(dets)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exr_dir", default=r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database\images")
    p.add_argument("--stem_list", default=r"D:\Data\HDR\hdr4rtt_ldr\random_probe_list.txt")
    p.add_argument("--full_ann", default=r"D:\Data\HDR\hdr4rtt_rod\annotations\hdr4rtt_rod_seqsafe_test.json")
    p.add_argument("--work_dir", default=r"D:\Data\HDR\hdr4rtt_ldr\tmo_sweep")
    p.add_argument("--detector", default="retinanet",
                   choices=["retinanet", "fasterrcnn"])
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args()

    import torchvision
    from convert_hdr4rtt_to_ldr import convert_one
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(args.work_dir, exist_ok=True)
    stems = [l.strip() for l in open(args.stem_list, encoding="utf-8") if l.strip()]

    # ground truth: only images that carry boxes in the chosen annotation file
    gt_path = os.path.join(args.work_dir, "gt.json")
    sub = build_subset_gt(args.full_ann, set(stems), gt_path)
    have = {os.path.splitext(im["file_name"])[0] for im in sub["images"]}
    stems = [s for s in stems if s in have]
    print(f"{len(stems)} probe images carry ground truth "
          f"({len(sub['annotations'])} boxes)\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.detector == "retinanet":
        w = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=w)
    else:
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w)
    model.eval().to(device)
    n_par = sum(x.numel() for x in model.parameters()) / 1e6
    print(f"detector: {args.detector} (COCO-pretrained, {n_par:.1f}M params) on {device}\n")

    candidates = [
        ("gamma",    dict(tmo="gamma",    pct=99.9, gamma=2.2)),
        ("gamma",    dict(tmo="gamma",    pct=99.5, gamma=2.2)),
        ("gamma",    dict(tmo="gamma",    pct=99.0, gamma=2.2)),
        ("gamma",    dict(tmo="gamma",    pct=95.0, gamma=2.2)),
        ("gamma",    dict(tmo="gamma",    pct=90.0, gamma=2.2)),
        ("reinhard", dict(tmo="reinhard", pct=99.5, gamma=2.2)),
        ("log",      dict(tmo="log",      pct=99.5, gamma=2.2)),
        ("log",      dict(tmo="log",      pct=99.9, gamma=2.2)),
    ]

    results = []
    for name, cfg in candidates:
        tag = f"{cfg['tmo']}_p{cfg['pct']:g}_g{cfg['gamma']:g}"
        img_dir = os.path.join(args.work_dir, tag)
        os.makedirs(img_dir, exist_ok=True)

        def work(stem):
            outp = os.path.join(img_dir, stem + ".png")
            if os.path.exists(outp):
                return
            bgr = cv2.imread(os.path.join(args.exr_dir, stem + ".exr"),
                             cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            u8 = convert_one(bgr[:, :, :3], cfg["tmo"], cfg["pct"], cfg["gamma"])
            cv2.imwrite(outp, u8[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, stems))

        r = evaluate(model, img_dir, gt_path, device)
        mean_b = np.mean([cv2.imread(os.path.join(img_dir, s + ".png")).mean()
                          for s in stems[:40]])
        r.update(tag=tag, mean_brightness=float(mean_b))
        results.append(r)
        print(f"{tag:24} mAP={r['mAP']:.4f}  AP50={r['AP50']:.4f}  "
              f"mean={mean_b:5.1f}  " +
              "  ".join(f"{k}={v:.3f}" for k, v in r["per_class"].items()))

    print("\n" + "=" * 72)
    best = max(results, key=lambda r: r["mAP"])
    worst = min(results, key=lambda r: r["mAP"])
    print(f"BEST : {best['tag']}   mAP={best['mAP']:.4f}  AP50={best['AP50']:.4f}")
    print(f"WORST: {worst['tag']}   mAP={worst['mAP']:.4f}")
    spread = best["mAP"] - worst["mAP"]
    print(f"spread across operators: {spread:.4f} mAP "
          f"({spread/max(worst['mAP'],1e-9)*100:.0f}% relative)")
    print("\n-> use the BEST operator for the row-2 training run, so the LDR arm is")
    print("   represented at its strongest and the comparison is not rigged.")
    with open(os.path.join(args.work_dir, "tmo_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"detector": args.detector, "results": results, "best": best["tag"]}, f, indent=1)


if __name__ == "__main__":
    main()
