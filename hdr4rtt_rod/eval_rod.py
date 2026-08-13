#!/usr/bin/env python3
"""
eval_rod.py -- COCO mAP for RAOD on converted HDR4RTT (or on ROD itself).

Uses RAOD's own preprocessing path (see run_single.py for why that matters):
load .npy.gz -> /255 -> resize r=min(H/h,W/w) -> preproc_raw pad. Our converted
images are already 1280x1280, so resize and pad are both no-ops and predicted
boxes land directly in the annotation coordinate space -- the same space
build_rod_annotations.py squashes the ground truth into.

Only images whose .npy.gz actually exists are evaluated; the COCO ground truth
is subset to match, so partial conversions give an honest number over what was
converted rather than counting missing files as misses.

--gains takes a comma-separated list and evaluates each in one run (the model is
loaded once). Gain scales the stored array before the /255, which is how the
input operating point is matched to what the network was trained on without
re-converting the dataset.
"""
import os
import sys
import gzip
import json
import time
import copy
import argparse
import contextlib
import io as _io

import numpy as np
import torch

RAOD_DIR = r"D:\Codes\HDR\Sana\RAOD\RAOD"
CLASS_NAMES = ["Pedestrian", "Car", "Cyclist", "Tram", "Truck"]


def load_exp(cfg_path):
    import importlib.util
    sys.path.insert(0, RAOD_DIR)
    os.chdir(RAOD_DIR)
    spec = importlib.util.spec_from_file_location("cfg_mod", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Exp()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True, help="COCO json from build_rod_annotations.py")
    p.add_argument("--img_dir", required=True, help="directory of .npy.gz")
    p.add_argument("--ckpt", default=os.path.join(RAOD_DIR, "pre-trained", "best-day_night.pth"))
    p.add_argument("--cfg", default=os.path.join(RAOD_DIR, "cfg_small.py"))
    p.add_argument("--gains", default="1.0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--nms", type=float, default=0.65)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    ann_path = os.path.abspath(args.ann)
    img_dir = os.path.abspath(args.img_dir)
    out_json = os.path.abspath(args.out_json) if args.out_json else None

    exp = load_exp(args.cfg)
    from yolox.data.data_augment import preproc_raw
    from yolox.utils import postprocess
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    exp.test_conf, exp.nmsthre = args.conf, args.nms
    model = exp.get_model()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.cuda().eval()

    with contextlib.redirect_stdout(_io.StringIO()):
        coco = COCO(ann_path)
    cat_ids = sorted(coco.getCatIds())
    id2name = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}

    present, missing = [], 0
    for img_id in sorted(coco.getImgIds()):
        fn = coco.loadImgs(img_id)[0]["file_name"]
        fp = os.path.join(img_dir, fn)
        if os.path.exists(fp):
            present.append((img_id, fp))
        else:
            missing += 1
    print(f"{os.path.basename(ann_path)}: {len(present)} images present, {missing} missing")
    if not present:
        raise SystemExit("nothing to evaluate")

    # subset ground truth to what we actually have
    gt = copy.deepcopy(coco.dataset)
    keep = {i for i, _ in present}
    gt["images"] = [im for im in gt["images"] if im["id"] in keep]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep]
    print(f"evaluating against {len(gt['annotations'])} ground-truth boxes")
    with contextlib.redirect_stdout(_io.StringIO()):
        coco_sub = COCO()
        coco_sub.dataset = gt
        coco_sub.createIndex()

    cache = {}

    def load(fp):
        if fp not in cache:
            with gzip.GzipFile(fp, "r") as f:
                cache[fp] = np.load(f).astype(np.float32)
        return cache[fp]

    summary = {}
    for gain in [float(g) for g in args.gains.split(",")]:
        t0 = time.time()
        dets = []
        for i in range(0, len(present), args.batch):
            chunk = present[i:i + args.batch]
            tens = []
            for _, fp in chunk:
                arr = load(fp)
                if gain != 1.0:
                    arr = arr * gain
                arr = np.clip(arr, 0, 255.0) / 255.0
                padded, _ = preproc_raw(arr, exp.test_size)
                tens.append(torch.from_numpy(padded))
            batch = torch.stack(tens).float().cuda()
            with torch.no_grad():
                outs = postprocess(model(batch), exp.num_classes, exp.test_conf,
                                   exp.nmsthre, class_agnostic=False)
            for (img_id, _), out in zip(chunk, outs):
                if out is None:
                    continue
                for d in out.cpu().numpy():
                    x1, y1, x2, y2, oc, cc, ci = d[:7]
                    dets.append({"image_id": int(img_id),
                                 "category_id": int(cat_ids[int(ci)]),
                                 "bbox": [float(x1), float(y1),
                                          float(x2 - x1), float(y2 - y1)],
                                 "score": float(oc * cc)})
            if (i // args.batch) % 20 == 0:
                print(f"   gain={gain:<8g} {min(i+args.batch,len(present))}/{len(present)}",
                      flush=True)

        print(f"\n=== gain={gain:g} | {len(dets)} raw detections | "
              f"{time.time()-t0:.0f}s ===")
        if not dets:
            print("   no detections at all")
            summary[gain] = {"mAP": 0.0, "AP50": 0.0, "n_dets": 0}
            continue
        with contextlib.redirect_stdout(_io.StringIO()):
            dt = coco_sub.loadRes(copy.deepcopy(dets))
            E = COCOeval(coco_sub, dt, "bbox")
            E.evaluate(); E.accumulate()
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            E.summarize()
        print(buf.getvalue().rstrip())
        s = E.stats
        # per-class AP50
        per_cls = {}
        prec = E.eval["precision"]      # [iou, rec, cls, area, maxdet]
        for ci, cid in enumerate(cat_ids):
            pr = prec[0, :, ci, 0, 2]   # IoU=0.50, area=all, maxDet=100
            pr = pr[pr > -1]
            per_cls[id2name[cid]] = float(np.mean(pr)) if pr.size else float("nan")
        shown = {k: v for k, v in per_cls.items() if not np.isnan(v)}
        print("   AP50 per class: " +
              "  ".join(f"{k}={v:.4f}" for k, v in shown.items()))
        summary[gain] = {"mAP": float(s[0]), "AP50": float(s[1]),
                         "AP_small": float(s[3]), "AP_med": float(s[4]),
                         "AP_large": float(s[5]), "AR100": float(s[8]),
                         "n_dets": len(dets), "per_class_AP50": per_cls}

    print("\n" + "=" * 70)
    print(f"{'gain':>10} {'mAP':>9} {'AP50':>9} {'AR100':>9} {'dets':>9}")
    print("=" * 70)
    for g, v in sorted(summary.items()):
        print(f"{g:>10g} {v['mAP']:>9.4f} {v['AP50']:>9.4f} "
              f"{v.get('AR100', 0):>9.4f} {v['n_dets']:>9}")
    best = max(summary, key=lambda g: summary[g]["mAP"])
    print(f"\nbest gain by mAP: {best:g}  (mAP={summary[best]['mAP']:.4f}, "
          f"AP50={summary[best]['AP50']:.4f})")

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"ann": ann_path, "img_dir": img_dir, "ckpt": args.ckpt,
                       "conf": args.conf, "n_images": len(present),
                       "n_gt": len(gt["annotations"]), "summary": summary}, f, indent=1)
        print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
