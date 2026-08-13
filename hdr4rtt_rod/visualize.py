#!/usr/bin/env python3
"""
visualize.py -- draw ground truth and predictions on converted .npy.gz images.

Primary purpose is verifying the coordinate contract, not making pretty
pictures. build_rod_annotations.py squashes boxes anisotropically (x by
1280/orig_w, y by 1280/orig_h) to match the converter's non-aspect-preserving
resize, mirroring RAOD's own preprocess_anno.py. If that scaling were wrong,
ground-truth boxes would sit visibly off their objects here.

Displayed pixels are tone-mapped for human viewing ONLY (the model sees the
linear data). The tone map is the same curve the network applies internally,
x**(1/7) after /255, so what you see approximates what the network sees.
"""
import os
import sys
import gzip
import json
import argparse
import numpy as np
import cv2

RAOD_DIR = r"D:\Codes\HDR\Sana\RAOD\RAOD"
CLASS_NAMES = ["Pedestrian", "Car", "Cyclist", "Tram", "Truck"]
GT_COLOR = (60, 220, 60)
PRED_COLOR = (60, 140, 255)


def tonemap(arr, gain=1.0, gamma=7.0):
    x = np.clip(arr.astype(np.float32) * gain, 0, 255) / 255.0
    x = np.clip(x, 0, 1) ** (1.0 / gamma)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def draw(img, boxes, color, label_key, score_key=None, thick=2):
    for b in boxes:
        x, y, w, h = b["bbox"]
        p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
        cv2.rectangle(img, p1, p2, color, thick)
        txt = b[label_key]
        if score_key and score_key in b:
            txt += f" {b[score_key]:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), color, -1)
        cv2.putText(img, txt, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", default=os.path.join(RAOD_DIR, "pre-trained", "best-day_night.pth"))
    p.add_argument("--cfg", default=os.path.join(RAOD_DIR, "cfg_small.py"))
    p.add_argument("--gain", type=float, default=0.02)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--no_pred", action="store_true", help="ground truth only (no model needed)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ann_path, img_dir = os.path.abspath(args.ann), os.path.abspath(args.img_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(ann_path, encoding="utf-8") as f:
        d = json.load(f)
    id2name = {c["id"]: c["name"] for c in d["categories"]}
    gt_by_img = {}
    for a in d["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(
            {"bbox": a["bbox"], "name": id2name[a["category_id"]]})
    imgs = [im for im in d["images"]
            if os.path.exists(os.path.join(img_dir, im["file_name"])) and gt_by_img.get(im["id"])]
    if not imgs:
        raise SystemExit("no images with ground truth found in img_dir")
    rng = np.random.default_rng(args.seed)
    imgs = [imgs[i] for i in rng.choice(len(imgs), min(args.n, len(imgs)), replace=False)]
    print(f"{len(imgs)} image(s) -> {out_dir}")

    model = exp = preproc_raw = postprocess = torch = None
    if not args.no_pred:
        import importlib.util, torch as _torch
        torch = _torch
        sys.path.insert(0, RAOD_DIR); os.chdir(RAOD_DIR)
        spec = importlib.util.spec_from_file_location("cfg_mod", args.cfg)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        exp = mod.Exp()
        from yolox.data.data_augment import preproc_raw as _pr
        from yolox.utils import postprocess as _pp
        preproc_raw, postprocess = _pr, _pp
        exp.test_conf, exp.nmsthre = args.conf, 0.65
        model = exp.get_model()
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); model.cuda().eval()

    for im in imgs:
        fp = os.path.join(img_dir, im["file_name"])
        with gzip.GzipFile(fp, "r") as f:
            arr = np.load(f).astype(np.float32)

        vis = tonemap(arr, args.gain)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # stored RGB -> cv2 BGR for writing

        preds = []
        if model is not None:
            x = np.clip(arr * args.gain, 0, 255) / 255.0
            padded, _ = preproc_raw(x, exp.test_size)
            t = torch.from_numpy(padded).unsqueeze(0).float().cuda()
            with torch.no_grad():
                out = postprocess(model(t), exp.num_classes, exp.test_conf,
                                  exp.nmsthre, class_agnostic=False)[0]
            if out is not None:
                for det in out.cpu().numpy():
                    x1, y1, x2, y2, oc, cc, ci = det[:7]
                    preds.append({"bbox": [float(x1), float(y1),
                                           float(x2 - x1), float(y2 - y1)],
                                  "name": CLASS_NAMES[int(ci)], "score": float(oc * cc)})

        draw(vis, gt_by_img[im["id"]], GT_COLOR, "name")
        draw(vis, preds, PRED_COLOR, "name", "score")
        cv2.putText(vis, f"green=GT ({len(gt_by_img[im['id']])})   "
                         f"blue=pred>{args.conf} ({len(preds)})",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        stem = im["file_name"].replace(".npy.gz", "")
        outp = os.path.join(out_dir, stem + ".jpg")
        cv2.imwrite(outp, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"   {stem}: {len(gt_by_img[im['id']])} GT, {len(preds)} pred -> {outp}")


if __name__ == "__main__":
    main()
