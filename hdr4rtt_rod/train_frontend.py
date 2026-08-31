#!/usr/bin/env python3
"""
train_frontend.py -- train ONE detector against a choice of tone-mapping front
end, so the detector can be held fixed while only the front end changes.

This is the controlled comparison the advisor asked for. Kocdemir's thesis
Tables 4.2 and 4.3 show the verdict flips with the detector -- the learned joint
method beats every classical operator under RetinaNet (31.6 vs Mantiuk 31.3) and
loses to four of them under Faster R-CNN (27.7 vs Fattal 29.5) -- so leaving the
detector free to vary makes the comparison uninterpretable.

Arms:
  gamma reinhard durand log   pre-computed 8-bit images from make_frontends.py
  hdr                          pre-computed float16 linear, no tone curve
  tmm                          reads the SAME linear float data as `hdr`, but
                               applies RAOD's Adaptive_Module inside the network,
                               trained jointly with the detector

The `tmm` arm is the one nobody has run. RAOD publishes it bolted to a
1M-parameter YOLOX; putting it in front of the same detector the classical
operators feed isolates the module's contribution from the detector's capacity.
Its weights are initialised from RAOD's released checkpoint rather than randomly,
so it starts from a trained tone-mapping behaviour.

Every arm shares the same annotations, split, geometry, schedule, optimiser and
augmentation. The only difference is what happens to the pixels.
"""
import os
import sys
import gzip
import json
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(1)

RAOD_DIR = r"D:\Codes\HDR\Sana\RAOD\RAOD"
RAOD_CKPT = os.path.join(RAOD_DIR, "pre-trained", "best-day_night.pth")
FRONTEND_ROOT = r"D:\Data\HDR\hdr4rtt_voc20\frontends"
ANN_DIR = r"D:\Data\HDR\hdr4rtt_voc20\annotations"
FLOAT_ARMS = {"hdr", "tmm"}
ARM_DIR = {"tmm": "hdr"}          # tmm consumes the untouched linear data


class FrontendDataset(Dataset):
    def __init__(self, ann_path, img_dir, is_float, train=True, flip_prob=0.5):
        from pycocotools.coco import COCO
        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            coco = COCO(ann_path)
        self.is_float = is_float
        self.train = train
        self.flip_prob = flip_prob
        self.items = []
        ext = ".npy.gz" if is_float else ".png"
        for img_id in sorted(coco.getImgIds()):
            info = coco.loadImgs(img_id)[0]
            stem = os.path.splitext(os.path.splitext(info["file_name"])[0])[0]
            fp = os.path.join(img_dir, stem + ext)
            if not os.path.exists(fp):
                continue
            boxes, labels = [], []
            for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id], iscrowd=False)):
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(int(a["category_id"]))
            if not boxes:
                continue
            self.items.append((fp, np.array(boxes, np.float32),
                               np.array(labels, np.int64), img_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fp, boxes, labels, img_id = self.items[i]
        if self.is_float:
            with gzip.GzipFile(fp, "r") as f:
                rgb = np.load(f).astype(np.float32)
        else:
            bgr = cv2.imread(fp)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        boxes = boxes.copy()
        if self.train and np.random.rand() < self.flip_prob:
            rgb = rgb[:, ::-1].copy()
            W = rgb.shape[1]
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return (torch.from_numpy(rgb).permute(2, 0, 1),
                {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels),
                 "image_id": torch.tensor([img_id])})


def collate(b):
    return tuple(zip(*b))


class TmmWrapper(nn.Module):
    """RAOD's Adaptive_Module in front of a torchvision detector.

    The module maps linear HDR to a display-referred image; torchvision detectors
    expect [0,1] and normalise internally, so the output is clamped to [0,1]
    rather than scaled to [0,255] as RAOD's own YOLOX does.
    """

    def __init__(self, detector, nf=16, gamma_range=(7.0, 10.5), init_ckpt=RAOD_CKPT,
                 input_gain=1.0):
        super().__init__()
        sys.path.insert(0, RAOD_DIR)
        from models.adaptive_module import Adaptive_Module
        self.tmm = Adaptive_Module(in_ch=3, nf=nf, gamma_range=list(gamma_range))
        self.detector = detector
        # RAOD trained this module on data whose mean sits near 0.012; the linear
        # HDR arm here averages ~0.068, about 6x brighter. The module predicts a
        # per-image gamma so it can in principle absorb that, but starting at the
        # scale its weights were fitted for is worth testing separately.
        self.input_gain = float(input_gain)
        if init_ckpt and os.path.exists(init_ckpt):
            sd = torch.load(init_ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            tmm_sd = {k[len("TMM."):]: v for k, v in sd.items() if k.startswith("TMM.")}
            missing, unexpected = self.tmm.load_state_dict(tmm_sd, strict=False)
            print(f"  TMM initialised from RAOD checkpoint: {len(tmm_sd)} tensors, "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print("  TMM randomly initialised (no checkpoint found)")

    def forward(self, images, targets=None):
        x = torch.stack(list(images))
        if self.input_gain != 1.0:
            x = x * self.input_gain
        x = torch.clamp(self.tmm(x), 0, 1)
        return self.detector([x[i] for i in range(x.shape[0])], targets)


def build_detector(arch, num_classes):
    from torchvision.models.detection import (
        retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    if arch == "retinanet":
        m = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        m.head.classification_head = RetinaNetClassificationHead(
            m.backbone.out_channels, m.anchor_generator.num_anchors_per_location()[0],
            num_classes, norm_layer=lambda c: nn.GroupNorm(32, c))
    else:
        m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        m.roi_heads.box_predictor = FastRCNNPredictor(
            m.roi_heads.box_predictor.cls_score.in_features, num_classes)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["gamma", "reinhard", "durand", "log", "hdr", "tmm"])
    p.add_argument("--arch", default="retinanet", choices=["retinanet", "fasterrcnn"])
    p.add_argument("--frontend_root", default=FRONTEND_ROOT)
    p.add_argument("--ann_dir", default=ANN_DIR)
    p.add_argument("--train_ann", default=None,
                   help="explicit training annotation file; overrides --ann_dir")
    p.add_argument("--val_ann", default=None,
                   help="explicit test annotation file; overrides --ann_dir")
    p.add_argument("--out_root", default=r"D:\Codes\HDR\Sana\hdr4rtt_rod\frontend_runs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--tmm_lr_mult", type=float, default=0.1,
                   help="TMM learns slower than the detector head; it arrives "
                        "pretrained and a full-rate update destroys it early")
    p.add_argument("--tmm_random_init", action="store_true",
                   help="ignore RAOD's released TMM weights and start fresh. "
                        "They were fitted on automotive RAW, a different domain, "
                        "so they may be a worse starting point than random.")
    p.add_argument("--input_gain", type=float, default=1.0,
                   help="scale fed to the TMM before its own processing; 0.18 "
                        "puts this data near the mean RAOD's weights were fitted at")
    p.add_argument("--tag", default="", help="suffix for the output directory, so "
                                             "variants do not overwrite each other")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_float = args.arm in FLOAT_ARMS
    img_dir = os.path.join(args.frontend_root, ARM_DIR.get(args.arm, args.arm))
    out_dir = os.path.join(args.out_root,
                           f"{args.arch}_{args.arm}" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(out_dir, exist_ok=True)

    import json as _json
    with open(train_ann_for_classes := (args.train_ann or
              os.path.join(args.ann_dir, "hdr4rtt_voc20_train.json")),
              encoding="utf-8") as f:
        n_classes = len(_json.load(f)["categories"]) + 1    # + background
    train_ann = args.train_ann or os.path.join(args.ann_dir, "hdr4rtt_voc20_train.json")
    tr = FrontendDataset(train_ann, img_dir, is_float, train=True)
    print(f"arm={args.arm}  arch={args.arch}  images={len(tr)}  classes={n_classes-1}")
    print(f"  train annotations: {os.path.basename(train_ann)}")
    print(f"  reading {img_dir}")
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, pin_memory=False, drop_last=True)

    det = build_detector(args.arch, n_classes)
    if args.arm == "tmm":
        model = TmmWrapper(det,
                           init_ckpt=None if args.tmm_random_init else RAOD_CKPT,
                           input_gain=args.input_gain)
        print(f"  tmm_lr_mult={args.tmm_lr_mult}  input_gain={args.input_gain}  "
              f"init={'random' if args.tmm_random_init else 'RAOD checkpoint'}")
    else:
        model = det
    model.to(device)
    print(f"  {sum(q.numel() for q in model.parameters())/1e6:.1f}M params")

    if args.arm == "tmm":
        groups = [{"params": [q for q in model.detector.parameters() if q.requires_grad],
                   "lr": args.lr},
                  {"params": [q for q in model.tmm.parameters() if q.requires_grad],
                   "lr": args.lr * args.tmm_lr_mult}]
    else:
        groups = [{"params": [q for q in model.parameters() if q.requires_grad],
                   "lr": args.lr}]
    opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=1e-4)
    iters = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    step, t0, skipped = 0, time.time(), 0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for imgs, targets in loader:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=True):
                loss = sum(model(imgs, targets).values())
            if not math.isfinite(loss.item()):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [q for g in groups for q in g["params"]], 10.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()
            run += loss.item(); step += 1
            if step % 200 == 0:
                el = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
                print(f"  ep{ep+1}/{args.epochs} it{step}/{iters} loss={run/200:.3f} "
                      f"{el:.0f}s eta {el/step*(iters-step):.0f}s peak={peak:.1f}GB",
                      flush=True)
                run = 0.0
        torch.save({"model": model.state_dict(), "arm": args.arm, "arch": args.arch,
                    "epoch": ep + 1, "n_classes": n_classes,
                    "input_gain": args.input_gain, "tmm_lr_mult": args.tmm_lr_mult,
                    "tmm_random_init": args.tmm_random_init},
                   os.path.join(out_dir, "last.pth"))
        print(f"epoch {ep+1} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[{args.arch}_{args.arm}] trained in {time.time()-t0:.0f}s, "
          f"{skipped} non-finite steps skipped -> {out_dir}\\last.pth")


if __name__ == "__main__":
    main()
