#!/usr/bin/env python3
"""
train_torchvision.py -- fine-tune a torchvision detector on tone-mapped HDR4RTT.

This is row 2 of the detector comparison: a normally-sized modern detector fed
ordinary 8-bit tone-mapped images, with no learned tone-mapping module. Row 1 is
RAOD's 1M-parameter detector with its module (mAP 0.349). Row 3 would put RAOD's
module in front of this detector; row 3 minus row 2 is what the module actually
contributes once the detector is not tiny.

Uses the same annotations, splits and image geometry as the RAOD arm, so the
only difference between rows is the model and the input representation.

The tone map was chosen by measurement, not preference: pick_tmo.py scored eight
operators with a COCO-pretrained detector and gamma at the 99th percentile won
(spread across operators was 10% relative, so it mattered).
"""
import os
import json
import time
import math
import argparse

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(1)
NUM_CLASSES = 3          # background + Pedestrian + Car


class LdrCocoDataset(Dataset):
    """Reads the RAOD-format COCO json but loads the .png sibling of each entry."""

    def __init__(self, ann_path, img_dir, train=True, flip_prob=0.5):
        from pycocotools.coco import COCO
        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            self.coco = COCO(ann_path)
        self.img_dir = img_dir
        self.train = train
        self.flip_prob = flip_prob
        self.items = []
        for img_id in sorted(self.coco.getImgIds()):
            info = self.coco.loadImgs(img_id)[0]
            png = os.path.splitext(os.path.splitext(info["file_name"])[0])[0] + ".png"
            fp = os.path.join(img_dir, png)
            if not os.path.exists(fp):
                continue
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id], iscrowd=False))
            boxes, labels = [], []
            for a in anns:
                x, y, w, h = a["bbox"]
                if w <= 1 or h <= 1:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(int(a["category_id"]))     # 1=Pedestrian, 2=Car
            if not boxes:
                continue
            self.items.append((fp, np.array(boxes, np.float32),
                               np.array(labels, np.int64), img_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fp, boxes, labels, img_id = self.items[i]
        bgr = cv2.imread(fp)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        boxes = boxes.copy()
        if self.train and np.random.rand() < self.flip_prob:
            rgb = rgb[:, ::-1].copy()
            W = rgb.shape[1]
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        img = torch.from_numpy(rgb).permute(2, 0, 1)
        target = {"boxes": torch.from_numpy(boxes),
                  "labels": torch.from_numpy(labels),
                  "image_id": torch.tensor([img_id])}
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def build(arch, device):
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
        retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    if arch == "fasterrcnn":
        m = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    else:
        m = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        in_ch = m.backbone.out_channels
        n_anchors = m.anchor_generator.num_anchors_per_location()[0]
        m.head.classification_head = RetinaNetClassificationHead(
            in_ch, n_anchors, NUM_CLASSES,
            norm_layer=lambda c: torch.nn.GroupNorm(32, c))
    return m.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_ann", default=r"D:\Data\HDR\hdr4rtt_rod\annotations\hdr4rtt_rod_seqsafe_train.json")
    p.add_argument("--val_ann", default=r"D:\Data\HDR\hdr4rtt_rod\annotations\hdr4rtt_rod_seqsafe_test.json")
    p.add_argument("--img_dir", default=r"D:\Data\HDR\hdr4rtt_ldr\images")
    p.add_argument("--arch", default="fasterrcnn", choices=["fasterrcnn", "retinanet"])
    p.add_argument("--out_dir", default=r"D:\Codes\HDR\Sana\hdr4rtt_rod\tv_runs")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(args.out_dir, args.arch)
    os.makedirs(out_dir, exist_ok=True)

    tr = LdrCocoDataset(args.train_ann, args.img_dir, train=True)
    print(f"train: {len(tr)} images with boxes")
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, collate_fn=collate,
                        pin_memory=False, drop_last=True)

    model = build(args.arch, device)
    n_par = sum(x.numel() for x in model.parameters()) / 1e6
    print(f"{args.arch}: {n_par:.1f}M params, {NUM_CLASSES-1} classes, batch {args.batch}")

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    iters = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=iters, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.amp.autocast("cuda", enabled=args.amp):
                losses = model(imgs, targets)
                loss = sum(losses.values())
            if not math.isfinite(loss.item()):
                print("  non-finite loss, skipping step")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()
            run += loss.item(); step += 1
            if step % 100 == 0:
                el = time.time() - t0
                print(f"  ep{ep+1}/{args.epochs} it{step}/{iters} "
                      f"loss={run/100:.3f} lr={sched.get_last_lr()[0]:.5f} "
                      f"{el:.0f}s eta {el/step*(iters-step):.0f}s", flush=True)
                run = 0.0
        torch.save({"model": model.state_dict(), "epoch": ep + 1, "arch": args.arch},
                   os.path.join(out_dir, "last.pth"))
        print(f"epoch {ep+1} done, checkpoint saved ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\ntraining done in {time.time()-t0:.0f}s -> {out_dir}\\last.pth")
    print("evaluate with:")
    print(f'  python eval_torchvision.py --ann "{args.val_ann}" '
          f'--arch {args.arch} --weights "{out_dir}\\last.pth"')


if __name__ == "__main__":
    main()
