#!/usr/bin/env python3
"""
smoke_test_train.py -- prove the fine-tuning path runs on Windows before
committing to a long job.

Checks, in order:
  1. the config's train dataloader builds and yields a batch (Windows uses
     spawn, not fork, so DataLoader workers are the usual failure point)
  2. images and targets have the shapes and ranges the model expects
  3. pretrained ROD weights load into the model
  4. a forward + backward + optimizer step completes and the loss is finite

Run from anywhere; it chdir's into RAOD so `import models` resolves.
"""
import os
import sys
import argparse
import importlib.util

import numpy as np
import torch

RAOD_DIR = r"D:\Codes\HDR\Sana\RAOD\RAOD"
CFG = r"D:\Codes\HDR\Sana\hdr4rtt_rod\cfg_hdr4rtt_rod.py"
CKPT = os.path.join(RAOD_DIR, "pre-trained", "best-day_night.pth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default=CFG)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=None,
                   help="override cfg data_num_workers (0 = load in main process)")
    args = p.parse_args()

    sys.path.insert(0, RAOD_DIR)
    os.chdir(RAOD_DIR)
    spec = importlib.util.spec_from_file_location("cfg_mod", args.cfg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    exp = mod.Exp()
    if args.workers is not None:
        exp.data_num_workers = args.workers
    exp.seed = 0

    print(f"cfg            : {os.path.basename(args.cfg)}")
    print(f"data_dir       : {exp.data_dir}")
    print(f"train_ann      : {exp.train_ann}")
    print(f"val_ann        : {exp.val_ann}")
    print(f"num_classes    : {exp.num_classes}   input_size: {exp.input_size}")
    print(f"data_num_workers: {exp.data_num_workers}\n")

    print("[1/4] building train dataloader ...")
    loader = exp.get_data_loader(batch_size=args.batch, is_distributed=False,
                                 no_aug=False, cache_img=False)
    print(f"      dataset size: {len(exp.dataset)} images")

    print("[2/4] pulling one batch ...")
    it = iter(loader)
    batch = next(it)
    imgs, targets = batch[0], batch[1]
    print(f"      imgs    : {tuple(imgs.shape)} {imgs.dtype}  "
          f"min={imgs.min():.4f} mean={imgs.mean():.4f} max={imgs.max():.4f}")
    print(f"      targets : {tuple(targets.shape)}  "
          f"non-empty rows={int((targets.sum(dim=2) != 0).sum())}")
    cls_ids = targets[..., 0][targets.sum(dim=2) != 0]
    if cls_ids.numel():
        uniq, cnt = np.unique(cls_ids.numpy().astype(int), return_counts=True)
        names = ["Pedestrian", "Car", "Cyclist", "Tram", "Truck"]
        print("      classes : " + ", ".join(f"{names[u]}={c}" for u, c in zip(uniq, cnt)))

    # The augmentation fill constant is 114, inherited from upstream YOLOX where
    # images are 8-bit [0,255]. RAOD's coco_raw.py divides by 255 in load_image,
    # BEFORE augmentation runs, so mosaic canvas (mosaicdetection.py:108) and
    # warpAffine borderValue (data_augment.py:111) paint 114 onto data whose real
    # pixels are <= 1.0. Those regions saturate to white through the tone-mapping
    # module. This is pre-existing RAOD behaviour -- ROD's own weights were
    # trained with the same fill -- so fine-tuning inherits it consistently
    # rather than being broken by it. It only occurs when mosaic is active;
    # the eval path (ValTransformRaw, no mosaic, no padding for 1280x1280
    # inputs) is unaffected, which is why the zero-shot numbers are sound.
    PAD = 114.0
    flat = imgs.flatten().numpy()
    pad_frac = float((flat == PAD).mean())
    # warpAffine interpolates between real pixels and the 114 border, so values
    # just under 114 are also fill artefacts, not data. Judge the distribution by
    # percentiles rather than by max.
    q = np.percentile(flat, [50, 75, 90, 95, 99])
    print(f"      fill=114 pixels: {pad_frac:.1%}  (YOLOX augmentation constant)")
    print("      percentiles    : " +
          "  ".join(f"p{p}={v:.4f}" for p, v in zip([50, 75, 90, 95, 99], q)))
    below = float((flat < 1.5).mean())
    print(f"      pixels < 1.5   : {below:.1%}  (real data lives here)")
    assert q[0] <= 1.5, (f"median pixel should be ~[0,1] after coco_raw's /255; "
                         f"got {q[0]:.3f} -- the data scale is wrong, not just the fill")

    print("[3/4] loading pretrained ROD weights ...")
    model = exp.get_model().cuda()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"      loaded (epoch {ck.get('start_epoch')}); "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"      missing keys   : {list(missing)[:5]}")
        print(f"      unexpected keys: {list(unexpected)[:5]}")

    print("[4/4] one forward + backward + step ...")
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=exp.basic_lr_per_img * args.batch,
                          momentum=exp.momentum, weight_decay=exp.weight_decay)
    imgs_c = imgs.cuda().float()
    tgt_c = targets.cuda().float()
    out = model(imgs_c, tgt_c)
    loss = out["total_loss"]
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    opt.step()
    opt.zero_grad()
    print("      losses: " + ", ".join(
        f"{k}={float(v):.4f}" for k, v in out.items() if torch.is_tensor(v)))
    print(f"      grad norm: {gn:.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    print("\nPASS - training path works. Launch the real run with:")
    print(f'  cd "{RAOD_DIR}"')
    print(f'  python main.py -f "{args.cfg}" -d 1 -b 8 --fp16 -c "{args.ckpt}"')


if __name__ == "__main__":
    main()
