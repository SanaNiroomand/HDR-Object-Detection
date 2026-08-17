#!/usr/bin/env python3
#
# Copyright (c) Megvii, Inc. and its affiliates.          (YOLOX)
# Copyright 2023 Huawei Technologies Co., Ltd.            (RAOD)
# Copyright (c) 2026 Sana Niroomand and OGAM Research Laboratory,
#                    Middle East Technical University (METU)  (modifications)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0, and a copy is
# included in this repository as LICENSE-Apache-2.0.txt.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# ---------------------------------------------------------------------------
# NOTICE OF MODIFICATION (Apache 2.0 section 4(b))
#
# This file is a MODIFIED DERIVATIVE of RAOD's cfg_small.py. Changes made:
#   * dataset paths, annotation filenames and split selection point at HDR4RTT
#   * fine-tuning schedule replaces training-from-scratch: 50 epochs not 200,
#     2 warmup epochs not 5, base learning rate 0.002/64 not 0.01/64
#   * data_num_workers reduced from 16 to 4 for Windows spawn semantics
#   * pin_memory disabled in both dataloaders (Windows shared-memory failures)
#   * random_resize overridden to CAP the multiscale range at input_size rather
#     than centre on it, because sizes above 1280 exceed a 16 GB card
# ---------------------------------------------------------------------------
"""
cfg_hdr4rtt_rod.py -- RAOD training/eval config for converted HDR4RTT.

Derived from RAOD's own cfg_small.py. Only the parts that must change for this
dataset and this machine are changed; model geometry, gamma_range and input
size are left exactly as ROD trained them so the pretrained weights load and
mean the same thing.

Expected layout (COCORawDataset joins data_dir + "annotations" + json_file, and
data_dir + name + file_name):

    D:\\Data\\HDR\\hdr4rtt_rod\\
        annotations\\hdr4rtt_rod_seqsafe_train.json
        annotations\\hdr4rtt_rod_seqsafe_test.json
        images\\hdr_XXXXX.npy.gz

Defaults to the SEQSAFE split. HDR4RTT's own split is leaky: S3 is a continuous
video that was split frame-by-frame, and 100% of its test frames have a training
frame within +-3 of them. That does not affect a zero-shot number (the model
never saw HDR4RTT), but it inflates everything measured after fine-tuning. Set
SPLIT = "original" only to reproduce older numbers for comparison.

num_classes stays 5. Verified from the checkpoints themselves: every .pth in
pre-trained/ has head.cls_preds.*.weight of shape (5, 64, 1, 1). The live
mapping in scripts/preprocess_anno.py is
    Pedestrian:1  Car:2  Cyclist:3  Tram:4  Truck:5
(the 6-class line including Tricycle is commented out there). HDR4RTT
contributes only Pedestrian and Car; the other three heads get no positive
examples and their AP will read 0/nan. That is expected, not a bug.

Windows notes:
  - data_num_workers is 4, not 16. Windows uses spawn rather than fork, so each
    worker re-imports the module. Measured: 4 workers sustain ~0.03 s/batch,
    which is far ahead of the GPU, so more workers buy nothing here.
  - Launch training through RAOD's main.py from the RAOD/RAOD directory so that
    `import models` and `import yolox` resolve.

USE BATCH SIZE 4, NOT 8, on a 16 GB card. Measured step time and peak VRAM at
1280x1280 with fp16:

    batch 1 : 0.135 s   2.21 GB
    batch 2 : 0.252 s   4.41 GB
    batch 4 : 0.485 s   8.81 GB      <- linear scaling holds to here
    batch 8 : 3.900 s  17.51 GB      <- 8x worse than linear

Batch 8 needs 17.51 GB on a 17.1 GB card. Windows does not raise OutOfMemory
for this; WDDM pages GPU memory to system RAM, so the step silently becomes
erratic and ~8x slower (observed: a 50-epoch run reporting an 8-day ETA, with
per-iteration times swinging between 12 s and 82 s). The cost lives in
Adaptive_Module.apply_local, which interpolates tm_pts_num*3*2 = 48 channels of
tone-curve parameters to full resolution: B*48*1280*1280*4 bytes is 1.57 GB at
B=8 for that one tensor, before activations kept for backward.

At batch 4 the run is ~6 min/epoch, ~5 h for 50 epochs.
"""
import os
import random

import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as YoloXBaseExp

DATA_ROOT = r"D:\Data\HDR\hdr4rtt_rod"
SPLIT = "seqsafe"          # "seqsafe" (leakage-free) or "original" (leaky, for comparison)


class Exp(YoloXBaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------------- model config (UNCHANGED from cfg_small.py) ----------
        self.num_classes = 5
        self.depth = 0.33
        self.width = 0.25
        self.act = 'silu'
        self.gamma_range = [7.0, 10.5]

        # ---------------- dataloader config ----------------------------------
        self.data_num_workers = 4          # Windows spawn; see module docstring
        self.input_size = (1280, 1280)
        self.multiscale_range = 5
        self.data_dir = DATA_ROOT
        self.train_ann = f'hdr4rtt_rod_{SPLIT}_train.json'
        self.val_ann = f'hdr4rtt_rod_{SPLIT}_test.json'
        self.train_ims = self.val_ims = 'images'

        # ---------------- transform config -----------------------------------
        self.enable_mixup = False
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)
        # hsv augmentation stays off: these are linear HDR values, not sRGB, so
        # an HSV jitter designed for 8-bit images does not mean what it means
        # there. cfg_small.py disables it for the same reason.
        self.hsv_prob = 0.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0

        # ---------------- training config ------------------------------------
        # Fine-tuning from ROD weights, not training from scratch: shorter
        # schedule, shorter warmup, and a lower base LR than cfg_small.py's
        # 0.01/64 so the pretrained tone-mapping module is not immediately
        # destroyed.
        self.warmup_epochs = 2
        self.max_epoch = 50
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.002 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 10
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 20
        self.eval_interval = 5
        self.output_dir = os.path.dirname(os.path.abspath(__file__))
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split('.')[0]

        # ---------------- testing config -------------------------------------
        self.test_size = (1280, 1280)
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels,
                                 act=self.act, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, strides=[16, 32, 64],
                             in_channels=in_channels, act=self.act, depthwise=True)
            self.model = YOLOX(backbone, head, nf=16, gamma_range=self.gamma_range)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        """Scale augmentation, capped at input_size instead of centred on it.

        cfg_small.py samples symmetrically, [1280-5*64, 1280+5*64] = [960, 1600].
        On a 16 GB card everything above 1280 exceeds VRAM and Windows pages GPU
        memory to system RAM rather than raising OutOfMemory. Measured, batch 4:

            960 : 0.234 s   7.65 GB reserved   ok
           1280 : 0.483 s  13.33 GB reserved   ok
           1408 : 0.917 s  17.56 GB reserved   OVER (card has 17.1)
           1600 : 2.415 s  22.67 GB reserved   OVER

        Mixed sampling reserved 20.11 GB and averaged 1.57 s/iter -- 16.3 h for
        50 epochs, against 5.0 h at a fixed 1280. Capping the range at 1280 keeps
        the scale augmentation (960..1280), stays inside VRAM, and is FASTER than
        fixed 1280 because the smaller sizes cost less: ~3.6 h for 50 epochs.

        Set multiscale_range = 0 to pin the size at 1280 entirely.
        """
        tensor = torch.LongTensor(2).cuda()
        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                max_size = int(self.input_size[0] / 64)          # cap, not centre
                min_size = max_size - self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(64 * size), 64 * int(size * size_factor))
            tensor[0], tensor[1] = size[0], size[1]
        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)
        return (tensor[0].item(), tensor[1].item())

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (COCORawDataset, TrainTransformRaw, YoloBatchSampler,
                                DataLoader, InfiniteSampler, MosaicDetectionRaw,
                                worker_init_reset_seed)
        from yolox.utils import wait_for_the_master, get_local_rank

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = COCORawDataset(
                data_dir=self.data_dir, json_file=self.train_ann, name=self.train_ims,
                img_size=self.input_size,
                preproc=TrainTransformRaw(max_labels=50, flip_prob=self.flip_prob,
                                          hsv_prob=self.hsv_prob),
                cache=cache_img)

        dataset = MosaicDetectionRaw(
            dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=TrainTransformRaw(max_labels=120, flip_prob=self.flip_prob,
                                      hsv_prob=self.hsv_prob),
            degrees=self.degrees, translate=self.translate,
            mosaic_scale=self.mosaic_scale, mixup_scale=self.mixup_scale,
            shear=self.shear, enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob, mixup_prob=self.mixup_prob)

        self.dataset = dataset
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size,
                                         drop_last=False, mosaic=not no_aug)
        # pin_memory=False on purpose. With pinned memory and multiple workers,
        # Windows raised "Couldn't open shared file mapping" from the pin_memory
        # thread during the smoke test. Unpinned host->device copies are slightly
        # slower, but this pipeline is gzip-decode bound, not transfer bound, so
        # the cost is negligible next to the risk of losing a multi-hour run.
        kw = {"num_workers": self.data_num_workers, "pin_memory": False,
              "batch_sampler": batch_sampler, "worker_init_fn": worker_init_reset_seed}
        return DataLoader(self.dataset, **kw)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCORawDataset, ValTransformRaw

        valdataset = COCORawDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_ims if not testdev else self.test_ims,
            img_size=self.test_size, preproc=ValTransformRaw(legacy=legacy))
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)
        kw = {"num_workers": self.data_num_workers, "pin_memory": False,
              "sampler": sampler, "batch_size": batch_size}
        return torch.utils.data.DataLoader(valdataset, **kw)
