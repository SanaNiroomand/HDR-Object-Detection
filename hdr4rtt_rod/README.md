# HDR4RTT → RAOD/ROD pipeline

Rebuilt on Windows, August 2026. Replaces the earlier attempt in
`HDR4RTT_dataset_operations_to_match_RAOD/` and `RAOD/RAOD/convert_exr_to_raod.py`.

Environment: `C:\Users\OGAM\miniconda3\envs\hs-ml\python.exe` (torch 2.8 + CUDA,
RTX 5070 Ti). Added `pycocotools loguru tabulate thop tqdm psutil`.

---

## Why the earlier attempt failed

Two independent problems, both silent.

**1. The evaluation script fed the model the wrong scale.** RAOD's dataloader
divides by 255 ([coco_raw.py:141](../RAOD/RAOD/yolox/data/datasets/coco_raw.py:141),
inside `load_image`), so the network expects `[0,1]`. `eval_single_image.py`
skipped that step and fed `[0,255]`; its own docstring conceded the preprocessing
was "NOT verified against this project's actual ValTransformRaw". With `[0,255]`
input the tone-mapping module computes `x**(1/gamma)` on values up to 255,
everything saturates past 1.0, `clamp(0,1)` flattens the frame to white, and the
model returns nothing.

This is measurable: RAOD's **own** sample `scripts/official_output/day-00018.npy.gz`
produced 0 detections at conf=0.001 in `cfg_small_HDR4RTT/val_single_log.txt`.
After the fix it produces 5, including a Car at 0.892. Every earlier
"this preprocessing method doesn't work" conclusion was measured through this bug.

**2. The converter was optimising for the opposite of what the model wants.**
RAOD tone-maps internally — [adaptive_module.py](../RAOD/RAOD/models/adaptive_module.py)
applies `img ** (1.0/gamma)` with gamma learned in `[7.0, 10.5]`, plus a learned
local curve. It is built to receive dark, linear, un-tone-mapped input and
brighten it itself. RAOD's own data sits at mean `0.0118` in `[0,1]`.

The old converter stretched each image to fill `[0,255]` (per-population white
point, p99, adaptive percentile, log, histogram matching — all variants of the
same goal), landing at mean `0.377`, **32× brighter** than anything the model saw
in training. Pre-tone-mapping HDR data before RAOD also discards exactly the
signal RAOD exists to exploit.

---

## Measured results

*Source: all numbers in this document were measured here. Where a figure is
quoted from someone else's paper it is labelled as such at that point.*

*Scale: this document reports mAP in the **0-1** form `pycocotools` prints
(`0.349`). The top-level [README](../README.md) reports the same values **x100**
(`34.9`), the convention the papers use. Identical numbers, different notation.*

Zero-shot, `best-day_night.pth`, no fine-tuning.

### Input gain sweep (144-image subset, leakage-free split)

*All **zero-shot** — pretrained ROD weights, no training on HDR4RTT.*

| gain | mAP | AP50 |
|---|---|---|
| 1.0 | 0.0216 | 0.0601 |
| 0.5 | 0.0448 | 0.1045 |
| 0.25 | 0.0521 | 0.1120 |
| 0.1 | 0.0473 | 0.1016 |
| 0.05 | 0.0475 | 0.0997 |
| **0.02** | **0.0560** | **0.1173** |
| 0.01 | 0.0529 | 0.1141 |

2.6× better at 0.02 than at 1.0, and flat between 0.01 and 0.25. The dataset is
converted at **gain 0.02**.

### Full test set

*All **zero-shot**.*

| split | images | GT boxes | mAP | AP50 | Pedestrian AP50 | Car AP50 |
|---|---|---|---|---|---|---|
| `seqsafe` (leakage-free) | 779 | 4,082 | **0.0590** | 0.1277 | 0.0182 | 0.2371 |
| `original` (leaky) | 758 | 4,433 | 0.0684 | 0.1470 | 0.0152 | 0.2787 |

Zero-shot, the two splits should agree — the model never trained on HDR4RTT, so
leakage cannot help it. They differ by ~0.01 mAP purely from different image
sampling. Report `seqsafe`; the gap between them **after fine-tuning** is the
leakage effect, and that is the number to watch.

### Fine-tuned (50 epochs from `best-day_night.pth`, batch 4, ~4.6 h)

Evaluated with `best_ckpt.pth` on the leakage-free `seqsafe` test set:

| | zero-shot | fine-tuned | |
|---|---|---|---|
| mAP | 0.0590 | **0.3490** | 5.9× |
| AP50 | 0.1277 | **0.6132** | 4.8× |
| **Pedestrian AP50** | 0.0182 | **0.6021** | **33×** |
| Car AP50 | 0.2371 | **0.6244** | 2.6× |
| AP large | 0.100 | 0.526 | |
| AP small | 0.003 | 0.026 | |

Validation by epoch (mAP): 5 → .253, 10 → .299, 15 → .318, 20 → .333, 25 → .340,
30 → .343, 35 → .347, 40 → .349, then flat to 50. Converged by ~epoch 35; a
30-epoch schedule would have reached the same place in under 3 h.

The Pedestrian result is the headline. Zero-shot it was 0.018 — effectively
blind, despite person being 85% of the boxes, because ROD's "Pedestrian" means a
full-body figure in a road scene. After fine-tuning it reaches 0.602, on par with
Car. The domain gap was real but entirely trainable; nothing about HDR4RTT is
structurally incompatible with RAOD.

AP stays low on small objects (0.026) and high on large (0.526), matching the
dataset: median object is 7.2% of the image side and only 0.7% are tiny.

### Leakage cost, measured

> **An earlier version of this section was wrong and has been replaced.** It
> compared ONE model against two test sets and reported +0.005 mAP. That
> comparison was invalid: 195 of the 758 images in the original test set (26%)
> were inside that model's training data, because the `seqsafe` split reassigns
> S3 frames. It was measuring memorisation, not duplicate frames.

The correct design is two models with identical settings, each evaluated on its
OWN split, compared **S3 against S3** so scene content is held constant (S3 is
the only source with frame adjacency; comparing whole splits would confound the
leakage with the difference between traffic video and bracketed stills).

| model trained on | training | S3 test mAP | S3 test AP50 |
|---|---|---|---|
| `seqsafe` (leakage-free) | fine-tuned 50 ep | 0.3197 | 0.6047 |
| `original` (leaky) | fine-tuned 50 ep | **0.3393** | **0.6181** |

**The original split overstates the video source by +0.020 mAP**, about 6%
relative. Control: the photo sources, never reassigned, agree closely between the
two models (S1 0.7554 vs 0.7591; S2 0.1416 vs 0.1392), confirming the S3 gap is
the duplicate frames rather than training noise.

Report `seqsafe`.

### What the numbers mean

Pedestrian AP50 is 0.018 despite person being 85% of the boxes; Car reaches 0.237.
The visualizations in `viz/` show why: ROD's "Pedestrian" is a full-body person in
a road scene, while HDR4RTT's "person" is four people sitting around an indoor
table or two performers on a stage. The model labels stage lights and a man's cap
"Car" and finds none of the people.

The plumbing is now correct; the remaining gap is domain, not preprocessing.
**Zero-shot on this pairing was never going to be good.** Fine-tuning is where the
result comes from.

---

## Pipeline contract

Established by reading RAOD's own scripts, not inferred.

| stage | rule | source |
|---|---|---|
| resize | 1280×1280, **aspect ratio deliberately not preserved** | `preprocess_raw.py` `cv2.resize(im,(1280,1280))` |
| boxes | squashed **anisotropically** (`x·1280/W`, `y·1280/H`) | `preprocess_anno.py` `_get_box` |
| json dims | `width = height = 1280`, hardcoded | `preprocess_anno.py` `_image` |
| scale | stored `[0,255]`, loader divides by 255 → model sees `[0,1]` | `coco_raw.py:141` |
| classes | `Pedestrian:1 Car:2 Cyclist:3 Tram:4 Truck:5` | `preprocess_anno.py:26` (6-class line commented out) |
| min box | drop `w<16` or `h<16` in 1280-space | `preprocess_anno.py` `to_coco` |

Class count verified independently from the weights: every `.pth` in
`pre-trained/` has `head.cls_preds.*.weight` of shape `(5,64,1,1)`.

HDR4RTT → ROD keeps **person→Pedestrian** and **car→Car** only: 21,399 of 32,964
boxes (64.9%) over 3,818 of 4,080 images. `bicycle→Cyclist` is rejected because
Cyclist means a person riding, not the bicycle object; bus/train have 7 and 8
instances total.

---

## Splits

`seqsafe` is the trustworthy one. HDR4RTT's own split is leaky: source S3 is a
continuous video (`000001.exr`…`001499.exr`) that was split frame-by-frame, and
**100% of its test frames have a training frame within ±3**. `seqsafe` groups S3
into 24 contiguous blocks, assigns whole blocks, and drops a ±3 guard band at every
boundary (45 frames).

Verified: `seqsafe` has **0** test frames within ±3 of a train frame; the original
split has 252/276 at ±1.

**Caveat:** S1 (1,289 images) carries no EXR metadata, so it has no frame numbering
to group by. If S1 is also video-derived, `seqsafe` still contains leakage from
that source. Detecting it needs image-similarity hashing.

---

## Batch size 4, not 8 — a 16 GB card cannot run batch 8

Measured step time and peak VRAM at 1280×1280, fp16, on an RTX 5070 Ti (17.1 GB):

| batch | step | peak VRAM |
|---|---|---|
| 1 | 0.135 s | 2.21 GB |
| 2 | 0.252 s | 4.41 GB |
| 4 | 0.485 s | 8.81 GB |
| 8 | **3.900 s** | **17.51 GB** |

Scaling is linear to batch 4, then batch 8 is 8× worse than linear because it
needs 17.51 GB on a 17.1 GB card. **Windows does not raise OutOfMemory for this** —
WDDM pages GPU memory to system RAM, so the step just becomes slow and erratic.
A batch-8 run reported an **8-day ETA** with per-iteration times swinging between
12 s and 82 s, and no error of any kind.

The memory sits in `Adaptive_Module.apply_local`, which interpolates
`tm_pts_num*3*2 = 48` channels of tone-curve parameters to full resolution:
`B*48*1280*1280*4` bytes is 1.57 GB at B=8 for that single tensor, before the
activations retained for backward.

The dataloader was ruled out first by measurement, not assumption: 4 workers
sustain ~0.03 s/batch median with 136 GB RAM free — far ahead of the GPU.

At batch 4: ~6 min/epoch, ~5 h for 50 epochs.

## Known pre-existing quirk (not introduced here)

`mosaicdetection.py:108` and `data_augment.py:111` fill augmentation borders with
**114**, a constant inherited from upstream YOLOX where images are 8-bit `[0,255]`.
RAOD divides by 255 *before* augmentation, so the fill sits ~114× above real pixel
values and saturates to white through the tone-mapping module. Typically 0–15% of a
training image.

This affects RAOD's own training identically, so fine-tuning inherits it
consistently rather than being broken by it. The eval path is unaffected
(`ValTransformRaw`, no mosaic, and no padding for 1280×1280 inputs), which is why
the zero-shot numbers above are sound. Setting the fill to `114/255` or `0`, or
`mosaic_prob = 0`, is a reasonable experiment.

---

## Files

| file | purpose |
|---|---|
| `build_rod_annotations.py` | ROD-format COCO json + both splits |
| `convert_hdr4rtt_to_rod.py` | EXR → `.npy.gz`, no tone mapping |
| `eval_rod.py` | COCO mAP, with `--gains` sweep |
| `run_single.py` | one image, detailed; uses RAOD's real preprocessing |
| `visualize.py` | draw GT + predictions; verifies the coordinate contract |
| `cfg_hdr4rtt_rod.py` | RAOD training/eval config for this dataset |
| `smoke_test_train.py` | proves the training path runs before a long job |

Data lives outside the repo at `D:\Data\HDR\hdr4rtt_rod\` (60 GB):
`annotations/`, `images/` (3,818 × ~16 MB), `sweep_images/` (144, unclipped).

---

## Commands

Rebuild annotations and splits:

```bash
python build_rod_annotations.py
```

Convert (≈6 min, 10 workers, 60 GB out):

```bash
python convert_hdr4rtt_to_rod.py --output_dir "D:\Data\HDR\hdr4rtt_rod\images" --gain 0.02
```

Evaluate:

```bash
python eval_rod.py --ann "D:\Data\HDR\hdr4rtt_rod\annotations\hdr4rtt_rod_seqsafe_test.json" --img_dir "D:\Data\HDR\hdr4rtt_rod\images" --gains 1.0
```

Check the training path before a long run:

```bash
python smoke_test_train.py --workers 0
```

Fine-tune (run from the RAOD directory so `import models` resolves). **`-b 4`, not 8** —
see the batch-size section above:

```bash
python main.py -f "D:\Codes\HDR\Sana\hdr4rtt_rod\cfg_hdr4rtt_rod.py" -d 1 -b 4 --fp16 -c "D:\Codes\HDR\Sana\RAOD\RAOD\pre-trained\best-day_night.pth"
```

Requires `tensorboard` (RAOD imports `SummaryWriter` unconditionally in
`yolox/core/trainer.py`). `data_num_workers` is 4; drop to 0 if Windows
shared-memory errors appear.
