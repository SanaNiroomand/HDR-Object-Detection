# HDR object detection on HDR4RTT

Work on running object detectors directly on high-dynamic-range imagery, using the
HDR4RTT dataset (4,080 OpenEXR images, 32,964 boxes, 20 Pascal VOC classes).

Two things are in here: a full audit of the dataset, and a rebuilt pipeline for
running [RAOD](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.pdf)
(CVPR 2023) on it, plus a comparison against conventional tone-map-then-detect.

**[→ Step-by-step progress note](hdr4rtt_rod/progress_note.html)** ·
**[→ Dataset audit](hdr4rtt_analysis/REPORT.md)** ·
**[→ Pipeline details](hdr4rtt_rod/README.md)**

---

## Headline results

All on the same 779-image leakage-free test set, 4,082 ground-truth boxes,
person and car only.

| arm | HDR4RTT training | mAP | AP50 | Pedestrian | Car |
|---|---|---|---|---|---|
| RAOD (1M params), zero-shot | none | 0.059 | 0.128 | 0.018 | 0.237 |
| **RAOD (1M params), fine-tuned** | 4.6 h | **0.349** | 0.613 | 0.602 | 0.624 |
| RetinaNet (38M), zero-shot | none | 0.296 | 0.574 | 0.502 | 0.646 |
| Faster R-CNN (44M), zero-shot | none | 0.310 | 0.606 | 0.561 | 0.650 |

Fine-tuning lifts RAOD 5.9× overall, and 33× on the class it was effectively
blind to. But an off-the-shelf COCO detector on tone-mapped images reaches
comparable AP50 with **no training on this dataset at all**, which is the
result worth thinking about.

---

## Two bugs that invalidated earlier work

**1. Wrong input scale in evaluation.** RAOD's data loader divides pixel values
by 255 inside `load_image`, so the network expects `[0,1]`. The evaluation
script skipped it and fed `[0,255]`. Because RAOD's tone-mapping module applies
`x^(1/gamma)` with gamma of 7–10.5, every pixel saturated and the frame
collapsed to white.

The trap: the same division appears **commented out** in `ValTransformRaw` —
the obvious place to look — because it has already happened one level up.

Proof: RAOD's *own* sample image gave **0 detections**; after restoring one
line, **5**, including a car at 0.892 confidence.

**2. The converter optimised for the opposite of what the model wants.** RAOD
tone-maps internally and expects dark, linear input at mean ≈ 0.012. The
previous converter stretched images to fill `[0,255]`, mean 0.377 — **32×
brighter** than anything the model saw in training. Every rescaling variant
tried (p99, adaptive percentile, log, histogram matching) shares that same
"use the full range" goal, so all of them fail the same way.

---

## What the dataset audit found

Every one of the 4,080 EXR files was decoded and measured. Full detail in
[REPORT.md](hdr4rtt_analysis/REPORT.md).

- **It is three sources glued together**, separable for free from the EXR
  headers: 1,289 video/rendered frames, 1,489 bracketed photographs, and 1,302
  frames of one continuous HDR video.
- **Neither value ceiling is over-exposure.** About one pixel per image touches
  it — each image was individually normalised, so the stored values carry no
  absolute brightness meaning.
- **41% of images contain negative pixel values**, down to −872. Any log or
  gamma step must clamp first.
- **303 images (7.4%) form a separate very dark cluster**, ~250× darker than the
  rest, holding 13.1% of all boxes.
- **Median dynamic range is 4.11 decades** (≈13.7 stops), maximum 7.83 (26 stops).
- **The shipped train/test split leaks**: 100% of the video source's test frames
  have a training frame within ±3. Measured cost, with a controlled two-model
  experiment: **+0.020 mAP**, about 6% relative.

Per-image dynamic-range statistics for all 4,080 files are in
[`hdr4rtt_analysis/hdr_stats_sources.csv`](hdr4rtt_analysis/hdr_stats_sources.csv),
which makes it possible to stratify detection results by scene difficulty.

---

## Caveats worth reading before quoting any number

**Scores vary 5× between sources.** Same model, scored per source:

| source | what it is | mAP | AP50 |
|---|---|---|---|
| S1 | video / rendered | 0.755 | **0.982** |
| S3 | HDR video | 0.320 | 0.605 |
| S2 | bracketed photographs | **0.142** | 0.395 |

S1's 0.982 is not a plausible generalisation result. S1 carries no metadata, so
frame sequences cannot be detected and no guard band could be built — it is very
likely contaminated by near-duplicate frames. Visual inspection supports this:
an arbitrary run of 200 consecutive S1 files turned out to be one continuous
living-room scene.

S2 is the only source where the number is unambiguously trustworthy, and it is
much lower than the headline.

**The training set is small.** 2,994 images, but the video source contributes
949 frames from only ~6 contiguous runs — roughly 2,051 genuinely distinct
scenes, about 1.7% the size of a standard detection training set. The good
result is better explained by an easier task (two classes, large objects) and a
strong pretrained starting point than by dataset size.

**Not comparable to TMO-Det.** TMO-Det uses the same underlying dataset but
evaluates all 20 classes on a de-duplicated 1,871-image subset with RetinaNet.
This work uses 2 classes. Placing the numbers side by side would be invalid.

---

## Reproducing

Requires an environment with PyTorch and OpenCV; RAOD's own repository supplies
the model code. Data paths are set at the top of each script.

```bash
# 1. audit the dataset (all 4,080 files)
python hdr4rtt_analysis/scripts/scan_hdr.py
python hdr4rtt_analysis/scripts/sources.py

# 2. build annotations and both splits
python hdr4rtt_rod/build_rod_annotations.py

# 3. convert EXR -> RAOD format (no tone mapping)
python hdr4rtt_rod/convert_hdr4rtt_to_rod.py --gain 0.02

# 4. verify the training path before a long run
python hdr4rtt_rod/smoke_test_train.py --workers 0

# 5. evaluate
python hdr4rtt_rod/eval_rod.py --ann <split>.json --img_dir <images> --gains 1.0
```

For the tone-mapped comparison arm:

```bash
python hdr4rtt_rod/pick_tmo.py                      # choose the operator by measurement
python hdr4rtt_rod/convert_hdr4rtt_to_ldr.py --tmo gamma --percentile 99
python hdr4rtt_rod/train_torchvision.py --arch fasterrcnn
```

**Use batch size 4, not 8, on a 16 GB GPU.** Batch 8 needs 17.51 GB on a 17.1 GB
card, and Windows does not raise an out-of-memory error — it pages GPU memory to
system RAM, so the run silently becomes ~8× slower with no error message. A
batch-8 run reported an 8-day ETA. Details in
[hdr4rtt_rod/README.md](hdr4rtt_rod/README.md).

---

## Layout

| path | contents |
|---|---|
| `hdr4rtt_analysis/` | dataset audit: scripts, report, per-image statistics |
| `hdr4rtt_rod/` | conversion, annotations, splits, training configs, evaluation |
| `hdr4rtt_rod/results/` | every reported number, as saved JSON |
| `hdr4rtt_rod/viz/` | ground truth and predictions drawn on converted images |
| `hdr4rtt_rod/progress_note.html` | step-by-step record of the work |

Checkpoints, converted imagery and the dataset itself are deliberately excluded —
see [.gitignore](.gitignore).

## Open questions

1. **Check S1 for duplicate frames** using image similarity, since it has no
   metadata. The 0.982 score makes this the highest priority.
2. **Fine-tune the large detectors** to complete the comparison, then put RAOD's
   tone-mapping module in front of one to isolate what the module contributes
   when the detector is not tiny.
3. **Run RAOD under TMO-Det's protocol** — their filtering, their splits, all 20
   classes — for a genuinely comparable number.
