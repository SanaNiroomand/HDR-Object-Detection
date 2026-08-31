# HDR object detection on HDR4RTT

Object detection on high dynamic range imagery, using the HDR4RTT dataset: 4,080
OpenEXR images, 32,964 boxes, 20 Pascal VOC classes.

There are three parts. An audit of the dataset, a rebuilt pipeline for running
[RAOD](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.pdf)
(CVPR 2023) on it, and a comparison of six tone mapping methods behind one fixed
detector.

[Progress note](hdr4rtt_rod/progress_note.html) ·
[Dataset audit](hdr4rtt_analysis/REPORT.md) ·
[Pipeline details](hdr4rtt_rod/README.md)

> **Work in progress, not peer reviewed.** This is an internship project. The
> numbers are preliminary and may change. Please ask before citing anything here.
> No warranty, see [LICENSE](LICENSE).

**Contents**

1. [Which tone map works best?](#1-which-tone-map-works-best)
2. [Detector comparison](#2-detector-comparison)
3. [Reading the numbers](#3-reading-the-numbers)
4. [Why the earlier numbers were higher](#4-why-the-earlier-numbers-were-higher)
5. [The dataset](#5-the-dataset)
6. [Two bugs](#6-two-bugs)
7. [Comparison with the MS thesis](#7-comparison-with-the-ms-thesis)
8. [Method notes](#8-method-notes)
9. [Reproducing](#9-reproducing)
10. [Layout and open questions](#10-layout-and-open-questions)
11. [Licence and attribution](#11-licence-and-attribution)

---

## 1. Which tone map works best?

HDR data has to be compressed into something a detector can read. This section
keeps the detector fixed and changes only that compression step, so any
difference in score comes from the tone map and nothing else.

Every row uses RetinaNet R50-FPN v2 (torchvision `retinanet_resnet50_fpn_v2`),
COCO-pretrained with a fresh 20-class head, fine-tuned for 10 epochs at batch 4.
No row is zero-shot. The split is deduplicated: 2,506 train, 633 test, 20
classes. I ran all of these.

![Five tone maps applied to the same two HDR photographs](hdr4rtt_rod/viz/tone_mapping_comparison.jpg)

Two photographs through the five fixed front ends. The left column has no tone
curve at all, and in the workshop scene the welding arc takes the whole output
range, leaving everything else black. The four to its right are the same data
under different curves. The learned method isn't shown because it builds a
different curve for every photograph.

| # | front end | whose method | mAP | AP50 | AP50 head-8 | S2 only |
|---|---|---|---|---|---|---|
| 1 | Reinhard *(global)* | classical, 2002 | **31.3** | 49.7 | **59.9** | 24.2 |
| 2 | HDR with gamma | standard display curve | 30.4 | 48.5 | 56.7 | 22.9 |
| 3 | Log compression | simple formula | 29.9 | 48.8 | 58.4 | 22.5 |
| 4 | Durand *(local)* | classical, 2002 | 29.7 | **50.6** | 58.1 | 22.7 |
| 5 | RAOD module | learned, CVPR 2023 | 28.7 | 46.5 | 54.1 | 22.0 |
| 6 | no tone curve | control | 24.4 | 41.8 | 49.3 | 18.8 |

Row 5 trains the learned module jointly with the detector, the way RAOD intends.
"AP50 head-8" averages only the eight classes with enough test instances to be
stable. "S2 only" scores the bracketed photographs on their own, the one source
where duplicates are impossible, so it's the most conservative figure here.

### What it shows

Skipping the tone curve costs 6.9 mAP, and it's the worst arm on every source.
Kocdemir's thesis found the same thing under both of its detectors, where raw
HDR scored 26.3 and 23.5.

The choice of curve matters much less than making one. The four sensible
operators sit between 29.7 and 31.3, a 1.6 point spread, against a 6.9 point gap
to using nothing.

RAOD's learned module lost to a one-line formula from 2002. Reinhard beats it by
2.6 mAP and it finishes fifth of six. I suspected my own settings, so I retrained
it three more times, changing one thing each time: full learning rate, random
init instead of RAOD's released weights, and input rescaled to the level those
weights expect. All three still landed below every fixed operator. Those three
ran before deduplication, so their absolute values are high in the same way, but
the ordering is what matters and it didn't move.

That isn't evidence the method is wrong. RAOD pairs it with a 1M-parameter
detector, where the front end has much more work to do. A 38M-parameter detector
may already handle internally whatever the module was supplying. Kocdemir's
tables point the same way from the other side: his learned method beats every
classical operator under RetinaNet and loses to four of them under Faster R-CNN.
Whether a learned tone map helps seems to depend on what sits behind it.

---

## 2. Detector comparison

A different question: how do the detectors themselves compare? Both the input and
the detector change here, so this table can't separate the two. Section 1 does
that. Same deduplicated split as section 1, but 2 classes rather than 20, so the
two still don't compare with each other.

779 test images before, 576 after deduplication. All numbers measured here,
4,082 boxes before and 3,785 after, person and car only.

| detector | input | params | training | mAP | AP50 | Pedestrian | Car |
|---|---|---|---|---|---|---|---|
| RAOD (YOLOX-nano + its module) | HDR | 1M | zero-shot | 7.0 | 15.3 | 1.8 | 28.8 |
| RAOD (YOLOX-nano + its module) | HDR | 1M | fine-tuned, 4.6 h | 27.0 | 55.0 | 49.7 | 60.2 |
| RetinaNet R50-FPN v2 | tone-mapped | 38M | zero-shot | 24.2 | 50.7 | 38.5 | 62.9 |
| Faster R-CNN R50-FPN v2 | tone-mapped | 44M | zero-shot | 26.4 | 54.7 | 44.6 | 64.8 |
| Faster R-CNN R50-FPN v2 | tone-mapped | 44M | fine-tuned, 0.5 h | **43.0** | **74.0** | 68.8 | 79.1 |

RAOD here is its released configuration: YOLOX at depth 0.33, width 0.25,
depthwise convolutions, plus its Adaptive_Module tone mapper. 1.0M parameters in
total.

An earlier version of this table, measured before deduplication, reported 5.9 /
34.9 / 29.6 / 31.0 / 51.7. The fine-tuned rows fell 8 to 9 points once the
duplicate frames went, and the zero-shot rows moved much less. That gap is itself
the tell: a model that never trained on this data can't benefit from duplicates
in the test set, so only the fine-tuned rows had anything to lose. RAOD zero-shot
even rose slightly, since removing 72% of S1 changed the mix of what remains.

Fine-tuning still lifts RAOD 3.9× overall and 28× on the class it was almost
blind to. Ordinary tone-map-then-detect still wins by a wide margin: 59% higher
mAP in a tenth of the training time, and an untrained off-the-shelf detector
still comes close to fine-tuned RAOD.

The comparison isn't capacity matched, though. 44M parameters against 1M, and the
large detector is COCO-pretrained on millions of ordinary photographs containing
person and car specifically, while RAOD was pretrained on ROD: five traffic
classes from a car-mounted sensor. Neither advantage has anything to do with HDR.

---

## 3. Reading the numbers

**Scale.** Everything here is mAP × 100, the convention the papers use.
`pycocotools` prints the same values between 0 and 1, so 31.3 here shows up as
0.313 in the JSON under `results/`. Same numbers.

**The two tables above don't compare.** Section 1 scores 20 classes, section 2
scores 2. A 2-class average runs far higher for the same detector because it
leaves out the rare classes that drag an average down.

**Class counts.** mAP averages over classes that have ground truth, not classes
declared. I checked this rather than assuming it: the reported AP50 equals the
mean of the per-class values the scorer emits, to two decimals.

| table | declared | actually averaged |
|---|---|---|
| Section 1 | 20 | 18, since `cat` and `cow` have no test instances |
| Section 2 | 5 | 2, only Pedestrian and Car have data |

Section 2 uses two classes because RAOD's released head has five fixed outputs,
only `person` and `car` map onto it honestly, and the zero-shot rows can't change
that head at all. It isn't a choice. Every row uses the same two classes, so the
table is consistent within itself.

Rare classes are a real problem here. The test set has 2 `bus`, 2 `motorbike`, 5
`train` and 6 `horse`. A class with two examples scores close to random but
carries the same weight in the average as `person` with 3,719. That's what the
head-8 column is for.

**Which Reinhard?** The 2002 paper describes a global operator and a local one
with dodging and burning. Mine is the global form: one curve, `L/(1+L)` after
scaling to a target key, applied uniformly. Kocdemir lists both variants
separately elsewhere (CityScapes: local 33.2, global 32.7) but his HDR4RTT table
says only "Reinhard 29.6" without specifying. His row and mine may not be the
same operator. Durand is local in both.

---

## 4. Why the earlier numbers were higher

An earlier version of section 1 reported 30.4 to 38.3. Near-duplicate frames
inflated those, and I've removed them rather than showing both.

The problem turned up when I scored the best arm on each source separately:

| source | what it is | mAP, before dedup |
|---|---|---|
| S1 | video / rendered | **90.1** (AP50 **99.2**) |
| S3 | HDR video | 63.3 |
| S2 | bracketed photographs | 24.7 |

Nothing generalises at 99.2. Running Ulaş's `dedupe_hdr_frames.py`, which
compares each frame by SSIM against the last one it kept, showed why:

| SSIM threshold | total kept | S1 kept | S2 kept | S3 kept |
|---|---|---|---|---|
| **0.92 (used)** | 3,145 | **28%** | 100% | 99% |
| 0.60 | 2,822 | 7% | 98% | 98% |
| 0.30 | 2,582 | 2% | 89% | 94% |

S1's 1,289 images contain only a few hundred distinct scenes. That's where the
inflation came from. S3 survives at 99%, so its frames genuinely differ, which
means my earlier focus on S3 (the only source with readable frame numbers) was
aimed at the wrong place.

I picked 0.92 based on what it removes, not by chasing a target count. The
harsher settings get closer to Kocdemir's 1,871 images, but only by deleting S2
photographs taken on separate days, which have no duplicates to remove.

This is not Kocdemir's split. His list wasn't available, so this is my own
deduplication and the count differs from his.

Every arm dropped 5.9 to 7.0 mAP afterwards, and the ranking held. S2 barely
moved, 24.7 to 24.2, which is what you'd expect since deduplication took almost
nothing from it. That's a useful control: the drop came from removing duplicates,
not from something else changing.

### What it does to comparability

| | detector | mAP |
|---|---|---|
| Kocdemir, best classical (Mantiuk) | RetinaNet, his implementation | 31.3 |
| Kocdemir, Reinhard *(variant unstated)* | RetinaNet, his implementation | 29.6 |
| Kocdemir, his own method (TMO-GAN) | RetinaNet, his implementation | 31.6 |
| This work, best operator, deduplicated | RetinaNet R50-FPN v2 | **31.3** |

Before deduplication this work sat about 7 points above his with no explanation I
could defend. Afterwards the numbers land in the same range, which suggests the
gap was duplicate frames rather than anything about the method.

Both are RetinaNet, but not the same implementation. Torchvision's v2 recipe
postdates the thesis. That, the different splits, and the unknown class counts
are three separate reasons the absolute values needn't line up. Convergence, not
a head-to-head.

---

## 5. The dataset

I decoded and measured all 4,080 EXR files. Full detail in
[REPORT.md](hdr4rtt_analysis/REPORT.md).

- It's three sources glued together, and the EXR headers separate them for free:
  1,289 video/rendered frames, 1,489 bracketed photographs, 1,302 frames of one
  continuous HDR video.
- Neither value ceiling is over-exposure. About one pixel per image reaches it,
  because each image was individually normalised, so the stored values carry no
  absolute brightness meaning.
- 41% of images contain negative pixel values, down to −872. Clamp before any log
  or gamma step.
- 303 images (7.4%) form a separate very dark cluster, roughly 250× darker than
  the rest, and they hold 13.1% of all boxes.
- Median dynamic range is 4.11 decades, about 13.7 stops. Maximum 7.83 decades,
  26 stops.
- The shipped train/test split leaks. Every test frame in the video source has a
  training frame within ±3. A controlled two-model experiment puts the cost at
  +2.0 mAP, about 6% relative.

Per-image dynamic range statistics for all 4,080 files are in
[`hdr_stats_sources.csv`](hdr4rtt_analysis/hdr_stats_sources.csv), which lets you
break results down by scene difficulty.

The training set is also small. 2,994 images, but the video source contributes
949 frames from only about 6 continuous runs, so roughly 2,051 genuinely distinct
scenes. That's around 1.7% the size of a standard detection training set.

---

## 6. Two bugs

**Wrong input scale in evaluation.** RAOD's data loader divides pixel values by
255 inside `load_image`, so the network expects the range 0 to 1. The evaluation
script skipped that and fed 0 to 255. RAOD's tone-mapping module raises its input
to the power 1/gamma with gamma between 7 and 10.5, so every pixel saturated and
the frame went white.

The trap is that the same division sits commented out in `ValTransformRaw`, which
is the obvious place to look, because it has already happened one level up.

RAOD's own sample image gave 0 detections. After restoring one line it gave 5,
including a car at 0.892 confidence.

**The converter aimed at the opposite of what the model wants.** RAOD tone-maps
internally and expects dark, linear input averaging around 0.012. The previous
converter stretched images to fill 0 to 255, averaging 0.377, which is 32×
brighter than anything the model trained on. Every rescaling variant tried (p99,
adaptive percentile, log, histogram matching) shares the same "use the full
range" goal, so they all fail the same way.

---

## 7. Comparison with the MS thesis

### The two review questions

**1. Include the MS thesis results here.** Done, tables below.

Reading them turned up something that bears on question 2: the thesis already
contains a detector-swap experiment, and its two detectors disagree. The learned
method beats every classical operator under RetinaNet, 31.6 against 31.3, and
loses to four of them under Faster R-CNN, 27.7 against 29.5. Same method, same
data, opposite conclusion.

**2. Compare the 2023 paper and the thesis method fairly.** Partly.

| | status |
|---|---|
| Detector held fixed, front end varied | done, [section 1](#1-which-tone-map-works-best) |
| RAOD's module measured under that control | done, 28.7 mAP |
| Thesis TMO-GAN measured under that control | not possible, software lost |
| Indirect comparison via a shared reference | done, below |

The two learned methods have still never been run side by side.

### The thesis tables

From İ. H. Kocdemir's MS thesis and the corresponding Pattern Recognition Letters
172 (2023) 230–236 paper, on the dataset the thesis calls OOD: 20 Pascal VOC
classes, near-identical video frames removed, 1,491 train and 380 test, images at
1024×576.

Everything below is quoted from his thesis. I did not re-run any of it. All his
rows are fine-tuned rather than zero-shot, since the detector trains on each
tone-mapped image set, and the ✓ under "joint" marks where his generator trains
alongside it.

**Detector: RetinaNet** (thesis Table 4.2)

| front end | joint | on real | mAP | TMQI-Q |
|---|---|---|---|---|
| HDR (raw, no normalisation) | | | **26.3** | – |
| LDR | | | 28.2 | 76.1 |
| Reinhard | | | 29.6 | 89.6 |
| HDR with gamma | | | 29.8 | – |
| Fattal | | | 29.8 | 88.8 |
| Best TMO per picture | | | 30.0 | 94.9 |
| TMO-GAN | ✗ | | 30.0 | 94.6 |
| Ashikhmin | | | 30.1 | 88.4 |
| TMO-GAN + RetinaNet (COCO) | ✓ | ✓ | 30.2 | 94.2 |
| Durand | | | 30.6 | 89.0 |
| Std. LDR | | | 31.0 | 88.9 |
| Mantiuk | | | 31.3 | 86.5 |
| **TMO-GAN + RetinaNet (OOD)** | ✓ | ✓ | **31.6** | 94.5 |

**Detector: Faster R-CNN** (thesis Table 4.3)

| front end | joint | on real | mAP | TMQI-Q |
|---|---|---|---|---|
| HDR (raw, no normalisation) | | | **23.5** | – |
| LDR | | | 24.7 | 76.1 |
| TMO-GAN + Faster R-CNN (COCO) | ✓ | ✗ | 26.3 | 94.2 |
| TMO-GAN + Faster R-CNN (COCO) | ✓ | ✓ | 27.3 | 94.0 |
| HDR with gamma | | | 27.7 | – |
| **TMO-GAN + Faster R-CNN (OOD)** | ✓ | ✓ | 27.7 | 94.3 |
| Reinhard | | | 28.1 | 89.6 |
| Std. LDR | | | 28.3 | 88.9 |
| TMO-GAN | ✗ | | 28.6 | 94.6 |
| Durand | | | 28.8 | 89.0 |
| Ashikhmin | | | 28.9 | 88.4 |
| Mantiuk | | | 29.1 | 86.5 |
| **Fattal** | | | **29.5** | 88.8 |

Thesis Table 3.1 runs the same comparison on CityScapes, where HDR with gamma
(33.3) barely separates from Std. LDR (33.1). The thesis says plainly that no
advantage for HDR showed up there.

Two things stand out. Raw HDR is worst in both tables, 26.3 and 23.5, below plain
LDR, so normalisation rather than bit depth is what the detector needs. That
matches what I found independently. And the detector changes the conclusion, as
above.

### Comparing the two learned methods

TMO-GAN can't be re-run since its software was lost, and the obvious workaround
doesn't hold up either.

I tried using the operators common to both experiments as calibration anchors,
to map one scale onto the other. It fails, because the rankings invert:

| operator | his rank (RetinaNet) | rank here |
|---|---|---|
| Reinhard | worst tone map, 29.6 *(variant unstated)* | **best**, 31.3 *(global)* |
| Durand *(local in both)* | best of the three, 30.6 | **worst of the three**, 29.7 |
| HDR with gamma | 29.8 | 30.4 |

An inversion isn't an offset, so no scale factor reconciles them and his 31.6
can't be placed on this axis. I'm recording it as a negative result. It also
survives deduplication, which changed the values but not the ordering. Part of
the Reinhard gap could be global against local rather than real disagreement, see
[section 3](#3-reading-the-numbers).

What does work is comparing each method against the best classical operator in
its own experiment. That reference means something in both, and it cancels the
differences in split, resolution and detector version.

| learned method | source | own score | best classical, same table | gap |
|---|---|---|---|---|
| TMO-GAN + RetinaNet *(his impl.)* | his paper, quoted | 31.6 | 31.3 (Mantiuk) | **+0.3** |
| RAOD module + RetinaNet *(R50-FPN v2)* | measured here | 28.7 | 31.3 (Reinhard) | **−2.6** |

His method edged past the strongest classical operator available to it. RAOD's
module fell 2.6 behind the strongest one available here. About 2.9 mAP apart in
his favour, measured against a shared reference rather than on a shared scale.
Deduplication brought the absolute scales into the same range, so this rests on
firmer ground than it did.

Caveats. The best classical operator differs between the two, Mantiuk there and
Reinhard here, and I haven't implemented Mantiuk. If it beat Reinhard on this
data the gap would widen rather than narrow. The splits differ and the RetinaNet
implementations differ. This is a defensible indirect comparison, not a
head-to-head.

To close it properly you'd re-implement TMO-GAN from the paper, a generator and
discriminator trained jointly with the detector, and run it as a seventh front
end. Then both learned methods sit in one table under one detector.

### Section 2 doesn't compare with any of this

| | thesis | section 2 |
|---|---|---|
| classes | 20 | 2 (person, car) |
| test images | 380 | 576 |
| duplicate frames | removed entirely | removed, but by a different method |
| resolution | 1024×576 | 1280×1280 |
| detectors | RetinaNet, Faster R-CNN | RAOD YOLOX, Faster R-CNN v2 |

Putting 43.0 next to 31.6 would be meaningless.
[Section 1](#1-which-tone-map-works-best) is the table to set beside his, since
it uses 20 classes.

---

## 8. Method notes

**Picking the tone-mapping operator.** A weak choice here would have rigged the
comparison before any training started, so I scored eight candidates with a
COCO-pretrained detector first. The spread was 10% relative, which is enough to
have decided the outcome on its own. Gamma at the 99th percentile won and I used
it throughout. TMO-Det does the same thing, comparing six operators and reporting
the best. `pick_tmo.py` reproduces the sweep.

**Use batch size 4 on a 16 GB GPU, not 8.** Batch 8 needs 17.51 GB on a 17.1 GB
card. Windows won't raise an out-of-memory error for that; it pages GPU memory to
system RAM instead, so the run just gets about 8× slower with nothing in the log
to explain it. My batch-8 run reported an 8-day ETA. More detail in
[hdr4rtt_rod/README.md](hdr4rtt_rod/README.md).

---

## 9. Reproducing

You need PyTorch and OpenCV. RAOD's own repository supplies the model code. Data
paths sit at the top of each script.

Audit the dataset:

```bash
python hdr4rtt_analysis/scripts/scan_hdr.py
```

Deduplicate, then build 20-class annotations from what survives:

```bash
python hdr4rtt_rod/build_deduped_split.py
```

Build one image set per front end:

```bash
python hdr4rtt_rod/make_frontends.py
```

Train an arm, then score it. Repeat per front end:

```bash
python hdr4rtt_rod/train_frontend.py --arm reinhard --arch retinanet --epochs 10 --batch 4
```

```bash
python hdr4rtt_rod/eval_frontend.py --arm reinhard --arch retinanet --batch 4
```

For the RAOD pipeline in section 2:

```bash
python hdr4rtt_rod/convert_hdr4rtt_to_rod.py --gain 0.02
```

```bash
python hdr4rtt_rod/smoke_test_train.py --workers 0
```

---

## 10. Layout and open questions

| path | contents |
|---|---|
| `hdr4rtt_analysis/` | dataset audit: scripts, report, per-image statistics |
| `hdr4rtt_rod/` | conversion, annotations, splits, training configs, evaluation |
| `hdr4rtt_rod/results/` | every reported number, as saved JSON |
| `hdr4rtt_rod/viz/` | ground truth and predictions drawn on converted images |
| `hdr4rtt_rod/progress_note.html` | step-by-step record of the work |

Checkpoints, converted imagery and the dataset itself are excluded on purpose,
see [.gitignore](.gitignore).

Open questions, roughly in order of how much they'd change:

1. Match model capacity honestly, either shrinking the tone-mapped arm or growing
   the HDR arm, so the HDR against tone-mapped question isn't confounded by a 44×
   difference in parameters.
2. Re-implement TMO-GAN and run it as a seventh front end, so the two learned
   methods can be compared directly instead of through a shared reference.
3. Run the local Reinhard variant, to find out whether the ranking inversion
   against the thesis is real or just two different operators being compared.
4. Chase the failure analysis. Holding object size fixed, misses track local
   contrast rather than brightness: objects with under a decade of range inside
   their own box are missed 66% of the time against 12% for objects with over
   three, and flat scenes are harder than high-contrast ones. Every operator here
   compresses global range; none targets contrast at the scale objects actually
   occupy.

---

## 11. Licence and attribution

Code © 2026 Sana Niroomand and the OGAM Research Laboratory, Middle East
Technical University (METU), under the [MIT licence](LICENSE). This is research
code, provided as is, with no warranty, and not meant for production or
safety-related use.

Full third-party attribution is in [NOTICE](NOTICE). In short:

| what | relationship | terms |
|---|---|---|
| **RAOD** (Xu et al., CVPR 2023) | two config files are modified derivatives; other files call it at runtime | © Huawei + Megvii, Apache 2.0, see [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt) |
| RAOD model weights | not redistributed | original terms |
| **TMO-Det** (Kocdemir et al.) | no code used, results quoted only | © authors and publisher |
| Reinhard, Durand operators | algorithms re-implemented from the papers | methods credited to their authors |
| torchvision, OpenCV, pycocotools | called as libraries | BSD / Apache 2.0 |
| HDR4RTT / OOD dataset | not redistributed in any form | obtain from its creators |

The two derivative files, `cfg_hdr4rtt_rod.py` and `cfg_hdr4rtt_rod_original.py`,
stay under Apache 2.0 and carry headers listing exactly what I changed, as that
licence requires. Everything else is original work under MIT.

The tone-mapping operators here are simplified re-implementations that I tuned
against this dataset. Don't treat them as reference implementations of the
published methods. Any weakness in them is mine, not the original authors'.

Please cite the original work rather than this repository for the methods it
builds on:

> R. Xu, C. Chen, J. Peng, C. Li, Y. Huang, F. Song, Y. Yan, Z. Xiong.
> *Toward RAW Object Detection: A New Benchmark and a New Model.* CVPR 2023.

> İ. H. Kocdemir, A. Koz, A. O. Akyuz, et al.
> *TMO-Det: Deep tone-mapping optimized with and for object detection.*
> Pattern Recognition Letters 172 (2023) 230–236.

Tables from the thesis and paper appear here for academic comparison, with
attribution. They're quoted, not reproduced experimentally, since the original
software was lost and those numbers couldn't be re-run. I've said so wherever
they appear.

Any errors here are mine, not the original authors'.
