# HDR object detection on HDR4RTT

Object detection on high dynamic range imagery, using the HDR4RTT dataset: 4,080
OpenEXR images, 32,964 boxes, 20 Pascal VOC classes.

Three parts: an audit of the dataset, a rebuilt pipeline for running
[RAOD](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.pdf)
(CVPR 2023) on it, and a comparison of six tone mapping methods behind one fixed
detector.

[Progress note](hdr4rtt_rod/progress_note.html) ·
[Dataset audit](hdr4rtt_analysis/REPORT.md) ·
[Pipeline details](hdr4rtt_rod/README.md)

> **Work in progress, not peer reviewed.** Internship project, numbers are
> preliminary. Please ask before citing. No warranty, see [LICENSE](LICENSE).

---

## Which tone map works best?

HDR data has to be compressed before a detector can read it. This keeps the
detector fixed and changes only that step, so any difference comes from the tone
map alone.

Every row: RetinaNet R50-FPN v2, COCO-pretrained with a fresh 20-class head,
fine-tuned 10 epochs at batch 4. None is zero-shot. Deduplicated split, 2,506
train and 633 test, 20 classes.

![Five tone maps applied to the same two HDR photographs](hdr4rtt_rod/viz/tone_mapping_comparison.jpg)

The left column has no tone curve, and in the workshop scene the welding arc
takes the whole output range, leaving everything else black. The four to its
right are the same data under different curves.

| # | front end | whose method | mAP | AP50 | AP50 head-8 | S2 only |
|---|---|---|---|---|---|---|
| 1 | Reinhard *(global)* | classical, 2002 | **31.3** | 49.7 | **59.9** | 24.2 |
| 2 | HDR with gamma | standard display curve | 30.4 | 48.5 | 56.7 | 22.9 |
| 3 | Log compression | simple formula | 29.9 | 48.8 | 58.4 | 22.5 |
| 4 | Durand *(local)* | classical, 2002 | 29.7 | **50.6** | 58.1 | 22.7 |
| 5 | RAOD module | learned, CVPR 2023 | 28.7 | 46.5 | 54.1 | 22.0 |
| 6 | no tone curve | control | 24.4 | 41.8 | 49.3 | 18.8 |

Row 5 trains the learned module jointly with the detector, the way RAOD intends.
"head-8" averages only the eight classes with enough test instances to be stable.
"S2 only" scores the bracketed photographs alone, the one source where duplicates
are impossible.

**Skipping the tone curve costs 6.9 mAP** and is worst on every source. Kocdemir
found the same under both his detectors, where raw HDR scored 26.3 and 23.5.

**The choice of curve matters much less than making one.** The four sensible
operators sit within 1.6 points of each other, against a 6.9 point gap to using
nothing.

**RAOD's learned module lost to a one-line formula from 2002**, finishing fifth
of six. I suspected my own settings, so I retrained it three more times changing
one thing each: full learning rate, random init instead of RAOD's weights, and
input rescaled to the level those weights expect. All three still landed below
every fixed operator.

That isn't evidence the method is wrong. RAOD pairs it with a 1M-parameter
detector, where the front end has far more work to do; a 38M-parameter detector
may already handle internally whatever the module supplied. Kocdemir's tables
point the same way from the other side, and whether a learned tone map helps
seems to depend on what sits behind it.

---

## Detector comparison

A different question: how do the detectors compare? Both the input and the
detector change here, so this can't separate the two. Same deduplicated split,
but 2 classes rather than 20, so it doesn't compare with the table above either.

576 test images, 3,785 boxes, person and car only.

| detector | input | params | training | mAP | AP50 | Pedestrian | Car |
|---|---|---|---|---|---|---|---|
| RAOD (YOLOX-nano + its module) | HDR | 1M | zero-shot | 7.0 | 15.3 | 1.8 | 28.8 |
| RAOD (YOLOX-nano + its module) | HDR | 1M | fine-tuned | 27.0 | 55.0 | 49.7 | 60.2 |
| RetinaNet R50-FPN v2 | tone-mapped | 38M | zero-shot | 24.2 | 50.7 | 38.5 | 62.9 |
| Faster R-CNN R50-FPN v2 | tone-mapped | 44M | zero-shot | 26.4 | 54.7 | 44.6 | 64.8 |
| Faster R-CNN R50-FPN v2 | tone-mapped | 44M | fine-tuned | **43.0** | **74.0** | 68.8 | 79.1 |

Ordinary tone-map-then-detect wins by a wide margin: 59% higher mAP in a tenth of
the training time, and an untrained off-the-shelf detector comes close to a
fine-tuned RAOD.

The comparison isn't capacity matched, though. 44M parameters against 1M, and the
large detector is COCO-pretrained on ordinary photographs containing person and
car specifically, while RAOD was pretrained on five traffic classes from a
car-mounted sensor. Neither advantage has anything to do with HDR.

---

## Reading the numbers

Everything is mAP × 100, the convention the papers use. `pycocotools` prints the
same values between 0 and 1, so 31.3 here is 0.313 in the JSON under `results/`.

**The two tables above don't compare with each other.** 20 classes against 2. A
2-class average runs far higher for the same detector.

**mAP averages over classes that have ground truth, not classes declared.** The
20-class tables therefore average over 18, since `cat` and `cow` have no test
instances, and the 2-class table averages over 2 of 5 declared. That second one
isn't a choice: RAOD's released head has five fixed outputs and only `person` and
`car` map onto it honestly.

Rare classes are a real problem. The test set has 2 `bus`, 2 `motorbike`, 5
`train`, 6 `horse`. A class with two examples scores close to random but carries
the same weight as `person` with 3,719. That's what the head-8 column is for.

**Reinhard here is the global variant**, one curve applied uniformly. Kocdemir
lists global and local separately elsewhere (CityScapes: 32.7 and 33.2) but his
HDR4RTT table says only "Reinhard 29.6", so his row and mine may not be the same
operator. Durand is local in both.

---

## Deduplication

One group of images, S1, is mostly near-duplicates: 1,289 files that reduce to
367 distinct scenes. Before removing them the best arm scored 90.1 mAP on S1
alone, at 99.2 AP50, which nothing generalises to.

I used Ulaş's `dedupe_hdr_frames.py`, which compares each frame by SSIM against
the last one it kept, at threshold 0.92. That keeps 3,145 of 4,080 images: 28% of
S1, 100% of S2, 99% of S3. I chose the threshold by what it removes rather than
by chasing a count. Harsher settings get closer to Kocdemir's 1,871 images but
only by deleting S2 photographs taken on separate days, which have no duplicates
to remove.

**This is not Kocdemir's split.** His list wasn't available, so this is my own
deduplication and the count differs.

Every arm dropped 5.9 to 7.0 mAP afterwards and the ranking held. Two things
confirm the drop came from duplicates rather than something else: S2 barely moved
(24.7 to 24.2), having lost almost nothing to deduplication, and in the detector
table the zero-shot rows barely moved while the fine-tuned rows fell 8 to 9
points. A model that never trained on this data can't benefit from duplicates in
the test set.

Note S3 survives at 99%, so its frames genuinely differ. My earlier focus on S3,
the only source with readable frame numbers, was aimed at the wrong place.

---

## The dataset

I decoded and measured all 4,080 EXR files. Full detail in
[REPORT.md](hdr4rtt_analysis/REPORT.md); the points that changed decisions here:

- Three sources glued together, and the EXR headers separate them for free: 1,289
  video frames, 1,489 bracketed photographs, 1,302 frames of one HDR video.
- Neither value ceiling is over-exposure. About one pixel per image reaches it,
  because each image was individually normalised, so the stored values carry no
  absolute brightness meaning.
- 41% of images contain negative pixel values, down to −872. Clamp before any log
  or gamma step.
- 303 images (7.4%) are a separate very dark cluster, roughly 250× darker than
  the rest, holding 13.1% of all boxes.
- Median dynamic range 4.11 decades, about 13.7 stops. Maximum 7.83.
- The shipped train/test split leaks. Every test frame in the video source has a
  training frame within ±3, worth +2.0 mAP in a controlled two-model experiment.

Per-image statistics for all 4,080 files are in
[`hdr_stats_sources.csv`](hdr4rtt_analysis/hdr_stats_sources.csv).

---

## Two bugs in the earlier pipeline

**Wrong input scale in evaluation.** RAOD's loader divides by 255 inside
`load_image`, so the network expects 0 to 1. The evaluation script skipped that
and fed 0 to 255. RAOD's tone-mapping module raises its input to the power
1/gamma with gamma between 7 and 10.5, so everything saturated and the frame went
white. The trap is that the same division sits commented out in
`ValTransformRaw`, the obvious place to look, because it already happened one
level up. RAOD's own sample image gave 0 detections; after restoring one line it
gave 5.

**The converter aimed at the opposite of what the model wants.** RAOD tone-maps
internally and expects dark, linear input averaging around 0.012. The previous
converter stretched images to fill 0 to 255, averaging 0.377, 32× brighter than
anything it trained on. Every rescaling variant tried shares the same "use the
full range" goal, so they all fail the same way.

---

## Comparison with the MS thesis

From İ. H. Kocdemir's MS thesis and the corresponding Pattern Recognition Letters
172 (2023) 230–236 paper, on the dataset the thesis calls OOD: 20 Pascal VOC
classes, near-identical frames removed, 1,491 train and 380 test, 1024×576.

Everything below is quoted from his thesis. I did not re-run any of it. All his
rows are fine-tuned, and ✓ under "joint" marks where his generator trains
alongside the detector.

**RetinaNet** (Table 4.2) · **Faster R-CNN** (Table 4.3)

| front end | joint | RetinaNet mAP | Faster R-CNN mAP |
|---|---|---|---|
| HDR (raw, no normalisation) | | **26.3** | **23.5** |
| LDR | | 28.2 | 24.7 |
| Reinhard | | 29.6 | 28.1 |
| HDR with gamma | | 29.8 | 27.7 |
| Fattal | | 29.8 | **29.5** |
| Best TMO per picture | | 30.0 | 29.3 |
| TMO-GAN | ✗ | 30.0 | 28.6 |
| Ashikhmin | | 30.1 | 28.9 |
| Durand | | 30.6 | 28.8 |
| Std. LDR | | 31.0 | 28.3 |
| Mantiuk | | 31.3 | 29.1 |
| **TMO-GAN (OOD)** | ✓ | **31.6** | 27.7 |

His two tables are merged here into one. Three TMO-GAN ablation rows that
differ between them are omitted for space: COCO-pretrained variants scoring 30.2
under RetinaNet, and 26.3 and 27.3 under Faster R-CNN. See the thesis for those.

Two things stand out. **Raw HDR is worst in both**, below plain LDR, so
normalisation rather than bit depth is what the detector needs. And **the
detector changes the conclusion**: his learned method beats every classical
operator under RetinaNet, 31.6 against 31.3, but loses to four of them under
Faster R-CNN, 27.7 against 29.5. Same method, same data, opposite answer.

Thesis Table 3.1 runs the same comparison on CityScapes, where HDR with gamma
(33.3) barely separates from Std. LDR (33.1), and the thesis says plainly that no
advantage for HDR showed up there.

### Status of the two review questions

**Include the thesis results here.** Done, above.

**Compare the 2023 paper and the thesis method fairly, given they used different
detectors.** Partly. The detector is now held fixed and RAOD's module measured
under that control at 28.7 mAP. TMO-GAN can't be measured the same way because
its software was lost, so what follows is an indirect comparison rather than a
head-to-head.

### Comparing the two learned methods

TMO-GAN can't be re-run since its software was lost, so the two learned methods
have never been measured side by side.

I tried using the operators common to both experiments to map one scale onto the
other. It fails: the rankings invert. Reinhard is his weakest tone map and my
strongest, Durand his strongest of the three and my weakest. An inversion isn't
an offset, so no scale factor reconciles them. Recording it as a negative result.
Part of the Reinhard gap may be global against local rather than real
disagreement.

What does work is comparing each method against the best classical operator in
its own experiment, which cancels the differences in split and detector version.
His method edged 0.3 past its best classical operator; RAOD's module fell 2.6
behind mine. About 2.9 apart in his favour, measured against a shared reference
rather than on a shared scale.

Deduplication helped here. My best operator now scores 31.3 against his best
classical 31.3, where before I sat about 7 points above him with no explanation I
could defend. Both are RetinaNet but not the same implementation, and the splits
and class counts still differ, so this is convergence rather than a head-to-head.

To close it properly you'd re-implement TMO-GAN and run it as a seventh front end.

---

## Where the detector fails

I classified every prediction and every missed box, then joined each box to
statistics measured from the original EXR. Details in
[`analyze_failures.py`](hdr4rtt_rod/analyze_failures.py) and the CSV under
`results/failures/`.

Of predictions above 0.3 confidence: 58.8% correct, **18.8% find the object but
draw a poor box**, 15.2% land on background, 4.6% wrong class, 2.6% duplicate.
38% of ground-truth boxes are missed entirely. The poor-box category is the
largest single fixable one.

Holding object size fixed, **misses track local contrast rather than darkness**:

| dynamic range inside the object's own box | missed (objects 5–10% of image side) |
|---|---|
| under 1 decade | **66%** |
| over 3 decades | **12%** |

Flat scenes are harder than high-contrast ones, and objects at the same
brightness as their surroundings are missed more often than objects that are much
darker or much brighter. The worst image in the set is a warehouse of 254
near-identical dark bottles, none found, at a median object size of 6.2% of the
image side, so size doesn't explain that one.

Every operator here compresses global range. None targets contrast at the scale
objects actually occupy.

---

## Input geometry: three attempts, one small gain

The failure analysis blames poor boxes and misses, which points at the input
resolution and the anchor boxes. I tried all three geometric knobs. Only the
first helped.

Baseline throughout: Reinhard, RetinaNet R50-FPN v2, fine-tuned 10 epochs at
batch 4 on the deduplicated split.

| change | anchor fit | mAP | AP50 | AP75 | small | med | large | AR100 |
|---|---|---|---|---|---|---|---|---|
| none, images reach the net at 800px | 75.3% | 31.3 | 49.7 | 31.4 | 6.2 | 18.3 | **42.9** | 44.5 |
| **short side raised to 1280** | 82.9% | **31.7** | **53.4** | **31.5** | 8.9 | 23.4 | 41.7 | 44.3 |
| 1280, anchor ratios 1, 2.5, 5 | 94.6% | 30.1 | 49.3 | 30.7 | 9.0 | 23.1 | 40.0 | 44.5 |
| 1280, anchor ratios 0.5, 1.6, 6.4 | 96.0% | 30.4 | 50.5 | 29.6 | **11.0** | 22.7 | 40.2 | **46.9** |
| native aspect ratio, long side 2048 | — | 31.4 | 51.6 | 30.7 | 10.1 | **23.9** | 37.0 | 44.8 |

**The first row was hiding a defect.** torchvision rescales its input so the short
side equals `min_size`, 800 by default, and nothing in the pipeline mentioned it.
Every 1280×1280 image had been arriving at the network as 800×800. Raising that
limit is the only change here that gained anything: +0.4 mAP, +3.7 AP50, and
small-object AP from 6.2 to 8.9. It is now `--min_size` on both scripts, stored in
the checkpoint so evaluation cannot silently disagree with training.

### The anchors fit badly, and fixing that made things worse

RetinaNet matches an object to a template box, an anchor, when the two overlap by
IoU 0.5. An object no anchor reaches cannot become a positive example, so it is
close to unlearnable. [`anchor_fit.py`](hdr4rtt_rod/anchor_fit.py) measures that
share before any training, which costs seconds rather than a GPU-hour.

It turns up a genuine mismatch. **85% of the objects here are taller than wide,
median height/width 2.39 and upper quartile 3.53, while torchvision's stock ratios
stop at 2.0.** Those defaults were fitted for COCO, not for people and bottles.
Retuning them lifts the matchable share from 82.9% to 96.0%.

The detector got worse anyway: 31.7 to 30.4 mAP.

What the retuned anchors do is find more and localise worse. Recall rises from
44.3 to 46.9, small-object AP from 8.9 to 11.0, while AP75, which demands tight
boxes, falls from 31.5 to 29.6. mAP averages overlap thresholds from 0.5 to 0.95,
so the sloppier boxes outweigh the extra finds.

The likely cause is structural. Only the classification head is rebuilt for 20
classes; the box regression head is inherited from COCO pretraining, where it
learned offsets tied to the default anchor shapes. New shapes invalidate those
priors, and 10 epochs on 2,506 images will not retrain them. Adding a fourth,
taller ratio would test this properly, but the anchor count fixes the regression
head's output width, so that means rebuilding it too. Untested.

One measurement worth keeping: scaling the anchor *sizes* to match the higher
resolution, the textbook move for preserving relative coverage, drops small-object
matching to **0.0%**. The smallest anchor moves from 32px to 51px and nothing
under 32² in area can reach IoU 0.5 against it. That configuration was abandoned
two epochs in rather than trained to completion.

### What this says

Resolution, aspect ratio and anchors all land within 1.3 mAP of each other, and
the headline barely moved. **Input geometry is not what limits this model.** That
agrees with the failure analysis, which pointed at local contrast rather than
anything geometric.

Two caveats against reading this as a dead end. The small-object and recall gains
are real, so the retuned-ratio model is the better choice if small objects matter
more than tight boxes. And the anchor-fit share is a ceiling, not a score: it says
what is reachable, never what gets learned. I predicted mAP from it and was wrong.

---

## Reproducing

You need PyTorch and OpenCV. RAOD's own repository supplies the model code. Data
paths sit at the top of each script.

```bash
python hdr4rtt_analysis/scripts/scan_hdr.py
```

```bash
python hdr4rtt_rod/build_deduped_split.py
```

```bash
python hdr4rtt_rod/make_frontends.py
```

```bash
python hdr4rtt_rod/train_frontend.py --arm reinhard --arch retinanet --epochs 10 --batch 4
```

```bash
python hdr4rtt_rod/eval_frontend.py --arm reinhard --arch retinanet --batch 4
```

**Pass `--min_size 1280 --max_size 2133` to train at full resolution.** Left at the
defaults, torchvision shrinks every 1280x1280 image to 800x800 before the backbone
sees it, silently. Evaluation reads the value back from the checkpoint, so the two
cannot drift apart. `--anchor_ratios` and `--anchor_scale` retune the anchors; both
are recorded the same way, and both lost to the defaults.

```bash
python hdr4rtt_rod/anchor_fit.py --search
```

**Use batch size 4 on a 16 GB GPU, not 8.** Batch 8 needs 17.51 GB on a 17.1 GB
card, and Windows won't raise an out-of-memory error for that. It pages GPU
memory to system RAM, so the run just gets about 8× slower with nothing in the
log to explain it. My batch-8 run reported an 8-day ETA.

One more method note: a weak tone map would have rigged the comparison before
training started, so I scored eight candidates with a COCO-pretrained detector
first. The spread was 10% relative, enough to have decided the outcome on its
own. `pick_tmo.py` reproduces that sweep.

---

## Layout

| path | contents |
|---|---|
| `hdr4rtt_analysis/` | dataset audit: scripts, report, per-image statistics |
| `hdr4rtt_rod/` | conversion, annotations, splits, configs, evaluation |
| `hdr4rtt_rod/results/` | every reported number, as saved JSON |
| `hdr4rtt_rod/viz/` | ground truth and predictions drawn on converted images |
| `hdr4rtt_rod/anchor_fit.py` | anchor coverage against the real box shapes, before training |
| `hdr4rtt_rod/progress_note.html` | step-by-step record of the work |

Checkpoints, converted imagery and the dataset itself are excluded on purpose,
see [.gitignore](.gitignore).

**Open questions**

1. Match model capacity honestly, either shrinking the tone-mapped arm or growing
   the HDR arm, so the comparison isn't confounded by a 44× parameter difference.
2. Re-implement TMO-GAN as a seventh front end, so the two learned methods can be
   compared directly.
3. Run the local Reinhard variant, to find out whether the ranking inversion
   against the thesis is real or just two different operators.
4. Target local contrast at object scale rather than global range, which is what
   the failure analysis points at, and is the one lead the geometry experiments
   did not rule out.
5. Rebuild the box regression head alongside the classification head, so anchor
   ratios can be retuned without discarding COCO's regression priors. That is the
   test the anchor result needs and did not get.

---

## Licence and attribution

Code © 2026 Sana Niroomand and the OGAM Research Laboratory, Middle East
Technical University (METU), under the [MIT licence](LICENSE). Research code,
provided as is, not meant for production use.

Full attribution in [NOTICE](NOTICE). In short:

| what | relationship | terms |
|---|---|---|
| **RAOD** (Xu et al., CVPR 2023) | two config files are modified derivatives; other files call it at runtime | © Huawei + Megvii, Apache 2.0, see [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt) |
| RAOD model weights | not redistributed | original terms |
| **TMO-Det** (Kocdemir et al.) | no code used, results quoted only | © authors and publisher |
| Reinhard, Durand operators | algorithms re-implemented from the papers | methods credited to their authors |
| torchvision, OpenCV, pycocotools | called as libraries | BSD / Apache 2.0 |
| HDR4RTT / OOD dataset | not redistributed in any form | obtain from its creators |

The two derivative files, `cfg_hdr4rtt_rod.py` and `cfg_hdr4rtt_rod_original.py`,
stay under Apache 2.0 with headers listing what I changed. Everything else is
original work under MIT.

The tone-mapping operators here are simplified re-implementations that I tuned
against this dataset. Don't treat them as reference implementations. Any weakness
in them is mine, not the original authors'.

Please cite the original work rather than this repository:

> R. Xu, C. Chen, J. Peng, C. Li, Y. Huang, F. Song, Y. Yan, Z. Xiong.
> *Toward RAW Object Detection: A New Benchmark and a New Model.* CVPR 2023.

> İ. H. Kocdemir, A. Koz, A. O. Akyuz, et al.
> *TMO-Det: Deep tone-mapping optimized with and for object detection.*
> Pattern Recognition Letters 172 (2023) 230–236.

Thesis tables appear here for academic comparison with attribution. They're
quoted, not reproduced experimentally, since the original software was lost.

Any errors here are mine, not the original authors'.
