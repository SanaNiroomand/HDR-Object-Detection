# HDR object detection on HDR4RTT

Running object detectors on high-dynamic-range imagery, using the HDR4RTT dataset
(4,080 OpenEXR images, 32,964 boxes, 20 Pascal VOC classes).

Three things are in here: a full audit of the dataset, a rebuilt pipeline for
running [RAOD](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Toward_RAW_Object_Detection_A_New_Benchmark_and_a_New_CVPR_2023_paper.pdf)
(CVPR 2023) on it, and a controlled comparison of six tone-mapping front ends
behind a single fixed detector.

**[→ Step-by-step progress note](hdr4rtt_rod/progress_note.html)** ·
**[→ Dataset audit](hdr4rtt_analysis/REPORT.md)** ·
**[→ Pipeline details](hdr4rtt_rod/README.md)**

> **⚠️ Work in progress, not peer reviewed.** An ongoing internship project.
> Results are preliminary and may change or be withdrawn; one table is explicitly
> marked as superseded. Please do not cite these as established findings — get in
> touch first. No warranty (see [LICENSE](LICENSE)).

**Contents**

1. [Main result: which tone map?](#1-main-result-which-tone-map)
2. [Detector comparison](#2-detector-comparison)
3. [How to read these numbers](#3-how-to-read-these-numbers)
4. [Why the earlier numbers were higher](#4-why-the-earlier-numbers-were-higher)
5. [What the dataset audit found](#5-what-the-dataset-audit-found)
6. [Two bugs that invalidated earlier work](#6-two-bugs-that-invalidated-earlier-work)
7. [Comparison with the MS thesis](#7-comparison-with-the-ms-thesis)
8. [Method notes](#8-method-notes)
9. [Reproducing](#9-reproducing)
10. [Layout and open questions](#10-layout-and-open-questions)
11. [Licence and attribution](#11-licence-and-attribution)

---

## 1. Main result: which tone map?

The detector is held fixed and **only the tone mapping changes**. Same data,
split, schedule, batch size and augmentation throughout.

* **Detector, every row:** RetinaNet R50-FPN v2 (torchvision
  `retinanet_resnet50_fpn_v2`), COCO-pretrained, fresh 20-class head
* **Training, every row:** fine-tuned 10 epochs at batch 4 — **none is zero-shot**
* **Split:** deduplicated, 2,506 train / 633 test, all 20 classes
* **Source:** every number measured here

![Five tone maps applied to the same two HDR photographs](hdr4rtt_rod/viz/tone_mapping_comparison.jpg)

*The same two photographs through the five fixed front ends. Left column is the
raw conversion with no tone curve: in the workshop scene the welding arc consumes
the entire output range and everything else goes black. The four to its right are
identical data under different curves. The learned method is not shown because it
produces a different curve for every photograph.*

| # | front end | whose method | mAP | AP50 | AP50 head-8 | S2 only |
|---|---|---|---|---|---|---|
| 1 | **Reinhard** *(global)* | classical, 2002 | **31.3** | 49.7 | **59.9** | 24.2 |
| 2 | HDR with gamma | standard display curve | 30.4 | 48.5 | 56.7 | 22.9 |
| 3 | Log compression | simple formula | 29.9 | 48.8 | 58.4 | 22.5 |
| 4 | Durand *(local)* | classical, 2002 | 29.7 | **50.6** | 58.1 | 22.7 |
| 5 | **RAOD module** | **learned, CVPR 2023** | 28.7 | 46.5 | 54.1 | 22.0 |
| 6 | **no tone curve** | control | **24.4** | 41.8 | **49.3** | **18.8** |

*Row 5 trains the learned module **jointly** with the detector, as RAOD intends.
**AP50 head-8** averages only the eight classes with enough test instances to be
stable; **S2 only** scores the bracketed-photograph source alone, where no
duplicates are possible — the most conservative figure available.*

### What it shows

**Applying no tone curve is by far the worst** — 6.9 mAP below the best operator,
and worst on every source. This independently replicates the thesis finding,
where raw HDR was also worst under both of its detectors (26.3 and 23.5).

**Which curve you choose barely matters.** The four sensible operators span 29.7
to 31.3, a 1.6-point range, against a 6.9-point gap to applying none at all.

**RAOD's learned module did not beat a formula from 2002.** Reinhard leads it by
2.6 mAP; it places fifth of six. Three variants isolating one suspect setting
each — full learning rate, random initialisation instead of RAOD's released
weights, and input rescaled to the level those weights were fitted at — were also
tried, and all landed below every sensible fixed operator. *(Those three ran on
the earlier, un-deduplicated split, so their absolute values are inflated the
same way; what carries over is that none overtakes a fixed operator.)*

**This does not show RAOD's method is wrong.** RAOD pairs the module with a
1M-parameter detector, where a strong front end plausibly matters far more; a
38M-parameter detector may already absorb internally whatever the module was
supplying. The thesis shows the same pattern from the other side — its learned
method beats every classical operator under RetinaNet and loses to four of them
under Faster R-CNN. **The value of a learned tone map appears to depend on the
detector behind it.**

---

## 2. Detector comparison

A different question: how do the detectors themselves compare? Here the input
representation and the detector both change, so this table cannot separate them —
that is what section 1 is for.

> **⚠️ Superseded. These absolute values are inflated.** This table predates the
> [deduplication](#4-why-the-earlier-numbers-were-higher) and its test set still
> contains the near-duplicate S1 frames. Comparisons *between* rows remain valid,
> since every row faced identical images, but read the absolute numbers as
> roughly 6–7 mAP high. A re-run on the clean split is the first
> [open question](#10-layout-and-open-questions).

Measured here, 779-image test set, 4,082 boxes, **person and car only**.

| detector | input | params | training | mAP | AP50 | Pedestrian | Car |
|---|---|---|---|---|---|---|---|
| RAOD (YOLOX-nano + its module) | HDR | 1M | zero-shot | 5.9 | 12.8 | 1.8 | 23.7 |
| RAOD (YOLOX-nano + its module) | HDR | 1M | fine-tuned, 4.6 h | 34.9 | 61.3 | 60.2 | 62.4 |
| RetinaNet R50-FPN v2 | tone-mapped | 38M | zero-shot | 29.6 | 57.4 | 50.2 | 64.6 |
| Faster R-CNN R50-FPN v2 | tone-mapped | 44M | zero-shot | 31.0 | 60.6 | 56.1 | 65.0 |
| **Faster R-CNN R50-FPN v2** | tone-mapped | 44M | **fine-tuned, 0.5 h** | **51.7** | **79.4** | 77.4 | 81.4 |

*RAOD in its released configuration: YOLOX at depth 0.33 / width 0.25 with
depthwise convolutions, plus its Adaptive_Module tone mapper, 1.0M parameters
total.*

Fine-tuning lifts RAOD 5.9× overall and 33× on the class it was effectively blind
to. But the comparison arm is the interesting part: conventional
tone-map-then-detect reaches **48% higher mAP in a tenth of the training time**,
and an untrained off-the-shelf detector already matches fine-tuned RAOD's AP50.

**This is not capacity-matched** — 44M parameters against 1M — and the large
detector is COCO-pretrained on millions of everyday photographs containing
exactly "person" and "car", while RAOD was pretrained on ROD: five traffic
classes from a car-mounted sensor. Both advantages favour the tone-mapped arm for
reasons unrelated to HDR.

---

## 3. How to read these numbers

**Scale.** Every number here is **mAP × 100** (percentage points), the convention
the papers use. `pycocotools` prints the same values in 0–1, so `31.3` here
appears as `0.313` in the saved JSON under `results/`. Identical numbers.

**The two tables above are not comparable with each other.** Section 1 scores 20
classes; section 2 scores 2. A 2-class average is far higher for the same
detector, because it excludes the rare classes that drag an average down.

**Class counts.** mAP averages over classes that *have ground truth*, not over
classes declared. Verified: the reported AP50 equals the mean of the per-class
AP50 values the scorer emits, to two decimals.

| table | declared | actually averaged |
|---|---|---|
| Section 1 | 20 | **18** — `cat` and `cow` have no test instances |
| Section 2 | 5 | **2** — only Pedestrian and Car have data |

Section 2's 2-class scope is **not a choice**: RAOD's released head has exactly
five fixed outputs, only `person` and `car` map onto it honestly, and the
zero-shot rows cannot change that head at all. Every row uses the same two
classes, so the table is internally consistent.

Rare classes matter here: this test set contains 2 `bus`, 2 `motorbike`, 5
`train` and 6 `horse` instances. A class with two examples scores almost
arbitrarily yet weighs as much as `person` with 3,719 — which is why the
**head-8** column exists.

**Which Reinhard?** The 2002 paper defines a **global** operator and a **local**
one with dodging-and-burning. The implementation here is the **global** form: one
curve, `L/(1+L)` after scaling to a target key, applied uniformly. The thesis
lists both variants separately elsewhere (CityScapes: local 33.2, global 32.7)
but its HDR4RTT table gives only "Reinhard 29.6" without saying which — **so that
row and ours may not be the same operator.** Durand is local in both.

---

## 4. Why the earlier numbers were higher

An earlier version of section 1 reported 30.4 to 38.3. Those numbers were
inflated by near-duplicate frames and have been **removed rather than kept
alongside**.

The problem showed up when the best arm was scored on each source separately:

| source | what it is | mAP, before dedup |
|---|---|---|
| S1 | video / rendered | **90.1** (AP50 **99.2**) |
| S3 | HDR video | 63.3 |
| S2 | bracketed photographs | 24.7 |

99.2 is not a plausible generalisation result. Running Ulaş's
`dedupe_hdr_frames.py` (SSIM against the last kept frame) confirmed why:

| SSIM threshold | total kept | S1 kept | S2 kept | S3 kept |
|---|---|---|---|---|
| **0.92 (used)** | 3,145 | **28%** | 100% | 99% |
| 0.60 | 2,822 | 7% | 98% | 98% |
| 0.30 | 2,582 | 2% | 89% | 94% |

**S1's 1,289 images contain only a few hundred distinct scenes** — that is the
source of the inflation. Note S3 survives at 99%: its frames genuinely differ, so
this project's earlier focus on S3 (the only source with readable frame numbers)
was looking in the wrong place.

Threshold 0.92 was chosen by *what it removes*, not by hitting a target count.
The more aggressive settings approach Kocdemir's 1,871 images, but only by
deleting S2 photographs taken on separate days, which contain nothing to
deduplicate.

**This is not Kocdemir's split.** His list was not available; this is our own
deduplication and the surviving count differs from his.

After deduplication every arm dropped 5.9 to 7.0 mAP and the ranking held. S2,
which lost almost nothing to deduplication, barely moved (24.7 → 24.2) — a
control confirming the drop came from removing duplicates rather than some other
change.

### The effect on comparability

| | detector | mAP |
|---|---|---|
| Kocdemir, best classical (Mantiuk) | RetinaNet, his implementation | 31.3 |
| Kocdemir, Reinhard *(variant unstated)* | RetinaNet, his implementation | 29.6 |
| Kocdemir, his own method (TMO-GAN) | RetinaNet, his implementation | 31.6 |
| **This work, best operator, deduplicated** | **RetinaNet R50-FPN v2** | **31.3** |

Before deduplication this work sat roughly 7 points above his across the board
with no principled explanation. After it, the numbers land in the same range —
evidence the gap was duplicate frames rather than any advantage in method.

Both are RetinaNet but **not the same implementation**: torchvision's v2 recipe
postdates the thesis. That, the different splits, and the unknown class counts
are three separate reasons the absolute values need not line up. This is
convergence, not a head-to-head.

---

## 5. What the dataset audit found

Every one of the 4,080 EXR files was decoded and measured. Full detail in
[REPORT.md](hdr4rtt_analysis/REPORT.md).

- **It is three sources glued together**, separable for free from the EXR
  headers: 1,289 video/rendered frames, 1,489 bracketed photographs, and 1,302
  frames of one continuous HDR video.
- **Neither value ceiling is over-exposure.** About one pixel per image touches
  it — each image was individually normalised, so the stored values carry no
  absolute brightness meaning.
- **41% of images contain negative pixel values**, down to −872. Any log or gamma
  step must clamp first.
- **303 images (7.4%) form a separate very dark cluster**, ~250× darker than the
  rest, holding 13.1% of all boxes.
- **Median dynamic range is 4.11 decades** (≈13.7 stops), maximum 7.83 (26 stops).
- **The shipped train/test split leaks**: 100% of the video source's test frames
  have a training frame within ±3. Measured cost, with a controlled two-model
  experiment: **+2.0 mAP**, about 6% relative.

Per-image dynamic-range statistics for all 4,080 files are in
[`hdr_stats_sources.csv`](hdr4rtt_analysis/hdr_stats_sources.csv), which makes it
possible to stratify detection results by scene difficulty.

**The training set is small.** 2,994 images, but the video source contributes 949
frames from only ~6 contiguous runs — roughly 2,051 genuinely distinct scenes,
about 1.7% the size of a standard detection training set.

---

## 6. Two bugs that invalidated earlier work

**1. Wrong input scale in evaluation.** RAOD's data loader divides pixel values
by 255 inside `load_image`, so the network expects `[0,1]`. The evaluation script
skipped it and fed `[0,255]`. Because RAOD's tone-mapping module applies
`x^(1/gamma)` with gamma of 7–10.5, every pixel saturated and the frame collapsed
to white.

The trap: the same division appears **commented out** in `ValTransformRaw` — the
obvious place to look — because it has already happened one level up.

Proof: RAOD's *own* sample image gave **0 detections**; after restoring one line,
**5**, including a car at 0.892 confidence.

**2. The converter optimised for the opposite of what the model wants.** RAOD
tone-maps internally and expects dark, linear input at mean ≈ 0.012. The previous
converter stretched images to fill `[0,255]`, mean 0.377 — **32× brighter** than
anything the model saw in training. Every rescaling variant tried (p99, adaptive
percentile, log, histogram matching) shares that same "use the full range" goal,
so all of them fail the same way.

---

## 7. Comparison with the MS thesis

### The two review questions

**1. Include the MS thesis results here.** Done — tables below.

Reading them surfaced something relevant to question 2: **the thesis already
contains a detector-swap experiment, and its two detectors disagree.** Its
learned method beats every classical operator under RetinaNet (31.6 vs 31.3) and
loses to four of them under Faster R-CNN (27.7 vs 29.5). Same method, same data,
opposite conclusion.

**2. Compare the 2023 paper and the thesis method fairly.** Partly answered.

| | status |
|---|---|
| Detector held fixed, front end varied | **done** — [section 1](#1-main-result-which-tone-map) |
| RAOD's module measured under that control | **done** — 28.7 mAP |
| Thesis TMO-GAN measured under that control | **not possible** — software lost |
| Indirect comparison via a shared reference | **done** — below |

The two learned methods have still never been run side by side.

### The thesis tables

From İ. H. Kocdemir's MS thesis (and the corresponding Pattern Recognition
Letters 172 (2023) 230–236 paper), on the dataset the thesis calls **OOD** — 20
Pascal VOC classes, near-identical video frames removed, 1,491 train / 380 test,
images at 1024×576.

*Every number below is **quoted from his thesis**, none re-run here. All his rows
are **fine-tuned**, not zero-shot: the detector is trained on each tone-mapped
image set, and ✓ in "joint" marks where his generator trains alongside it.*

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

For context, thesis Table 3.1 reports the same comparison on **CityScapes**,
where HDR with gamma (33.3) barely separates from Std. LDR (33.1) — the thesis
states plainly that no advantage for HDR was observed there.

**Two things these tables show.** *Raw HDR is worst in both* (26.3 and 23.5,
below plain LDR) — normalisation, not bit depth, is what the detector needs,
which is exactly what this work found independently. And *the detector changes
the conclusion*, as above.

### Comparing the two learned methods

TMO-GAN could not be re-run: its software was lost. The obvious workaround does
not work either.

**The failed approach: shared operators as calibration anchors.** Four operators
appear in both experiments, so in principle they could map one scale onto the
other. They do not — the rankings invert:

| operator | his rank (RetinaNet) | rank here |
|---|---|---|
| Reinhard | **worst** tone map, 29.6 *(variant unstated)* | **best**, 31.3 *(global)* |
| Durand *(local in both)* | best of the three, 30.6 | **worst of the three**, 29.7 |
| HDR with gamma | 29.8 | 30.4 |

An inversion, not an offset, so no scale factor reconciles them and the thesis's
31.6 cannot be placed on this axis. Recorded as a negative result — and it
survives deduplication, which changed the values but not the ordering. *(Part of
the Reinhard gap could be global-versus-local rather than real disagreement; see
[section 3](#3-how-to-read-these-numbers).)*

**What does work: each method against the best classical operator in its own
experiment.** That reference is meaningful in both, and cancels the differences
in split, resolution and detector version.

| learned method | source | own score | best classical, same table | **gap** |
|---|---|---|---|---|
| TMO-GAN + RetinaNet *(his impl.)* | his paper, quoted | 31.6 | 31.3 (Mantiuk) | **+0.3** |
| RAOD module + RetinaNet *(R50-FPN v2)* | measured here | 28.7 | 31.3 (Reinhard) | **−2.6** |

The thesis method **slightly beat** the strongest classical operator available to
it; RAOD's module **fell behind** the strongest one available here, by 2.6 mAP.
About 2.9 mAP apart in the thesis method's favour, measured against a shared
reference rather than on a shared scale. Since deduplication brought the absolute
scales into the same range, this is on firmer ground than it was.

**Caveats.** The best classical operator differs between the two (Mantiuk there,
Reinhard here) and Mantiuk is not implemented here; were it stronger than
Reinhard on this data, the gap would widen rather than narrow. Different splits,
different RetinaNet implementations. A defensible indirect comparison, not a
head-to-head.

**What would close it:** re-implement TMO-GAN from the paper — a generator and
discriminator trained jointly with the detector — and run it as a seventh front
end. Then both learned methods sit in one table under one detector.

### Section 2 is not comparable with any of this

| | thesis | section 2 |
|---|---|---|
| classes | 20 | 2 (person, car) |
| test images | 380 | 779 |
| duplicate frames | removed entirely | still present |
| resolution | 1024×576 | 1280×1280 |
| detectors | RetinaNet, Faster R-CNN | RAOD YOLOX, Faster R-CNN v2 |

Placing 51.7 next to 31.6 would be meaningless.
[Section 1](#1-main-result-which-tone-map) is the table to put beside his, since
it uses 20 classes.

---

## 8. Method notes

**Choosing the tone-mapping operator.** The comparison could easily have been
rigged by picking a weak operator. Scoring eight candidates with a
COCO-pretrained detector showed a **10% relative spread**, so a careless choice
would have decided the outcome before any training happened. Gamma at the 99th
percentile won and was used throughout. TMO-Det handles this the same way,
comparing six operators and reporting the best. `pick_tmo.py` reproduces it.

**Use batch size 4, not 8, on a 16 GB GPU.** Batch 8 needs 17.51 GB on a 17.1 GB
card, and Windows does not raise an out-of-memory error — it pages GPU memory to
system RAM, so the run silently becomes ~8× slower with no error message. A
batch-8 run reported an 8-day ETA. Details in
[hdr4rtt_rod/README.md](hdr4rtt_rod/README.md).

---

## 9. Reproducing

Requires PyTorch and OpenCV; RAOD's own repository supplies the model code. Data
paths are set at the top of each script.

Audit the dataset (all 4,080 files):

```bash
python hdr4rtt_analysis/scripts/scan_hdr.py
```

Deduplicate, then build 20-class annotations on what survives:

```bash
python hdr4rtt_rod/build_deduped_split.py
```

Build one image set per tone-mapping front end:

```bash
python hdr4rtt_rod/make_frontends.py
```

Train one arm, then score it (repeat per front end):

```bash
python hdr4rtt_rod/train_frontend.py --arm reinhard --arch retinanet --epochs 10 --batch 4
```

```bash
python hdr4rtt_rod/eval_frontend.py --arm reinhard --arch retinanet --batch 4
```

For the RAOD pipeline of section 2:

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

Checkpoints, converted imagery and the dataset itself are deliberately excluded —
see [.gitignore](.gitignore).

**Open questions, in priority order:**

1. **Re-run section 2 on the deduplicated split**, so both tables sit on the same
   footing. ~6 hours of training.
2. **Match model capacity honestly** — either shrink the tone-mapped arm or grow
   the HDR arm — so the HDR-versus-tone-mapped question is not confounded by a
   44× difference in parameters.
3. **Re-implement TMO-GAN** and run it as a seventh front end, so the two learned
   methods can be compared directly rather than through a shared reference.
4. **Run the local Reinhard variant**, to settle whether the ranking inversion
   against the thesis is real or an artefact of comparing two different operators.

---

## 11. Licence and attribution

Code © 2026 Sana Niroomand and the **OGAM Research Laboratory, Middle East
Technical University (METU)**, released under the [MIT licence](LICENSE).
Research code: provided as is, without warranty, not intended for production or
safety-related use.

Full third-party attribution is in [NOTICE](NOTICE). In summary:

| what | relationship | terms |
|---|---|---|
| **RAOD** (Xu et al., CVPR 2023) | two config files are **modified derivatives**; other files call it at runtime | © Huawei + Megvii, **Apache 2.0** — see [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt) |
| RAOD model weights | not redistributed | original terms |
| **TMO-Det** (Kocdemir et al.) | **no code used** — results quoted only | © authors and publisher |
| Reinhard, Durand operators | algorithms re-implemented from the papers | methods credited to their authors |
| torchvision, OpenCV, pycocotools | called as libraries | BSD / Apache 2.0 |
| HDR4RTT / OOD dataset | **not redistributed**, in any form | obtain from its creators |

The two derivative files — `cfg_hdr4rtt_rod.py` and `cfg_hdr4rtt_rod_original.py`
— remain under Apache 2.0 and carry headers stating exactly what was changed, as
that licence requires. Everything else is original work under MIT.

**The tone-mapping operators here are simplified re-implementations**, tuned
against this dataset. They should not be taken as reference implementations of
the published methods, and any weakness in them is mine rather than the original
authors'.

**Please cite the original work, not this repository, for the methods it builds
on:**

> R. Xu, C. Chen, J. Peng, C. Li, Y. Huang, F. Song, Y. Yan, Z. Xiong.
> *Toward RAW Object Detection: A New Benchmark and a New Model.* CVPR 2023.

> İ. H. Kocdemir, A. Koz, A. O. Akyuz, et al.
> *TMO-Det: Deep tone-mapping optimized with and for object detection.*
> Pattern Recognition Letters 172 (2023) 230–236.

Tables from the thesis and paper are reproduced for academic comparison with
attribution. They are **quoted, not reproduced experimentally** — the original
software was lost, so those numbers could not be re-run, and this is stated
wherever they appear.

Any errors here are mine and not those of the original authors.
