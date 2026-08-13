# HDR4RTT Dataset Analysis

Source: `D:\Data\HDR\HDR4RTT Database\HDR4RTT Database`
Scope: all 4,080 EXR files decoded and measured (no sampling), plus both COCO JSONs and all 4,080 label files.
Environment: `C:\Users\OGAM\miniconda3\envs\hs-ml\python.exe` (OpenCV 4.12 with `OPENCV_IO_ENABLE_OPENEXR=1`).

---

## 1. Inventory

| | |
|---|---|
| Images | 4,080 OpenEXR, **HALF (fp16)** channels B/G/R, **PIZ** compression, 56 GB |
| Splits | train 3,264 / test 816 (exactly 80/20), **zero overlap**, every image in exactly one split |
| Labels | 4,080 `.txt` + `instances_train2020.json` / `instances_test2020.json` |
| Boxes | 32,964 — `.txt` and COCO totals match exactly, per class |
| Classes | 20 (Pascal VOC set) |
| Read failures | 0 |
| EXR dims vs COCO dims | 4,080 / 4,080 match |

Label format is `classname x1 y1 x2 y2`, absolute pixel **xyxy** — a faithful re-encoding of the COCO `bbox` (x, y, w, h). The `segmentation` fields are the box corners replayed as 4-point polygons, so there are **no real masks**.

Annotation hygiene is good: no degenerate boxes, no `iscrowd`, `area == w*h` for all 32,964, 6 empty label files. Only **47 boxes (0.14%)** cross an image edge (max overflow 1,180 px) — worth clamping, not worth worrying about.

---

## 2. The dataset is a merge of three sources

This is the single most important structural fact, and it is recoverable from the EXR headers (`whiteLuminance`, `LUMINANCE`, `FILE_NAME`). It explains why a single global white point cannot work.

| | S1 | S2 | S3 |
|---|---|---|---|
| Images | 1,289 (31.6%) | 1,489 (36.5%) | 1,302 (31.9%) |
| Header metadata | **none** | `whiteLuminance = 1.0` | `whiteLuminance = 1.15 … 29.98` |
| Original `FILE_NAME` | — | `./batchN_hdr/0006_20160721_190736_527_.hdr` | `000001.exr … 001499.exr` |
| Resolutions | 1920×1080, 1280×720 | 15 distinct, 8–13 MP, incl. portrait | 1920×1080 only |
| Max pixel value | 255 | 255 | **65504** (fp16 ceiling) |
| Median p99.9 | 161 | 181 | **12,800** |
| Median dynamic range | 5.99 dec | 3.38 dec | 4.85 dec |
| Images with negative pixels | 0% | 42.2% | 80.2% |
| Median file size | 4.96 MB | 30.47 MB | 6.34 MB |
| Nature | video / rendered | bracketed HDR stills, 2016 | HDR video sequence |

The 68% / 32% split into "max=255" and "max=65504" populations that earlier work found on ~400 files **holds across all 4,080** (2,776 / 1,302, plus 2 stragglers at 12064 and 17536). But the population boundary is not the useful cut — the *source* boundary is, and S1 and S2 share a max of 255 while being otherwise unalike.

**Both ceilings are normalization ceilings, not saturation.** Sampling 120 images per population, the median fraction of pixels sitting at the ceiling is 0.000016% (S3) and 0.000004% (A) — roughly one pixel per image. Each image was individually scaled so its brightest sample lands on the ceiling. There is no clipped highlight region to recover, and no absolute radiometric meaning to the values.

### `whiteLuminance` does not fix the scale gap

Worth stating explicitly because it looks like it should. `whiteLuminance` is cd/m² at pixel value 1.0, so multiplying by it ought to put everything on one physical scale. It does not:

| | median | spread |
|---|---|---|
| raw p99.9 | 237 | 3.34 decades |
| p99.9 × whiteLuminance | 238 | **4.62 decades** |

It is exactly 1.0 for S1 and S2 and only meaningful for S3, so it scales S3 *further away* (median `vmax × whiteLuminance` = 1.49e6 vs 255). It is metadata about S3's own capture, not a cross-source calibration.

---

## 3. Train/test leakage in S3 — the finding that matters most

S3's `FILE_NAME` values are sequential video frames, `000001.exr` through `001499.exr`. The train/test split was drawn **randomly over individual frames**, not over sequences:

| | |
|---|---|
| Adjacent frame pairs (n, n+1) present | 1,128 |
| …that land in *different* splits | **380 (33.7%)** |
| S3 test frames with a train frame at ±1 | **252 / 277 (91.0%)** |
| S3 test frames with a train frame at ±2 | 276 / 277 (99.6%) |
| S3 test frames with a train frame at ±3 | **277 / 277 (100%)** |

Every single S3 test frame has a near-duplicate in train within three frames. S3 contributes 277 of the 816 test images, so **~34% of the whole test set is contaminated**. Any mAP measured on the full test split is inflated by an unknown but non-trivial margin.

This is consistent with the TMO-Det author deduplicating similar video frames before benchmarking — the dedup was not optional cleanup, it was correcting this.

**Recommendation:** report on a sequence-disjoint split, or at minimum report S1/S2/S3 test scores separately so the contaminated portion is visible. Comparing your numbers against any paper that deduplicated, without doing the same, is not a fair comparison in either direction.

---

## 4. HDR characteristics

**Dynamic range** (log10 of p99.9 / p0.1 over nonzero pixels): median **4.11 decades ≈ 13.7 stops**, p75 5.86, max 7.83 decades (26 stops). Using full `vmax / min_nonzero`, the median is 6.41 decades. This is genuinely wide-range data.

**Scene brightness is bimodal.** A cleanly separated dark cluster of **303 images (7.4%)** sits below log-mean luminance 0.1, with a real gap in the histogram (−1.03 to −0.82 in log10). These are:
- entirely population A / max=255, 300 of 303 at 1920×1080
- median pixel value 0.014 vs 3.63 for the rest — **250× darker**
- 39.6% of their pixels below 0.01 (vs 2.7% elsewhere)
- median dynamic range 5.87 decades vs 3.99
- carrying **4,322 boxes = 13.1% of all annotations**

A single global tone curve tuned on the main mode will crush these to black. They are too large a fraction of the annotations to ignore.

**Negative pixel values in 1,672 images (41%)** — down to −872. Concentrated in S3 (80.2% of its images) and S2 (42.2%); S1 has none. Median affected fraction is small (0.0055%) but the p95 is 14% of pixels. Almost certainly debayer/color-conversion undershoot. **Any log or gamma step must clamp at 0 first**, or it will produce NaNs on 41% of the dataset.

**Sub-unit range:** median image has **24.3% of its pixels below 1.0** (p75 = 48.2%) — i.e. below the 8-bit LSB if you naively treat the 0–255 range as 8-bit. Linear rescale to uint8 throws away a quarter of the signal on a typical image, and far more on the dark cluster.

**Colour:** median R/G = 1.098, B/G = 0.997, but p5–p95 spans 0.67–2.27 for R/G. The data is not consistently white-balanced.

---

## 5. Detection-task characteristics

**Extreme class imbalance, 6,084:1.**

| class | train | test | total | share |
|---|---|---|---|---|
| person | 14,213 | 4,040 | 18,253 | 55.4% |
| bottle | 5,403 | 1,678 | 7,081 | 21.5% |
| car | 2,449 | 697 | 3,146 | 9.5% |
| chair | 1,483 | 341 | 1,824 | 5.5% |
| pottedplant | 878 | 219 | 1,097 | 3.3% |
| diningtable | 635 | 187 | 822 | 2.5% |
| *14 tail classes* | 565 | 176 | 741 | 2.2% |

Top three are 86% of all boxes. The tail is unusable: cat 3, bus 7, train 8 instances total. **`cow` and `cat` have zero test instances**, so their AP is undefined and any 20-class mAP average silently depends on how your eval tool handles that. Consider reporting mAP over the 6 head classes.

**Objects are large.** Median object is 7.2% of the image side; 29% occupy >15%. Only 0.7% qualify as tiny (<1%). `diningtable` (median 36%) and `dog` (54%) are near-frame-filling. person appears in 88.3% of images at 5.07 instances/image.

| scale bucket | boxes | share |
|---|---|---|
| tiny (<1% of side) | 217 | 0.7% |
| small (1–5%) | 9,962 | 30.2% |
| medium (5–15%) | 13,207 | 40.1% |
| large (15–40%) | 6,564 | 19.9% |
| huge (>40%) | 3,014 | 9.1% |

This is a very different scale profile from traffic-oriented RAW benchmarks, where small distant objects dominate. Anchor/scale priors tuned on RAOD are unlikely to transfer without adjustment.

**Resolution varies 14× in area** — 16 distinct sizes, three aspect ratios (16:9, 4:3, 3:4), **145 portrait images**. Any fixed-size pipeline must handle orientation. The split is stratified proportionally, so test carries the same mix.

---

## 6. Practical implications for preprocessing

1. **No single global white point can work.** Per-image p99.9 spans 3.34 decades. Rescaling everything by 255 blows out 98.8% of S3; rescaling by 65504 maps the median S1/S2 image's p99.9 to 0.69/255 — black. This is a structural property of the data, not a tuning problem.

2. **Per-image robust normalization (p99/p99.9) is the only quantity comparable across all three sources** — which is what the earlier p99 experiments were reaching for. The catch is that it discards cross-image brightness consistency, and the 303-image dark cluster is exactly where that hurts: per-image normalization will expose night scenes like day scenes.

3. **A source-aware or two-stage scheme is worth trying**, since source identity is recoverable for free from the EXR header (`whiteLuminance` present/absent and its value). You can branch on it without any clustering heuristic — one branch per source, or a global scale per source plus mild per-image correction.

4. **Clamp negatives before any log/gamma.** 41% of images have them.

5. **Fix the split before trusting any number.** See §3.

---

## Files

| file | contents |
|---|---|
| `hdr_stats_sources.csv` | per-image row for all 4,080: dims, percentiles, dynamic range, clipping, negatives, channel means, population, source group, header metadata |
| `annotation_summary.json` | split sizes, class counts, resolutions |
| `hdr4rtt_analysis.png` | 9-panel visual summary |
| `scripts/` | every script used, re-runnable |

Run order: `scan_hdr.py` → `aggregate_hdr.py` → `headers_all.py` → `sources.py` → `make_figures.py`; `analyze_annotations.py` and `analyze_boxes.py` are independent.
