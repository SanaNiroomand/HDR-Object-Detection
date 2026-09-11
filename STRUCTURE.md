# Repository structure

What lives where, and why. For results see the [top-level README](README.md); for
the pipeline's design decisions see [`hdr4rtt_rod/README.md`](hdr4rtt_rod/README.md).

136 tracked files: 33 Python scripts, 63 result files, two project folders.

---

## The shape of it

Two project folders and five files at the root. Nothing else.

```
HDR-Object-Detection/
├── hdr4rtt_analysis/        what is in this dataset?
│   ├── REPORT.md            the written audit, and the thing to read first
│   ├── hdr_stats_sources.csv    one row per EXR file, every measured statistic
│   └── scripts/             9 scripts, run as a chain
├── hdr4rtt_rod/             the experiments
│   ├── results/             63 JSON files: every number in the READMEs
│   ├── viz/                 predictions drawn on converted images
│   ├── progress_note.html   step-by-step record of the work
│   └── ...                  24 scripts, 7 sweep runners, run logs
├── README.md                results and conclusions
├── LICENSE                  MIT, covers original work
├── LICENSE-Apache-2.0.txt   covers the three RAOD-derived config files
├── NOTICE                   full third-party attribution
└── .gitignore               an allow-list, see below
```

`hdr4rtt_analysis/` runs first and is self-contained: it measures the raw EXR
files and produces per-image statistics. `hdr4rtt_rod/` consumes one of those
outputs (`hdr_stats_sources.csv`) to split results by source group, but is
otherwise independent.

---

## What is deliberately absent

This is the single fact that explains the repository's contents. The working
directory it was built in also holds copyrighted paper PDFs, third-party
repositories with their own git history, and tens of gigabytes of converted
imagery. So `.gitignore` ignores **everything** with `/*`, then re-includes the
two project folders and the licence files by name.

| not in the repo | where it lives |
| --- | --- |
| the HDR4RTT dataset | obtained from its creators, not redistributable |
| converted images (`.npy.gz`, `.png`, `.npz`), ~60 GB | `D:\Data\HDR\hdr4rtt_rod\` |
| annotation JSON consumed by training | `D:\Data\HDR\hdr4rtt_rod\annotations\` |
| model checkpoints (~170 MB each) | `frontend_runs/`, `tv_runs/`, `cfg_*/` |
| RAOD's model code and pretrained weights | RAOD's own repository |

The repository holds the recipe. The ingredients are reproduced by running it.

---

## Reading a filename

These recur in filenames, command-line flags and result keys throughout
`hdr4rtt_rod/`.

| term | meaning |
| --- | --- |
| **arm** | which tone map sits in front of the detector: `reinhard`, `durand`, `gamma`, `log`, `hdr` (no curve), `tmm` (RAOD's learned module) |
| **arch** | which detector: `retinanet` or `fasterrcnn` |
| **front end** | the tone-mapping step itself, i.e. whatever converts HDR to something the detector reads |
| **dedup** | run on the near-duplicate-free split. A name *without* it is from the earlier, inflated full split |
| **S1 / S2 / S3** | the three source groups inside HDR4RTT, separated by EXR header metadata: video/rendered frames, bracketed photographs, and one continuous HDR video |
| **seqsafe** / **original** | leakage-free split vs HDR4RTT's shipped (leaky) split |
| **voc20** / **rod** | 20 Pascal VOC classes vs the 2 classes (person, car) that map onto RAOD's fixed 5-output head |

Result files follow `{arch}_{arm}[_dedup][_S#].json`, so
`retinanet_reinhard_dedup_eval_S2.json` is RetinaNet, Reinhard tone map,
deduplicated split, scored on source group S2 alone.

---

## `hdr4rtt_analysis/` — the dataset audit

A chain of scripts, each reading the CSV the previous one wrote and adding
columns. Run them in this order; skipping one breaks the next.

```
   4,080 .exr files
        │
        ▼
   scan_hdr.py          per-image pixel statistics        → hdr_stats.csv
        │
        ▼
   aggregate_hdr.py     dataset-level findings            → hdr_stats_annotated.csv
        │
        ▼
   headers_all.py       EXR header metadata               → hdr_stats_full.csv
        │
        ▼
   sources.py           assign S1/S2/S3, test leakage     → hdr_stats_sources.csv  ← committed
        │
        ▼
   make_figures.py      summary plots                     → hdr4rtt_analysis.png   ← committed
```

Four scripts sit off the chain and can be run whenever:

| file | what it does |
| --- | --- |
| `scripts/analyze_annotations.py` | structure of HDR4RTT's own COCO and YOLO annotation files → `annotation_summary.json` |
| `scripts/analyze_boxes.py` | box geometry and image resolution per split |
| `scripts/check_precision.py` | reads EXR headers to determine HALF vs FLOAT storage and compression |
| `scripts/followup.py` | three specific questions: is the 65504 ceiling real saturation, how bad are negative pixels, is the dark cluster genuinely separate |

**Committed outputs** sit in `hdr4rtt_analysis/`, one level above the scripts
that write them: `REPORT.md`, `hdr_stats_sources.csv` (2 MB, one row per EXR
file), `annotation_summary.json`, and `hdr4rtt_analysis.png`. The three
intermediate CSVs are regenerated, not committed.

---

## `hdr4rtt_rod/` — the pipeline

24 Python files, grouped by role rather than by filename.

### 1. Converters — EXR in, trainable images out

| file | output |
| --- | --- |
| `convert_hdr4rtt_to_rod.py` | `.npy.gz` for RAOD. No tone mapping: RAOD tone-maps internally and expects dark linear input, so `--gain 0.02` matches its operating point |
| `convert_hdr4rtt_to_ldr.py` | ordinary 8-bit sRGB PNG, for the torchvision detectors |
| `make_frontends.py` | one image set **per tone-mapping arm**, all from the same EXR source and identical in every other way. This is what makes the comparison controlled |
| `make_frontend_native.py` | one arm regenerated at native aspect ratio and resolution, instead of being squashed to 1280×1280 |

All converters share the same geometry deliberately (negatives clamped,
BGR→RGB, squashed to 1280×1280 without preserving aspect, gray-world white
balance), so a single set of annotations describes every version and the only
difference between arms is the tone curve.

### 2. Annotation and split builders

| file | produces |
| --- | --- |
| `build_rod_annotations.py` | 2-class COCO JSON in RAOD's exact format, plus the leakage-free `seqsafe` split |
| `build_voc20_annotations.py` | all 20 Pascal VOC classes, for the fixed-detector experiment |
| `build_deduped_split.py` | the 20-class annotations restricted to images surviving near-duplicate removal |
| `build_rod_dedup_annotations.py` | the same for the 2-class RAOD-format set |
| `make_source_subsets.py` | splits each test annotation file by source group, so S3 can be scored against S3 |

### 3. Training

| file | trains |
| --- | --- |
| `train_frontend.py` | one torchvision detector against one chosen arm. The main experiment script |
| `train_torchvision.py` | the plain tone-mapped baseline row of the detector comparison |
| `smoke_test_train.py` | trains nothing — checks the dataloader builds, weights load, and one forward/backward step completes, before committing hours to a real run |

### 4. Evaluation

| file | scores |
| --- | --- |
| `eval_frontend.py` | a `train_frontend.py` checkpoint; mirrors that script's data handling exactly so an arm is scored on the representation it trained on |
| `eval_rod.py` | RAOD, with a `--gains` sweep in a single load |
| `eval_torchvision.py` | a torchvision detector, either COCO-pretrained zero-shot or from a checkpoint |
| `run_single.py` | one image in detail, calling RAOD's own preprocessing functions rather than reimplementing them |

### 5. Diagnostics

| file | answers |
| --- | --- |
| `analyze_failures.py` | where the detector fails, broken down TIDE-style per predicted box, joined to statistics from the original EXR |
| `analyze_failures_controlled.py` | whether brightness and dynamic range still matter once object size is held fixed |
| `anchor_fit.py` | what share of ground-truth boxes any anchor can reach at IoU 0.5 — a ceiling, measurable in seconds instead of a GPU-hour |
| `pick_tmo.py` | which tone map to use for the baseline, scored rather than chosen by eye, so the comparison isn't rigged before training |
| `visualize.py` | draws ground truth and predictions to verify the coordinate contract holds |

### 6. RAOD configs

`cfg_hdr4rtt_rod.py` is the base config, a **modified derivative of RAOD's
`cfg_small.py`** and therefore Apache 2.0, with a modification notice in its
header. `cfg_hdr4rtt_rod_dedup.py` and `cfg_hdr4rtt_rod_original.py` subclass it
and change only the annotation filenames and experiment name — deduplicated
split and HDR4RTT's original leaky split respectively.

These three files are the only Apache-licensed code here. Everything else is
MIT. See [NOTICE](NOTICE).

### 7. Sweep runners

Thin wrappers that chain the scripts above. They carry the measured settings
(batch 4, not 8) and continue past a failed arm so one crash doesn't cost the
whole sweep.

| file | runs |
| --- | --- |
| `run_all_frontends.ps1` | every arm against the same detector |
| `run_dedup_frontends.ps1` | the same six arms on the deduplicated split |
| `run_headline_dedup.ps1` | the 2-class detector-comparison table, deduplicated |
| `run_tmm_variants.ps1` | three isolated retests of the learned tone-mapping arm |
| `run_anchors.sh` | the retuned-anchor-ratio runs |
| `run_capacity.sh`, `run_capacity2.sh` | RAOD's tone map behind RetinaNet and Faster R-CNN |

---

## Where the numbers live

Every figure in the READMEs traces to a file in `hdr4rtt_rod/results/`.

| path | files | contents |
| --- | --- | --- |
| `results/frontends_dedup/` | 24 | the six tone maps on the deduplicated split, with per-source `_S1/_S2/_S3` breakdowns |
| `results/frontends/` | 9 | the same sweep on the earlier, inflated full split |
| `results/geometry/` | 6 | resolution and anchor experiments, plus `anchor_fit.txt` |
| `results/headline_dedup/` | 5 | the 2-class detector comparison |
| `results/failures/` | 4 | per-box failure CSV, the worst-image list, and the two worst images as JPEG |
| `results/capacity/` | 3 | RAOD's tone map behind RetinaNet and Faster R-CNN |
| `results/*.json` | 18 | baselines, leakage tests, the deduplication report, sweeps |

Result JSON is uniform: `arm`, `arch`, checkpoint path, image and box counts,
then `mAP`, `AP50`, `AP75`, size-bucketed AP, `AR100`, and `per_class_AP50`.

Alongside these, `viz/` holds ground truth and predictions drawn on converted
images plus `tone_mapping_comparison.jpg`, `progress_note.html` records the work
in order, and the `*.log` files are raw console output from the sweeps, kept so
numbers can be traced back.

**Scale note.** `pycocotools` prints mAP between 0 and 1, and that is what the
JSON stores. The top-level README multiplies by 100, the convention the papers
use. So `0.3129` in a result file is `31.3` in a table.

---

## Typical run order

```bash
# 1. audit the dataset
python hdr4rtt_analysis/scripts/scan_hdr.py

# 2. build the deduplicated split and its annotations
python hdr4rtt_rod/build_deduped_split.py

# 3. generate one image set per tone-mapping arm
python hdr4rtt_rod/make_frontends.py

# 4. train, one arm at a time
python hdr4rtt_rod/train_frontend.py --arm reinhard --arch retinanet \
    --epochs 10 --batch 4 --min_size 1280 --max_size 2133

# 5. evaluate; settings are read back from the checkpoint
python hdr4rtt_rod/eval_frontend.py --arm reinhard --arch retinanet --batch 4
```

Data paths are constants at the top of each script and point at `D:\Data\HDR\`.
Change them there before running elsewhere.

Two traps worth repeating, both silent failures rather than errors:

- **Pass `--min_size 1280`.** Left at the default, torchvision shrinks every
  1280×1280 image to 800×800 before the backbone sees it, and says nothing. The
  value is stored in the checkpoint so evaluation cannot disagree with training.
- **Use batch 4 on a 16 GB card.** Batch 8 needs 17.51 GB on a 17.1 GB card, and
  Windows pages GPU memory to system RAM rather than raising an out-of-memory
  error. The run just becomes roughly 8× slower with nothing in the log to
  explain it.

---

## Licensing

Original work is MIT, © 2026 Sana Niroomand and the OGAM Research Laboratory,
Middle East Technical University (METU). The three `cfg_hdr4rtt_rod*.py` files
are modified derivatives of RAOD's `cfg_small.py` and carry Apache 2.0, each
with a modification notice in its header. Full attribution in [NOTICE](NOTICE).
