# TMO-GAN as a seventh front end: feasibility check

Date: 2026-10-25. Scope: reading the code and the thesis, plus a short GPU test
(about 15 steps per setting). Nothing was trained, and nothing in the repo was changed
except this file.

## Short answer

| question | answer |
|---|---|
| 1. Is `reference_code/tmo_detector_github/` the TMO-GAN / TMO-Det code? | **Yes.** It is the thesis-era version (last commit 22 Dec 2021). It matches Chapter 4 of the thesis, with three small differences. The 2023 paper's final training recipe is **not** in it. |
| 2. Are trained generator weights anywhere under `D:\Codes\HDR\Sana\`? | **No.** The three `h3dr…/checkpoints/*/best.pth` files are Cityscapes Faster R-CNN detectors, as suspected. |
| 3. Can it run here? | **Not as published.** It needs mmdetection 2.x, mmcv 1.x and a data loader that is missing from the repo. **The generator on its own works**: I tested it in front of our torchvision RetinaNet without mmcv or mmdetection. |
| 4. Cost of a `tmogan` arm | **About 45 min** of GPU time for the cheapest version. **About 2 to 3.5 h** for the full thesis recipe. It fits on the 16 GB card **only if the generator runs at 800×800**, not 1280×1280. |

Because there are no weights, any number we get is **his architecture and losses,
trained by us**. It is not his trained model.

## Words used here

| word | meaning |
|---|---|
| GAN | Two networks trained against each other. The **generator** makes the tone-mapped image. The **discriminator** tries to tell it apart from a "real" tone-mapped image. |
| target image | The "real" tone-mapped image the discriminator compares against. In the thesis it is, for each picture, the classical operator with the best TMQI score. |
| TMQI | Tone-Mapping Quality Index: a score for how good a tone-mapped image looks compared with its HDR original. |
| joint training | The generator and the detector learn at the same time, and the detector's error also pushes the generator. |
| weights / checkpoint | The learned numbers of a trained network, saved as a `.pth` file. |
| mmdetection, mmcv | The detection library Kocdemir used. mmcv contains "compiled ops": small GPU programs that must be built for each PyTorch, CUDA and GPU version. |
| stub | A tiny fake module that only satisfies an `import` line. |
| batch, step, epoch | Batch = images per update (4 here). Step = one update. Epoch = one pass over the 2,500 training images, i.e. 625 steps. |
| fp16 | Half-precision numbers. They use about half the memory, but very small or very large values can overflow and turn into NaN ("not a number"). |
| paging | When a job needs more GPU memory than the card has, Windows silently moves memory to system RAM. Nothing crashes, but the run becomes very slow. |
| mAP | Detection accuracy, 0 to 100, higher is better (COCO-style, averaged over 10 overlap thresholds). |

---

## 1. Is this the thesis method? Yes

The code is a fork of the public pix2pix framework (its `LICENSE` is Jun-Yan Zhu and
Taesung Park's BSD licence), extended with a detector. One training step does
exactly what thesis Eq. 4.1 (p. 30) describes:

| thesis says | code does | where |
|---|---|---|
| generator G, discriminator D and detector F are trained together | builds all three; the detector comes from an mmdetection config | `models/pix2pix_model.py:113-120`, `:130-135` |
| order: update D, update F, update G | `forward` → `backward_D` → `backward_Det` → `backward_G` | `pix2pix_model.py:619-631` |
| G's loss = GAN loss + gradient-profile loss + feature-matching loss + detection loss | the four terms are summed into `loss_G` | `pix2pix_model.py:497-519` (GAN), `:524-549` (feature matching), `:551-572` (gradient profile), `:574-604` (detection) |
| λ_det scales the gradient flowing from F back to G, not F's own update (p. 31) | `MulConstant`: does nothing going forward, multiplies the gradient going back | `models/networks.py:1065-1113`, used at `pix2pix_model.py:303-326` |
| least-squares GAN (thesis Eq. 4.3 and 4.4) | `gan_mode='lsgan'` → MSE loss | `pix2pix_model.py:44`, `networks.py:301-302` |
| U-Net generator, bilinear upsampling, leaky ReLU, instance norm (p. 32) | `UnetGeneratorBilinear` | `networks.py:496-691` |
| attention on every skip connection, queries taken from the original image (Eq. 4.5) | `EfficientAttention(…, in_channels=3)`, called with `(feature, input)` | `networks.py:514-521`, `:1001-1040` |
| hard-tanh output and "skip-image" (0.8 × input added to the output) (p. 37) | `--tanh_G_out --hardtanh --skip 0.8` | `networks.py:678-689` |
| 70×70 PatchGAN discriminator (p. 32) | `NLayerDiscriminator`, 3 layers | `networks.py:936-994` |
| RetinaNet R50-FPN and Faster R-CNN R50-FPN, 20 classes | two configs, `num_classes=20` | `config/retinanet_DeepTMO-HDR.py:3,25`; `config/faster-rcnn_DeepTMO-HDR.py:3,47` |
| "OOD version": pretrained GAN + pretrained detector, fine-tuned together for 20 epochs, also trained on real images (p. 34, 36) | loads a pretrained GAN, `--train_on_real`, 20 epochs, Adam | `scripts/TRAIN_retinanet_pretrainedGAN.slurm:20-58` |
| dataset: 20 VOC classes, resized to 1024×576, batch 8, targets = best-TMQI picture (p. 35) | `data_root='/HDD/H3DR/HDR4RTT/0_RESIZED/'`, `Resize 1024×576`, `samples_per_gpu=8`, `img_prefix='images_TMQI/'` | `config/hdr_aligned_tmqi.py:3,18,56,65` |

That last row also confirms that the thesis's "OOD dataset" **is HDR4RTT**, the same
data we use.

### Where code and text disagree

| item | thesis / paper | code |
|---|---|---|
| kernel sizes in the "multiple kernel block" | 3, 5, 7, 9 (thesis p. 32, paper §3.1.1) | 1, 3, 5, 7 (`networks.py:548-559`) |
| learning rate, OOD fine-tune | 0.00001 (thesis §4.2.3.3) | 0.00002 (`TRAIN_retinanet_pretrainedGAN.slurm:49-51`) |
| detector epochs, COCO version | 5 + 25 + 5 = 35 (thesis §4.2.3.3) | 55, with the decay at epoch 50 (`TRAIN_retinanet_COCO.slurm:39,45`) |
| 2023 paper recipe: SGD at 0.0001, 2-epoch warm-up, 15 + 15 epochs, cosine schedule (paper §4.3.3, p. 233) | — | **no script uses it.** The repo ends in Dec 2021, before the paper |

So the repo reproduces the **thesis** tables (TMO-GAN + RetinaNet OOD, 31.6) better
than the **paper's** 32.1 ± 0.09.

### Missing pieces

- **No weights** (section 2).
- **No data loader.** The configs ask for `type='HDRDataset'`, a
  `LoadImageFromFile(aligned_hdr=True, …)` option and an `img_HDR` key
  (`config/hdr_aligned_tmqi.py:2,12,23`). None of these exist in the repo or in stock
  mmdetection: a search for `HDRDataset|aligned_hdr|img_HDR` only finds his own
  configs and `pix2pix_model.py`. They must have lived in his private copy of
  mmdetection.
- The `mmdetection/` folder inside the clone is **not his**. It is an untracked,
  separate clone of upstream mmdetection v2.22.0 from Feb 2022
  (`mmdetection/mmdet/version.py:3`), added locally on 2026-06-06. The `help.txt`
  files are also local notes, not part of his repo (`git status` lists both as
  untracked).

---

## 2. Trained weights: none

I searched every `.pth .pt .ckpt .pkl .tar .h5 .onnx .safetensors .bin` file under
`D:\Codes\HDR\Sana\`, and also listed the inside of the 26 GB archive.

| place | what is there | generator? |
|---|---|---|
| `RAOD/RAOD/pre-trained/` | RAOD's 3 YOLOX checkpoints (7.8 MB each) | no |
| `hdr4rtt_rod/cfg_hdr4rtt_rod*/`, `frontend_runs/`, `tv_runs*/` | our own runs | no |
| `previous_intern_ulas/AODRaw-mmdetection/work_dirs/` + the cascade-rcnn `.pth` | Ulaş's AODRaw detectors | no |
| `previous_intern_ulas/Backups/` | only `trilinear_cpp_backup/`: source code and Linux builds of a small CUDA extension; 30 files, 73 MB, no `.pth` | no |
| `previous_intern_ulas/Oğuz_Hoca_mail/*.zip` | debug PNGs and preprocessing scripts | no |
| `previous_intern_ulas/OGAM_without_datasets_and_matlab.7z` (26 GB) | listed with Windows' built-in `tar -tvf`: 17,580 entries, no extraction needed. Its weight files are the **same** ones already unpacked on disk; `tmo_detector_github/` inside has no `checkpoints/` | no |
| `reference_code/tmo_detector_github/` | no weight files at all | no |
| `reference_code/h3dr_object_detection_cityscapes/…/checkpoints/{8bit,16bit,Opt}/best.pth` | see below | no |

**The three `best.pth` files**, opened with `torch.load(weights_only=True)`:
mmdetection 2.4.0, mmcv 1.1.4, epoch 96. Their classes are the 8 Cityscapes classes
(person, rider, car, truck, bus, train, motorcycle, bicycle). The layer groups are
`backbone / neck / rpn_head / roi_head`, with `fc_cls` of shape 9 × 1024 (8 classes
+ background). So they are **Faster R-CNN detectors with no generator inside**,
confirmed.

**Why there are none:** the repo's own `.gitignore` excludes `checkpoints/`
(line 3). `run_inference.sh:4-5` expects `OOD-init_G.pth` and `OOD-init_Det.pth`
from `/home/ismail/git_repos/checkpoints/`, his own machine. They were never
published. The only way to get them would be to ask the author.

---

## 3. Can it run on this machine?

### 3a. The original way: no

| blocker | evidence |
|---|---|
| mmcv and mmdetection are not installed in `hs-ml` (Python 3.11, torch 2.8.0+cu129, RTX 5070 Ti) | import test: `ModuleNotFoundError` for both |
| The code needs **mmcv 1.x**. It uses `mmcv.parallel`, `mmcv.runner` and `mmcv.Config`, which mmcv 2.x removed | `networks.py:14-15`, `pix2pix_model.py:14-15`, `train.py:27`, `data/__init__.py:17` |
| mmcv 1.x would have to be compiled for torch 2.8 / CUDA 12.9 on Windows. Per the brief, that build already failed on this machine | not retried |
| Even if it built, training would not start: `HDRDataset` and the paired HDR + LDR loader are missing | section 1 |

### 3b. The generator alone: yes, tested

`networks.py` imports mmcv and mmdetection only to wrap networks
(`MMDataParallel`) and to build mmdetection detectors (`define_Det`, lines 178-189).
The generator, discriminator and losses themselves are plain PyTorch. Ten lines of
stubs make the **unmodified** file import (the stubs are in the probe script listed at
the end).

The test wrapper works the same way as `TmmWrapper` (`train_frontend.py:106-141`):
the `hdr` arm's [0,1] linear image is mapped to [-1,1], passed through the
generator, mapped back to [0,1], and handed to `build_detector("retinanet", 21)`.
I ran about 15 real steps at **batch 4** on real dedup training images (1280×1280),
with the thesis flags (attention, skip 0.8, hard-tanh). The generator has
**3.84 M parameters**; the discriminator has 0.70 M.

| setting | GPU memory (card: 17.1 GB) | seconds per step |
|---|---|---|
| RetinaNet alone, for reference | 3.6 GB | 0.11 |
| generator + RetinaNet, generator at **1280** | **19.4 GB, over the limit** | **19.6** (paging) |
| generator + RetinaNet, generator at **800** | 9.0 GB | 0.39 |
| generator + discriminator + RetinaNet, 800, fp16 (full TMO-Det step) | 11.7 GB | 0.36 |
| same, fp32 | **20.6 GB, over the limit** | 22.2 (paging) |
| generator + discriminator only (GAN pretraining, no detector), 800, fp16 | 7.5 GB | 0.23 |

For comparison, the existing arms in `dedup_sweep.log`: classical arms peak at
4.0 GB, `tmm` at 12.0 GB.

Three things this test showed:

1. **1280 does not fit.** The generator keeps 32 to 64 feature channels at full
   resolution. At 1280 it goes over the card, Windows pages, and each step takes
   ~20 s instead of ~0.4 s (34 h for 10 epochs). This is the same silent trap that
   once hit RAOD. **Run the generator at 800×800.** That costs the detector nothing:
   `train_frontend.py` already shrinks every input to 800×800 before RetinaNet sees
   it (`min_size=800`, lines 161-164). The wrapper must resize the generator's output
   back to 1280, because the boxes are in 1280 coordinates. My probe skipped that
   step; it does not change the memory figures.
2. **fp16 needs a second gradient scaler.** My first full-GAN run turned NaN on step
   2, because I updated the discriminator without its own `GradScaler`. With one,
   15 steps ran cleanly. fp32 does not fit, so fp16 is required.
   `train_frontend.py:276` has only one scaler today.
3. The probe's numbers are "reserved" memory, which is a little higher than the
   "allocated" figure `train_frontend.py` prints, so they err on the safe side.

---

## 4. Plan for a `tmogan` arm

### Two versions

| version | what it is | thesis equivalent | GPU time |
|---|---|---|---|
| **A. generator only** | His generator, randomly initialised, in front of RetinaNet, trained only by the detection loss. Built exactly like the `tmm` arm. | Experiment 5, "TMO-GAN without the discriminator" (p. 47) | **~45 min** |
| **B. full TMO-Det** | Stage 1: pretrain the GAN on (HDR, target) image pairs. Stage 3: train jointly with RetinaNet, keeping the GAN losses. Optional stage 2: a detector trained on frozen generator output, which gives his "TMO-GAN" (not joint) row. | "TMO-GAN + RetinaNet, OOD version" | **~2 h** (20 GAN epochs) to **~3.5 h** (50, as in the thesis) |

Version A has one weakness: with random weights and skip 0.8, the generator starts
out producing roughly 0.8 × input + 0.1. That is close to the plain `hdr` arm
(24.4 mAP), not to a tone-mapped image, so it has more to learn than `tmm` did.
Version B starts from a generator that has already learned to tone-map.

**Recommendation:** run A first. It is cheap, and it is directly comparable with
`tmm` (28.7), because both are "a learned module trained only by the detector". Then
run B if the advisors want the thesis method itself.

### Files

| file | change |
|---|---|
| `hdr4rtt_rod/train_frontend.py` | add `"tmogan"` to `--arm` choices (line 190), `FLOAT_ARMS` and `ARM_DIR = {"tmogan": "hdr"}` (lines 48-49). Add a `TmoGanWrapper` next to `TmmWrapper`: stub-import `networks.py`, build `define_G(3, 3, 32, "unet_bilinear", opt, "instance", …)` with the slurm flags, resize 1280 → 800, map [0,1] → [-1,1], run G, map back to [0,1], resize to 1280, then the detector. Add a second optimiser group with `--tmogan_lr_mult` (as `tmm` does, lines 264-268). Add `--tmogan_init PATH` (load a stage-1 generator). Add `--tmogan_gan` (version B: add the discriminator, its own Adam optimiser at 2e-5 with β1 = 0.5, a **second GradScaler**, and the GAN, feature-matching ×10 and gradient-profile ×0.8 losses). Save these settings in the checkpoint (lines 305-312). |
| `FrontendDataset.__getitem__` (same file, lines 84-99) | for `--tmogan_gan` only: also load the target PNG with the same stem, flipped with the **same** coin flip as the HDR image |
| `hdr4rtt_rod/eval_frontend.py` | add `"tmogan"` to choices (line 33), `FLOAT_ARMS` and `ARM_DIR` (lines 26-27), and build `TmoGanWrapper` when `arm == "tmogan"` (next to line 73). The discriminator is not needed at test time. |
| new `hdr4rtt_rod/pretrain_tmogan.py` | version B, stage 1: generator + discriminator on (hdr, target) pairs, no detector. Adam at 2e-4, β1 = 0.5, linear decay (thesis §4.2.3.2). Saves `G.pth`. |
| new `hdr4rtt_rod/run_tmogan.ps1` | a copy of `run_dedup_frontends.ps1` for the one arm: `--tag dedup`, dedup annotation files, then evaluate the whole test set plus S1, S2 and S3 |
| later, after a result | README rows (`README.md:331`, `:372`, `:542`), and the third-party table at `README.md:567` and NOTICE: TMO-Det code would then be used, not just quoted |

Import the author's file via stubs or copy it? I suggest the **stub import**. The
`tmm` arm already imports RAOD's file from an untracked folder, and both `RAOD/`
and `reference_code/` are outside git (`git check-ignore` matches `.gitignore:7`).
That way the code under test is byte-for-byte his. Copying the ~300 needed lines
into `hdr4rtt_rod/`, with the BSD notice, would make the repo self-contained; that
is the alternative.

### Design choices that keep it like-for-like

| choice | value | why |
|---|---|---|
| input data | the `hdr` arm's float files (`frontends/hdr`) | the same input `tmm` uses. Note that it is already scaled by the 99.9th percentile and clipped (`make_frontends.py:78-79`), unlike his fixed global scaling (`pix2pix_model.py:287-293`). |
| detector | torchvision RetinaNet R50-FPN v2, COCO start, 10 epochs, batch 4, SGD + OneCycle, flip only | the same as all six other arms. His recipe instead starts the detector from stage 2 and adds random crops; that would give this arm more training than the others got. |
| generator size | 800×800 | the only size that fits (section 3b) |
| target image (version B) | **Reinhard PNGs** first. Best-TMQI per picture as a follow-up. | Reinhard is free: `frontends/reinhard/` is already pixel-aligned with `frontends/hdr/`. Best-TMQI matches the thesis, but `util/tmqi.py` does not import here: `scipy.signal.gaussian` was removed after scipy 1.12 (installed: 1.16.3), and the `contracts` package is missing. It is a two-line fix, but the CPU time over 2,500 × 4 images is unmeasured. |
| scoring | dedup test split: 633 images, 5,684 boxes; whole set plus S1, S2, S3 | the same as the other arms |

### GPU time estimate (batch 4, generator at 800, fp16)

Step times come from the probe, plus ~0.05 s per step for data loading. That
overhead is calibrated on the Reinhard arm: 0.11 s in the probe against 998 s ÷
6,250 steps = 0.16 s in the real run.

| run | memory | time |
|---|---|---|
| A: generator + RetinaNet, 10 epochs (6,250 steps × ~0.43 s) | ~9 GB | **~45 min** |
| B1: GAN pretraining, 20 epochs (12,500 × ~0.28 s); 50 epochs as in the thesis | ~7.5 GB | **~1 h**; ~2.4 h for 50 |
| B2 (optional): save generator output as PNGs, then train like a classical arm | ~4 GB | a few minutes (estimate) + ~17 min |
| B3: joint generator + discriminator + RetinaNet, 10 epochs (6,250 × ~0.41 s) | ~11.7 GB | **~43 min** |
| each evaluation (whole set + S1, S2, S3) | small | a few minutes |

For scale: a classical arm takes ~17 min (998 s) and `tmm` took ~60 min (3,584 s).

### What to compare against (dedup split, RetinaNet)

| arm | mAP |
|---|---|
| Reinhard (best) | **31.3** |
| `tmm` (RAOD's learned module) | 28.7 |
| `hdr` (no curve) | 24.4 |

His published numbers (31.6 in the thesis, 32.1 in the paper) come from his own
split (1,491 train / 380 test) and his own detector. The useful question is
therefore whether `tmogan` beats **our** Reinhard, not whether it reproduces 31.6.

### Risks

- GAN training can be unstable, and the thesis's settings were tuned for his data
  normalisation, not ours. Watch the first epoch for NaN or a collapsing loss.
- Every arm so far is a single run, so the run-to-run noise is unknown. A gap
  under ~1 mAP should not be called a win or a loss.
- With no weights, a poor result could mean "the method is weaker" or "we trained
  it differently". The README should say this plainly.

---

## How the evidence was gathered

- Probe scripts (outside the repo):
  `C:\Users\OGAM\AppData\Local\Temp\claude\D--Codes-HDR-Sana\2cdf62ba-8128-44df-8f2d-aec9f6360b83\scratchpad\probe_tmogan.py`
  and
  `C:\Users\OGAM\AppData\Local\Temp\claude\D--Codes-HDR-Sana\2cdf62ba-8128-44df-8f2d-aec9f6360b83\scratchpad\probe_gd.py`.
  They write no checkpoints.
- Listing of the 26 GB archive:
  `C:\Users\OGAM\AppData\Local\Temp\claude\D--Codes-HDR-Sana\2cdf62ba-8128-44df-8f2d-aec9f6360b83\scratchpad\ogam_7z_list.txt`
- Thesis page numbers refer to `papers/TMO-Det_thesis.pdf`; paper sections refer
  to `papers/TMO-Det_paper.pdf` (Pattern Recognition Letters 172, 2023, 230-236).
