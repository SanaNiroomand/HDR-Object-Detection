"""Follow-ups: (1) is 65504 a real saturation clip? (2) how bad are negatives?
(3) is the dark cluster a distinct subpopulation?"""
import os, json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database"
IMG = os.path.join(ROOT, "images")
df = pd.read_csv(os.path.join(OUT, "hdr_stats_annotated.csv"))


def sec(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


sec("Q1. IS 65504 A SATURATION CLIP OR JUST A NORMALISATION CEILING?")
B = df[df["pop"].str.startswith("B")]["stem"].tolist()
A = df[df["pop"].str.startswith("A")]["stem"].tolist()
rng = np.random.default_rng(0)
sampB = list(rng.choice(B, size=min(120, len(B)), replace=False))
sampA = list(rng.choice(A, size=min(120, len(A)), replace=False))


def ceil_frac(args):
    stem, ceil = args
    im = cv2.imread(os.path.join(IMG, stem + ".exr"),
                    cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    f = im.ravel()
    n = f.size
    at = float((f >= ceil * 0.999999).mean())
    # also: how many pixels within 1% of ceiling
    near = float((f >= ceil * 0.99).mean())
    return stem, at, near


for name, samp, ceil in [("B (65504)", sampB, 65504.0), ("A (255)", sampA, 255.0)]:
    with ThreadPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(ceil_frac, [(s, ceil) for s in samp]))
    at = np.array([r[1] for r in res])
    near = np.array([r[2] for r in res])
    print(f"\npopulation {name}  (n={len(samp)} sampled)")
    print(f"   pixels exactly at ceiling : median={np.median(at):.6%}  mean={at.mean():.6%}  max={at.max():.4%}")
    print(f"   pixels within 1% of ceil  : median={np.median(near):.6%}  mean={near.mean():.6%}  max={near.max():.4%}")
    print(f"   images with >0.1% at ceil : {int((at>0.001).sum())}/{len(at)}")
    print(f"   images with >1%   at ceil : {int((at>0.01).sum())}/{len(at)}")
    verdict = ("REAL SATURATION (many pixels pinned at ceiling)" if np.median(at) > 1e-4
               else "NORMALISATION ceiling (max touches it, bulk does not)")
    print(f"   -> {verdict}")

sec("Q2. NEGATIVE PIXEL VALUES")
neg = df[df["frac_neg"] > 0]
print(f"images containing negative pixels: {len(neg)} / {len(df)} ({len(neg)/len(df):.1%})")
print(f"   by population: {neg['pop'].value_counts().to_dict()}")
print(f"   frac of pixels negative: median={neg['frac_neg'].median():.4%}  "
      f"p95={neg['frac_neg'].quantile(.95):.3%}  max={neg['frac_neg'].max():.3%}")
print(f"   most-negative vmin across dataset: {df['vmin'].min():.6g}")
print(f"   vmin distribution (negatives only): median={neg['vmin'].median():.4g}  "
       f"min={neg['vmin'].min():.4g}")
worst = df.nsmallest(8, "vmin")[["stem", "split", "pop", "vmin", "frac_neg", "p50"]]
print("\n   most negative images:")
print(worst.to_string(index=False))
print("\n   NOTE: negatives break log/gamma tone-mapping unless clamped first.")

sec("Q3. THE DARK / NIGHT SUBPOPULATION")
df["ll"] = np.log10(df["logmean"].replace(0, np.nan))
dark = df[df["ll"] < -1.0]
rest = df[df["ll"] >= -1.0]
print(f"cleanly separated dark cluster (log-mean luminance < 0.1): {len(dark)} images ({len(dark)/len(df):.1%})")
print(f"   gap in histogram between {dark['ll'].max():.2f} and {rest['ll'].min():.2f} "
      f"(log10 units) -> {'well separated' if rest['ll'].min()-dark['ll'].max() > 0.2 else 'contiguous'}")
print(f"\n   dark cluster composition:")
print(f"     population : {dark['pop'].value_counts().to_dict()}")
print(f"     split      : {dark['split'].value_counts().to_dict()}")
print(f"     resolution : {dark['res'].value_counts().head(6).to_dict()}")
print(f"     median p50 : {dark['p50'].median():.5g}   (rest: {rest['p50'].median():.4g})")
print(f"     median p99.9: {dark['p99.9'].median():.5g}   (rest: {rest['p99.9'].median():.4g})")
print(f"     median dynamic range: {dark['dr_decades'].median():.2f} decades  (rest: {rest['dr_decades'].median():.2f})")
print(f"     frac pixels < 0.01: median {dark['frac_lt_0p01'].median():.1%}  (rest: {rest['frac_lt_0p01'].median():.2%})")

# how many objects live in the dark images?
splits = {}
for name, fn in [("train", "instances_train2020.json"), ("test", "instances_test2020.json")]:
    with open(os.path.join(ROOT, "annotations", fn), encoding="utf-8") as f:
        splits[name] = json.load(f)
stem_boxes = {}
for name, d in splits.items():
    idmap = {im["id"]: os.path.splitext(os.path.basename(im["file_name"]))[0] for im in d["images"]}
    for a in d["annotations"]:
        stem_boxes[idmap[a["image_id"]]] = stem_boxes.get(idmap[a["image_id"]], 0) + 1
dark_boxes = sum(stem_boxes.get(s, 0) for s in dark["stem"])
print(f"\n   annotations inside dark images: {dark_boxes} ({dark_boxes/32964:.1%} of all boxes)")

sec("Q4. POPULATION B IS RESOLUTION-LOCKED — CONFIRMING")
print("population B resolutions:", df[df['pop'].str.startswith('B')]['res'].value_counts().to_dict())
r1080 = df[df["res"] == "1920x1080"]
print(f"\n1920x1080 images: {len(r1080)}  -> {r1080['pop'].value_counts().to_dict()}")
print("all other resolutions:", df[df['res'] != '1920x1080']['pop'].value_counts().to_dict())
print("\n=> the dataset is a merge of >=2 sources with different value conventions;")
print("   the 65504-convention source contributes ONLY 1920x1080 frames.")

# filesize signature per population (different encoder settings?)
print("\nfile size (MB) by population:")
for p, g in df.groupby("pop"):
    print(f"   {p:<30} median={g['filesize_mb'].median():.2f}  mean={g['filesize_mb'].mean():.2f}")
print("\nfile size for 1920x1080 only, by population:")
for p, g in r1080.groupby("pop"):
    print(f"   {p:<30} median={g['filesize_mb'].median():.2f} MB   n={len(g)}")
