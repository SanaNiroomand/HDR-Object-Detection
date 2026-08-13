"""Aggregate per-image HDR stats into dataset-level findings."""
import os
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, "hdr_stats.csv"))


def sec(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


sec("0. SCAN INTEGRITY")
print(f"rows: {len(df)}")
bad = df[df["dtype"].astype(str).str.contains("ERR|FAIL", na=False)]
print(f"read failures: {len(bad)}")
if len(bad):
    print(bad[["stem", "dtype"]].head(20).to_string(index=False))
df = df[~df["dtype"].astype(str).str.contains("ERR|FAIL", na=False)].copy()
print(f"dtypes: {df['dtype'].value_counts().to_dict()}")
print(f"channels: {df['channels'].value_counts().to_dict()}")
print(f"non-finite pixels anywhere: {int((df['frac_nonfinite'] > 0).sum())} images")
print(f"negative pixels anywhere  : {int((df['frac_neg'] > 0).sum())} images")
print(f"exact-zero pixels present : {int((df['frac_zero'] > 0).sum())} images")

sec("1. EXR DIMENSIONS vs COCO DECLARED DIMENSIONS")
mm = df[df["dim_match"] == 0]
print(f"mismatched: {len(mm)} / {len(df)}")
if len(mm):
    print(mm[["stem", "split", "w", "h", "coco_w", "coco_h"]].head(25).to_string(index=False))
    # is it a pure transpose (rotation) issue?
    tr = mm[(mm["w"] == mm["coco_h"]) & (mm["h"] == mm["coco_w"])]
    print(f"\nof those, pure width/height SWAP (rotation mismatch): {len(tr)}")
    print(f"other mismatches: {len(mm) - len(tr)}")

sec("2. VALUE-RANGE POPULATIONS  (the 'two conventions' issue)")
vmax = df["vmax"].values
print("distinct vmax values (top 15):")
vc = df["vmax"].round(2).value_counts().head(15)
for v, n in vc.items():
    print(f"   {v:>12}   {n:>5} images   ({n/len(df):>6.1%})")


def pop(v):
    if abs(v - 255.0) < 0.51:
        return "A: max=255"
    if abs(v - 65504.0) < 1.0:
        return "B: max=65504 (fp16 ceiling)"
    return "C: other"


df["pop"] = df["vmax"].map(pop)
print("\npopulation breakdown:")
for p, g in df.groupby("pop"):
    print(f"   {p:<30}{len(g):>5} images  ({len(g)/len(df):>6.1%})")

print("\npopulation x split:")
print(pd.crosstab(df["pop"], df["split"]).to_string())

print("\npopulation x resolution:")
df["res"] = df["w"].astype(str) + "x" + df["h"].astype(str)
ct = pd.crosstab(df["res"], df["pop"])
ct["total"] = ct.sum(axis=1)
print(ct.sort_values("total", ascending=False).to_string())

if (df["pop"] == "C: other").any():
    o = df[df["pop"] == "C: other"]
    print(f"\n'other' population vmax range: {o['vmax'].min():.4g} .. {o['vmax'].max():.4g}")
    print(o[["stem", "split", "vmax", "p99.9", "mean"]].sort_values("vmax").head(15).to_string(index=False))

sec("3. CLIPPING / SATURATION (per population)")
for p, g in df.groupby("pop"):
    ceil = 255.0 if p.startswith("A") else (65504.0 if p.startswith("B") else np.nan)
    print(f"\n{p}   (n={len(g)})")
    if p.startswith("A"):
        f = g["frac_at_ceil"]
        print(f"   pixels exactly at 255 : median={f.median():.4%}  mean={f.mean():.4%}  max={f.max():.2%}")
        print(f"   images with >1% clipped : {int((f>0.01).sum())} ({(f>0.01).mean():.1%})")
        print(f"   images with >5% clipped : {int((f>0.05).sum())} ({(f>0.05).mean():.1%})")
    print(f"   p99.9 : median={g['p99.9'].median():.4g}   p99: median={g['p99'].median():.4g}")
    print(f"   median pixel (p50) : median={g['p50'].median():.4g}")
    print(f"   mean : median={g['mean'].median():.4g}")

sec("4. DYNAMIC RANGE")
dr = df["dr_decades"].dropna()
print(f"log10(p99.9 / p0.1 of nonzero)  — 'decades' of dynamic range")
print(f"   min={dr.min():.2f}  p25={dr.quantile(.25):.2f}  median={dr.median():.2f}  "
      f"p75={dr.quantile(.75):.2f}  max={dr.max():.2f}")
print(f"   ~stops (log2): median={dr.median()*3.32:.1f}  max={dr.max()*3.32:.1f}")
print("\n   by population:")
for p, g in df.groupby("pop"):
    d = g["dr_decades"].dropna()
    if len(d):
        print(f"     {p:<30} median={d.median():.2f} decades ({d.median()*3.32:.1f} stops)   n={len(d)}")

print("\n   full-range ratio vmax/min_nonzero:")
rr = np.log10(df["vmax"] / df["min_nonzero"].replace(0, np.nan))
rr = rr.replace([np.inf, -np.inf], np.nan).dropna()
print(f"     median={rr.median():.2f} decades   p95={rr.quantile(.95):.2f}   max={rr.max():.2f}")

sec("5. SCENE BRIGHTNESS / DAY-NIGHT STRUCTURE")
lm = df["logmean"].replace(0, np.nan).dropna()
print(f"log-mean luminance: min={lm.min():.4g}  p5={lm.quantile(.05):.4g}  median={lm.median():.4g}  "
      f"p95={lm.quantile(.95):.4g}  max={lm.max():.4g}")
ll = np.log10(lm)
print(f"log10(log-mean): min={ll.min():.2f}  median={ll.median():.2f}  max={ll.max():.2f}  spread={ll.max()-ll.min():.2f} decades")
hist, edges = np.histogram(ll, bins=14)
print("\n   histogram of log10(log-mean luminance):")
for i in range(len(hist)):
    bar = "#" * int(hist[i] / max(hist) * 46)
    print(f"     [{edges[i]:>6.2f},{edges[i+1]:>6.2f})  {hist[i]:>5}  {bar}")

# crude day/night split using p50
print("\n   dark-scene indicators:")
for thr in [0.1, 0.5, 1.0, 5.0]:
    k = int((df["p50"] < thr).sum())
    print(f"     images with median pixel < {thr:<5}: {k:>5} ({k/len(df):>6.1%})")
print(f"\n   fraction of pixels below 1.0 (i.e. below 8-bit LSB if 0-255 scaled):")
print(f"     median over images={df['frac_lt_1'].median():.1%}   p25={df['frac_lt_1'].quantile(.25):.1%}   p75={df['frac_lt_1'].quantile(.75):.1%}")
print(f"   fraction below 0.01: median={df['frac_lt_0p01'].median():.2%}")

sec("6. CHANNEL BALANCE (cv2 reads BGR)")
for p, g in df.groupby("pop"):
    print(f"{p}: mean_R={g['mean_R'].median():.4g}  mean_G={g['mean_G'].median():.4g}  mean_B={g['mean_B'].median():.4g}")
rg = (df["mean_R"] / df["mean_G"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
bg = (df["mean_B"] / df["mean_G"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
print(f"R/G ratio: median={rg.median():.3f}  p5={rg.quantile(.05):.3f}  p95={rg.quantile(.95):.3f}")
print(f"B/G ratio: median={bg.median():.3f}  p5={bg.quantile(.05):.3f}  p95={bg.quantile(.95):.3f}")
print("(ratios far from 1.0 => strong colour cast / un-white-balanced data)")

sec("7. PRACTICAL IMPLICATION: single global white point")
a = df[df["pop"].str.startswith("A")]
b = df[df["pop"].str.startswith("B")]
print(f"If you rescale everything by 255:")
if len(b):
    print(f"   population B ({len(b)} imgs) median p99.9 = {b['p99.9'].median():.4g}  -> "
          f"{(b['p99.9'] > 255).mean():.1%} of B images have p99.9 above 255 (blown out)")
print(f"If you rescale everything by 65504:")
if len(a):
    print(f"   population A ({len(a)} imgs) median p99.9 = {a['p99.9'].median():.4g}  -> maps to "
          f"{a['p99.9'].median()/65504*255:.4f}/255 (crushed to black)")
print(f"\nper-image p99.9 spread across whole dataset: "
      f"min={df['p99.9'].min():.4g}  median={df['p99.9'].median():.4g}  max={df['p99.9'].max():.4g}  "
      f"({np.log10(df['p99.9'].max()/max(df['p99.9'].min(),1e-9)):.1f} decades)")

df.to_csv(os.path.join(OUT, "hdr_stats_annotated.csv"), index=False)
print("\n[saved hdr_stats_annotated.csv]")
