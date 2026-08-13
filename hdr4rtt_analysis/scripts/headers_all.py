"""Extract EXR header metadata (whiteLuminance, FILE_NAME, LUMINANCE) for every file."""
import os, struct, glob, re, collections
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database"
IMG = os.path.join(ROOT, "images")
stats = pd.read_csv(os.path.join(OUT, "hdr_stats_annotated.csv"))


def read_header(path):
    out = {}
    with open(path, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        if magic != 20000630:
            return out
        while True:
            name = b""
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                name += c
            if not name:
                break
            atype = b""
            while True:
                c = f.read(1)
                if c in (b"\x00", b""):
                    break
                atype += c
            (size,) = struct.unpack("<I", f.read(4))
            val = f.read(size)
            n, t = name.decode("ascii", "replace"), atype.decode("ascii", "replace")
            if t == "float":
                out[n] = struct.unpack("<f", val)[0]
            elif t == "string":
                out[n] = val.decode("ascii", "replace")
        return out


rows = []
for p in sorted(glob.glob(os.path.join(IMG, "*.exr"))):
    h = read_header(p)
    rows.append({
        "stem": os.path.splitext(os.path.basename(p))[0],
        "whiteLuminance": h.get("whiteLuminance", np.nan),
        "LUMINANCE": h.get("LUMINANCE", ""),
        "FILE_NAME": h.get("FILE_NAME", ""),
    })
hdr = pd.DataFrame(rows)
df = stats.merge(hdr, on="stem", how="left")
df.to_csv(os.path.join(OUT, "hdr_stats_full.csv"), index=False)


def sec(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


sec("1. METADATA COVERAGE")
has_wl = df["whiteLuminance"].notna()
print(f"images with whiteLuminance attr : {int(has_wl.sum())} / {len(df)} ({has_wl.mean():.1%})")
print(f"images with LUMINANCE=ABSOLUTE  : {int((df['LUMINANCE']=='ABSOLUTE').sum())}")
print(f"images with FILE_NAME attr      : {int((df['FILE_NAME'].fillna('')!='').sum())}")
print("\nmetadata presence by population:")
print(pd.crosstab(df["pop"], has_wl.map({True: "has whiteLuminance", False: "NO metadata"})).to_string())
print("\nmetadata presence by resolution:")
ct = pd.crosstab(df["res"], has_wl.map({True: "has_wL", False: "no_meta"}))
ct["total"] = ct.sum(axis=1)
print(ct.sort_values("total", ascending=False).to_string())

sec("2. whiteLuminance DISTRIBUTION")
wl = df.loc[has_wl, "whiteLuminance"]
print(f"n={len(wl)}   min={wl.min():.4g}  p5={wl.quantile(.05):.4g}  median={wl.median():.4g}  "
      f"p95={wl.quantile(.95):.4g}  max={wl.max():.4g}")
print(f"spread: {np.log10(wl.max()/wl.min()):.2f} decades")
print(f"exactly 1.0 : {int((wl == 1.0).sum())}")
print("\nby population:")
for p, g in df[has_wl].groupby("pop"):
    w = g["whiteLuminance"]
    print(f"   {p:<30} n={len(g):<5} median={w.median():.4g}  min={w.min():.4g}  max={w.max():.4g}")

sec("3. DOES whiteLuminance EXPLAIN THE SCALE DIFFERENCE?")
print("Interpretation: whiteLuminance = cd/m^2 corresponding to pixel value 1.0.")
print("So absolute luminance = pixel_value * whiteLuminance.\n")
sub = df[has_wl].copy()
sub["p99_abs"] = sub["p99.9"] * sub["whiteLuminance"]
sub["p50_abs"] = sub["p50"] * sub["whiteLuminance"]
for col, lab in [("p99.9", "raw p99.9"), ("p99_abs", "p99.9 * whiteLuminance")]:
    v = sub[col].replace(0, np.nan).dropna()
    print(f"   {lab:<26} median={v.median():.4g}  spread={np.log10(v.max()/max(v.min(),1e-12)):.2f} decades  "
          f"IQR ratio={v.quantile(.75)/max(v.quantile(.25),1e-12):.1f}x")
print("\n   -> a SMALLER spread after multiplying means the attribute genuinely")
print("      normalises the two conventions onto one physical scale.")

sec("4. ORIGINAL FILE_NAME  (source-sequence structure)")
fn = df[df["FILE_NAME"].fillna("") != ""].copy()
print(f"images carrying an original filename: {len(fn)}")
print(f"distinct original filenames: {fn['FILE_NAME'].nunique()}")
dup = fn["FILE_NAME"].value_counts()
dupv = dup[dup > 1]
print(f"original filenames used by >1 image: {len(dupv)}  (max reuse {dup.max()})")
print("\nsample of original names:")
print(fn[["stem", "split", "pop", "FILE_NAME"]].head(12).to_string(index=False))

nums = fn["FILE_NAME"].str.extract(r"(\d+)")[0].astype(float)
fn = fn.assign(num=nums)
print(f"\nnumeric part: min={nums.min():.0f}  max={nums.max():.0f}  n_unique={nums.nunique():.0f}")

sec("5. TRAIN/TEST LEAKAGE RISK FROM SEQUENCE STRUCTURE")
print("If these are consecutive video frames, near-duplicate frames split across")
print("train and test would inflate test scores.\n")
f2 = fn.dropna(subset=["num"]).sort_values("num")
print(f"frames with numeric ids: {len(f2)}")
# look for runs of consecutive numbers that straddle the split
f2 = f2.reset_index(drop=True)
straddle = 0
adj_pairs = 0
for i in range(len(f2) - 1):
    if f2.loc[i + 1, "num"] - f2.loc[i, "num"] == 1:
        adj_pairs += 1
        if f2.loc[i, "split"] != f2.loc[i + 1, "split"]:
            straddle += 1
print(f"adjacent-numbered frame pairs (n, n+1): {adj_pairs}")
print(f"   of those, the two frames are in DIFFERENT splits: {straddle} ({straddle/max(adj_pairs,1):.1%})")
print(f"   -> {'HIGH leakage risk: consecutive frames split across train/test' if straddle > adj_pairs*0.1 else 'low'}")

# same for the duplicate original names
if len(dupv):
    print("\noriginal filenames appearing in BOTH splits:")
    g = fn.groupby("FILE_NAME")["split"].nunique()
    both = g[g > 1]
    print(f"   {len(both)} original filenames appear in both train and test")
    if len(both):
        ex = fn[fn["FILE_NAME"].isin(both.index[:4])].sort_values("FILE_NAME")
        print(ex[["stem", "split", "FILE_NAME", "res", "pop"]].head(16).to_string(index=False))
