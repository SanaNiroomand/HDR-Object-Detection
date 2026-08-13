"""Decompose HDR4RTT into its source groups and test leakage within each."""
import os, re
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(OUT, "hdr_stats_full.csv"))
df["FILE_NAME"] = df["FILE_NAME"].fillna("")


def sec(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


# ---- define source groups
def src(r):
    if r["FILE_NAME"] == "":
        return "S1: no metadata"
    if pd.isna(r["whiteLuminance"]) or r["whiteLuminance"] == 1.0:
        return "S2: whiteLuminance = 1.0"
    return "S3: whiteLuminance != 1.0"


df["source"] = df.apply(src, axis=1)

sec("1. THREE SOURCE GROUPS")
for s, g in df.groupby("source"):
    res = g["res"].value_counts()
    print(f"\n{s}   n={len(g)} ({len(g)/len(df):.1%})")
    print(f"   populations : {g['pop'].value_counts().to_dict()}")
    print(f"   resolutions : {dict(list(res.items())[:6])}{' ...' if len(res)>6 else ''}  ({len(res)} distinct)")
    print(f"   split       : {g['split'].value_counts().to_dict()}")
    print(f"   whiteLum    : {'n/a' if g['whiteLuminance'].isna().all() else f'{g.whiteLuminance.min():.4g} .. {g.whiteLuminance.max():.4g}'}")
    print(f"   median p50={g['p50'].median():.4g}  p99.9={g['p99.9'].median():.4g}  vmax={g['vmax'].median():.6g}")
    print(f"   median dynamic range: {g['dr_decades'].median():.2f} decades")
    print(f"   median filesize: {g['filesize_mb'].median():.2f} MB")
    print(f"   images w/ negatives: {int((g['frac_neg']>0).sum())} ({(g['frac_neg']>0).mean():.1%})")

sec("2. FILE_NAME PATTERNS PER SOURCE")
for s, g in df[df["FILE_NAME"] != ""].groupby("source"):
    names = g["FILE_NAME"]
    pats = names.str.replace(r"\d", "#", regex=True).value_counts()
    print(f"\n{s}: {len(g)} names, {names.nunique()} unique")
    print(f"   patterns: {dict(list(pats.items())[:5])}")
    nums = names.str.extract(r"(\d+)")[0].astype(float)
    print(f"   numeric range: {nums.min():.0f} .. {nums.max():.0f}   unique: {nums.nunique():.0f}")
    print(f"   sample: {list(names.head(6))}")

sec("3. LEAKAGE TEST — WITHIN EACH SOURCE SEPARATELY")
print("Consecutive video frames are near-duplicates. If frame n is in train and")
print("frame n+1 is in test, the test set is contaminated.\n")
for s, g in df[df["FILE_NAME"] != ""].groupby("source"):
    g = g.copy()
    g["num"] = g["FILE_NAME"].str.extract(r"(\d+)")[0].astype(float)
    g = g.dropna(subset=["num"]).sort_values("num").reset_index(drop=True)
    if g["num"].duplicated().any():
        print(f"{s}: numeric ids not unique within source, skipping")
        continue
    n = g["num"].values
    sp = g["split"].values
    adj = n[1:] - n[:-1] == 1
    diff = sp[1:] != sp[:-1]
    both = adj & diff
    print(f"{s}   (n={len(g)}, id range {n.min():.0f}-{n.max():.0f})")
    print(f"   adjacent pairs (n, n+1)     : {int(adj.sum())}")
    print(f"   adjacent AND cross-split    : {int(both.sum())}  ({both.sum()/max(adj.sum(),1):.1%} of adjacent pairs)")
    # every test frame: does it have a train neighbour within +-1 / +-2?
    trainset = set(g.loc[g["split"] == "train", "num"])
    testnums = g.loc[g["split"] == "test", "num"]
    for w in (1, 2, 3):
        near = sum(1 for t in testnums if any((t + d) in trainset for d in range(-w, w + 1) if d != 0))
        print(f"   test frames with a train frame within +-{w}: {near}/{len(testnums)} ({near/max(len(testnums),1):.1%})")

sec("4. WHAT whiteLuminance ACTUALLY DOES")
print("Claim to test: multiplying by whiteLuminance puts all sources on one")
print("physical (cd/m^2) scale, collapsing the 255-vs-65504 difference.\n")
sub = df[df["whiteLuminance"].notna()].copy()
sub["p999_abs"] = sub["p99.9"] * sub["whiteLuminance"]
sub["vmax_abs"] = sub["vmax"] * sub["whiteLuminance"]
for col, lab in [("p99.9", "raw p99.9"), ("p999_abs", "p99.9 x whiteLuminance"),
                 ("vmax", "raw vmax"), ("vmax_abs", "vmax x whiteLuminance")]:
    v = sub[col].replace(0, np.nan).dropna()
    print(f"   {lab:<26} median={v.median():>10.4g}  spread={np.log10(v.max()/max(v.min(),1e-12)):>5.2f} dec  "
          f"IQR={v.quantile(.25):.4g}..{v.quantile(.75):.4g}")
print("\n   per source, median vmax x whiteLuminance:")
for s, g in sub.groupby("source"):
    print(f"     {s:<28} {(g['vmax']*g['whiteLuminance']).median():.6g}")
print("\n   VERDICT: whiteLuminance is 1.0 for every source except S3, so it cannot")
print("   reconcile S1/S2 with S3 -- it makes the S3 values 1.15-30x LARGER, widening")
print("   the gap. It is metadata about S3's own capture, not a cross-source calibration.")

sec("5. RECOMMENDED NORMALISATION TARGET")
print("Per-image robust white point (p99.9) is the only quantity that is comparable")
print("across all three sources. Current spread of p99.9:\n")
for s, g in df.groupby("source"):
    v = g["p99.9"]
    print(f"   {s:<28} median={v.median():>9.4g}  p5={v.quantile(.05):>9.4g}  p95={v.quantile(.95):>9.4g}")
print(f"\n   whole dataset p99.9 spread: {np.log10(df['p99.9'].max()/df['p99.9'].min()):.2f} decades")
print("   after per-image p99.9 normalisation, every image maps 0..1 by construction.")

df.to_csv(os.path.join(OUT, "hdr_stats_sources.csv"), index=False)
print("\n[saved hdr_stats_sources.csv]")
