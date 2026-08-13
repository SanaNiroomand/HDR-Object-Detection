"""Figures summarising the HDR4RTT dataset analysis."""
import os, json, collections
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database"
df = pd.read_csv(os.path.join(OUT, "hdr_stats_sources.csv"))
df["FILE_NAME"] = df["FILE_NAME"].fillna("")

splits = {}
for name, fn in [("train", "instances_train2020.json"), ("test", "instances_test2020.json")]:
    with open(os.path.join(ROOT, "annotations", fn), encoding="utf-8") as f:
        splits[name] = json.load(f)
cats = {c["id"]: c["name"] for c in splits["train"]["categories"]}

SC = {"S1: no metadata": "#4C72B0", "S2: whiteLuminance = 1.0": "#DD8452",
      "S3: whiteLuminance != 1.0": "#55A868"}
SHORT = {"S1: no metadata": "S1 video/synth", "S2: whiteLuminance = 1.0": "S2 bracketed photo",
         "S3: whiteLuminance != 1.0": "S3 HDR video seq"}

plt.rcParams.update({"figure.dpi": 130, "font.size": 8, "axes.grid": True,
                     "grid.alpha": .25, "axes.axisbelow": True})
fig, ax = plt.subplots(3, 3, figsize=(15.5, 11.5))

# 1 class distribution
inst = collections.Counter()
for d in splits.values():
    for a in d["annotations"]:
        inst[cats[a["category_id"]]] += 1
items = inst.most_common()
a = ax[0, 0]
a.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#4C72B0")
a.set_xscale("log"); a.set_xlabel("instances (log scale)")
a.set_title("1. Class distribution — 32,964 boxes, 6084:1 imbalance")
a.tick_params(labelsize=6)

# 2 resolution by source
a = ax[0, 1]
rc = df.groupby(["res", "source"]).size().unstack(fill_value=0)
rc["t"] = rc.sum(axis=1); rc = rc.sort_values("t").drop(columns="t").tail(10)
bot = np.zeros(len(rc))
for s in rc.columns:
    a.barh(rc.index, rc[s], left=bot, color=SC.get(s), label=SHORT.get(s, s))
    bot += rc[s].values
a.set_xlabel("images"); a.set_title("2. Resolution (16 distinct) by source")
a.legend(fontsize=6); a.tick_params(labelsize=6)

# 3 vmax populations
a = ax[0, 2]
lv = np.log10(df["vmax"].replace(0, np.nan).dropna())
a.hist(lv, bins=80, color="#937860")
a.set_xlabel("log10(max pixel value)"); a.set_ylabel("images")
a.set_title("3. Two value conventions: max=255 vs max=65504")
for x, lab in [(np.log10(255), "255\n(68%)"), (np.log10(65504), "65504\n(32%)")]:
    a.axvline(x, color="crimson", ls="--", lw=1)
    a.text(x, a.get_ylim()[1] * .72, lab, color="crimson", ha="center", fontsize=6.5)

# 4 dynamic range by source
a = ax[1, 0]
for s, g in df.groupby("source"):
    d = g["dr_decades"].dropna()
    a.hist(d, bins=45, alpha=.62, color=SC.get(s), label=f"{SHORT.get(s,s)} (med {d.median():.1f})")
a.set_xlabel("dynamic range, decades  log10(p99.9 / p0.1)"); a.set_ylabel("images")
a.set_title("4. Per-image dynamic range (median 4.11 dec = 13.7 stops)")
a.legend(fontsize=6)

# 5 scene brightness -> the dark cluster
a = ax[1, 1]
ll = np.log10(df["logmean"].replace(0, np.nan))
a.hist(ll.dropna(), bins=70, color="#8172B3")
a.axvspan(ll.min(), -1.0, color="crimson", alpha=.13)
a.text(-1.6, a.get_ylim()[1] * .78, "dark cluster\n303 imgs (7.4%)\n13.1% of boxes",
       color="crimson", ha="center", fontsize=6.5)
a.set_xlabel("log10(log-mean luminance)"); a.set_ylabel("images")
a.set_title("5. Scene brightness — separated night population")

# 6 relative object size
recs_rel, recs_cls = [], []
for name, d in splits.items():
    dims = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    for an in d["annotations"]:
        W, H = dims[an["image_id"]]
        _, _, bw, bh = an["bbox"]
        recs_rel.append(np.sqrt(bw * bh / (W * H)))
        recs_cls.append(cats[an["category_id"]])
rel = np.array(recs_rel)
a = ax[1, 2]
a.hist(rel * 100, bins=80, color="#C44E52")
a.set_xlabel("relative object size, % of image side"); a.set_ylabel("boxes")
a.set_title("6. Object scale — median %.1f%%, only 0.7%% tiny" % (np.median(rel) * 100))

# 7 leakage: S3 frame index vs split
a = ax[2, 0]
s3 = df[df["source"] == "S3: whiteLuminance != 1.0"].copy()
s3["num"] = s3["FILE_NAME"].str.extract(r"(\d+)")[0].astype(float)
s3 = s3.sort_values("num")
w = s3[s3["num"] <= 200]
a.scatter(w.loc[w["split"] == "train", "num"], np.ones((w["split"] == "train").sum()),
          s=7, color="#4C72B0", label="train")
a.scatter(w.loc[w["split"] == "test", "num"], np.zeros((w["split"] == "test").sum()),
          s=7, color="#C44E52", label="test")
a.set_ylim(-.6, 1.6); a.set_yticks([0, 1]); a.set_yticklabels(["test", "train"])
a.set_xlabel("original video frame number (first 200 of S3)")
a.set_title("7. LEAKAGE: consecutive frames split randomly")
a.legend(fontsize=6, loc="center right")

# 8 relative size by class
a = ax[2, 1]
byc = collections.defaultdict(list)
for r, c in zip(recs_rel, recs_cls):
    byc[c].append(r * 100)
top = sorted(byc, key=lambda c: -len(byc[c]))[:8]
a.boxplot([byc[c] for c in top], tick_labels=top, showfliers=False)
a.set_yscale("log"); a.set_ylabel("relative size %"); a.set_title("8. Object scale by class")
a.tick_params(axis="x", rotation=45, labelsize=6)

# 9 p99.9 white point spread by source
a = ax[2, 2]
for s, g in df.groupby("source"):
    a.hist(np.log10(g["p99.9"].replace(0, np.nan).dropna()), bins=45, alpha=.62,
           color=SC.get(s), label=SHORT.get(s, s))
a.set_xlabel("log10(p99.9) — per-image white point"); a.set_ylabel("images")
a.set_title("9. White point spans 3.34 decades across sources")
a.legend(fontsize=6)

plt.tight_layout()
p = os.path.join(OUT, "hdr4rtt_analysis.png")
plt.savefig(p, bbox_inches="tight")
print("saved", p, os.path.getsize(p) / 1e6, "MB")
