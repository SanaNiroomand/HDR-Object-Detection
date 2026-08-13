"""Full per-image HDR statistics scan over all 4080 EXR files. Writes hdr_stats.csv."""
import os, csv, json, time, glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database"
IMG = os.path.join(ROOT, "images")
OUT = os.path.dirname(os.path.abspath(__file__))

# split membership from COCO
split_of, dims_of = {}, {}
for name, fn in [("train", "instances_train2020.json"), ("test", "instances_test2020.json")]:
    with open(os.path.join(ROOT, "annotations", fn), encoding="utf-8") as f:
        d = json.load(f)
    for im in d["images"]:
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        split_of[stem] = name
        dims_of[stem] = (im["width"], im["height"])

files = sorted(glob.glob(os.path.join(IMG, "*.exr")))
PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 99.99]
HDR_CEIL = 255.0

FIELDS = (["stem", "split", "w", "h", "coco_w", "coco_h", "dim_match", "channels", "dtype",
           "filesize_mb", "vmin", "vmax", "mean", "logmean"]
          + [f"p{p}" for p in PCTS]
          + ["frac_zero", "frac_neg", "frac_nonfinite", "frac_at_ceil", "frac_ge_254",
             "frac_lt_1", "frac_lt_0p01", "dr_decades", "mean_R", "mean_G", "mean_B",
             "min_nonzero"])


def one(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if im is None:
            return {"stem": stem, "split": split_of.get(stem, "?"), "dtype": "READ_FAIL"}
        h, w = im.shape[:2]
        ch = 1 if im.ndim == 2 else im.shape[2]
        cw, chh = dims_of.get(stem, (0, 0))
        f = im.astype(np.float32, copy=False).ravel()
        n = f.size
        finite = np.isfinite(f)
        nf = int((~finite).sum())
        g = f[finite]
        # luminance proxy for day/night: use per-pixel mean over channels
        if ch == 3:
            lum = im[..., 0].astype(np.float32) * 0.0722 + im[..., 1].astype(np.float32) * 0.7152 \
                  + im[..., 2].astype(np.float32) * 0.2126  # cv2 gives BGR
            lum = lum[np.isfinite(lum)]
        else:
            lum = g
        pos = lum[lum > 0]
        logmean = float(np.exp(np.log(pos + 1e-8).mean())) if pos.size else 0.0
        pv = np.percentile(g, PCTS) if g.size else np.zeros(len(PCTS))
        nz = g[g > 0]
        mnz = float(nz.min()) if nz.size else 0.0
        p999 = float(np.percentile(g, 99.9)) if g.size else 0.0
        p01 = float(np.percentile(nz, 0.1)) if nz.size else 0.0
        dr = float(np.log10(p999 / p01)) if p01 > 0 and p999 > 0 else float("nan")
        r = {
            "stem": stem, "split": split_of.get(stem, "?"), "w": w, "h": h,
            "coco_w": cw, "coco_h": chh, "dim_match": int(w == chh and h == cw) if False else int(w == cw and h == chh),
            "channels": ch, "dtype": str(im.dtype),
            "filesize_mb": round(os.path.getsize(path) / 1e6, 2),
            "vmin": float(g.min()) if g.size else 0.0, "vmax": float(g.max()) if g.size else 0.0,
            "mean": float(g.mean()) if g.size else 0.0, "logmean": logmean,
            "frac_zero": float((g == 0).mean()) if g.size else 0.0,
            "frac_neg": float((g < 0).mean()) if g.size else 0.0,
            "frac_nonfinite": nf / n,
            "frac_at_ceil": float((g >= HDR_CEIL).mean()) if g.size else 0.0,
            "frac_ge_254": float((g >= 254.0).mean()) if g.size else 0.0,
            "frac_lt_1": float((g < 1.0).mean()) if g.size else 0.0,
            "frac_lt_0p01": float((g < 0.01).mean()) if g.size else 0.0,
            "dr_decades": dr, "min_nonzero": mnz,
        }
        for p, v in zip(PCTS, pv):
            r[f"p{p}"] = float(v)
        if ch == 3:
            r["mean_B"], r["mean_G"], r["mean_R"] = [float(im[..., i][np.isfinite(im[..., i])].mean()) for i in range(3)]
        else:
            r["mean_B"] = r["mean_G"] = r["mean_R"] = r["mean"]
        return r
    except Exception as e:
        return {"stem": stem, "split": split_of.get(stem, "?"), "dtype": f"ERR:{type(e).__name__}"}


t0 = time.time()
done = 0
with open(os.path.join(OUT, "hdr_stats.csv"), "w", newline="", encoding="utf-8") as fh:
    wtr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    wtr.writeheader()
    with ThreadPoolExecutor(max_workers=12) as ex:
        for r in ex.map(one, files):
            wtr.writerow(r)
            done += 1
            if done % 250 == 0:
                el = time.time() - t0
                print(f"{done}/{len(files)}  {el:.0f}s  eta {el/done*(len(files)-done):.0f}s", flush=True)
print(f"DONE {done} files in {time.time()-t0:.0f}s -> hdr_stats.csv", flush=True)
