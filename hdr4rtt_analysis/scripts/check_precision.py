"""Determine EXR pixel storage type (HALF vs FLOAT) and compression, from the file header."""
import os, struct, collections
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Data\HDR\HDR4RTT Database\HDR4RTT Database"
IMG = os.path.join(ROOT, "images")
df = pd.read_csv(os.path.join(OUT, "hdr_stats_annotated.csv"))

PIXTYPE = {0: "UINT", 1: "HALF", 2: "FLOAT"}
COMP = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ", 5: "PXR24", 6: "B44", 7: "B44A",
        8: "DWAA", 9: "DWAB"}


def read_header(path):
    """Minimal OpenEXR header parser: magic, version, then name/type/size/value attrs."""
    out = {}
    with open(path, "rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        if magic != 20000630:
            return {"error": "not EXR"}
        out["version"] = ver & 0xFF
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
            if n == "channels":
                chans, off = [], 0
                while off < len(val) - 1:
                    e = val.index(b"\x00", off)
                    cn = val[off:e].decode("ascii", "replace")
                    if not cn:
                        break
                    pt = struct.unpack("<i", val[e + 1:e + 5])[0]
                    chans.append((cn, PIXTYPE.get(pt, pt)))
                    off = e + 1 + 16
                out["channels"] = chans
            elif n == "compression":
                out["compression"] = COMP.get(val[0], val[0])
            elif n in ("dataWindow", "displayWindow"):
                out[n] = struct.unpack("<iiii", val)
            elif t == "float":
                out[n] = round(struct.unpack("<f", val)[0], 6)
            elif t == "string":
                out[n] = val.decode("ascii", "replace")
        return out


print("=" * 78)
print("EXR HEADER INSPECTION")
print("=" * 78)
rng = np.random.default_rng(1)
samples = {}
for pop, g in df.groupby("pop"):
    n = min(60, len(g))
    samples[pop] = list(rng.choice(g["stem"].values, size=n, replace=False))

for pop, stems in samples.items():
    ptypes, comps, extra = collections.Counter(), collections.Counter(), collections.Counter()
    for s in stems:
        h = read_header(os.path.join(IMG, s + ".exr"))
        if "channels" in h:
            ptypes[tuple(sorted(c[0] + ":" + str(c[1]) for c in h["channels"]))] += 1
        comps[h.get("compression", "?")] += 1
        for k in h:
            if k not in ("channels", "compression", "dataWindow", "displayWindow", "version"):
                extra[k] += 1
    print(f"\n{pop}   (n={len(stems)} sampled)")
    for k, v in ptypes.most_common():
        print(f"   channels/type : {k}  x{v}")
    print(f"   compression   : {dict(comps)}")
    print(f"   other attrs   : {dict(extra)}")

print("\n" + "=" * 78)
print("FULL HEADER OF ONE FILE PER POPULATION")
print("=" * 78)
for pop, stems in samples.items():
    h = read_header(os.path.join(IMG, stems[0] + ".exr"))
    print(f"\n--- {stems[0]}.exr   [{pop}] ---")
    for k, v in h.items():
        print(f"   {k:<18}{v}")

print("\n" + "=" * 78)
print("ARE VALUES EXACTLY REPRESENTABLE IN FLOAT16?")
print("=" * 78)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
for pop, stems in samples.items():
    s = stems[0]
    im = cv2.imread(os.path.join(IMG, s + ".exr"),
                    cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    rt = im.astype(np.float16).astype(np.float32)
    same = float((rt == im).mean())
    print(f"   {pop:<30} {s}: {same:.4%} of values survive a float32->16->32 round trip")
    uniq = len(np.unique(im.ravel()[:2_000_000]))
    print(f"   {'':<30} distinct values in first 2M samples: {uniq}")
