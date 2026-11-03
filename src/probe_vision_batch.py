import argparse
import csv
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--device", choices=["GPU","NPU"], required=True)
p.add_argument("--batch", type=int, choices=[1,2,4], required=True)
p.add_argument("--repeats", type=int, default=12)
p.add_argument("--out", required=True)
a = p.parse_args()

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)

image = Image.open("data/test_dog.jpg").convert("RGB")
w,h = image.size

regions = [
    image,
    image.crop((0,0,w//2,h//2)),
    image.crop((w//2,0,w,h//2)),
    image.crop((0,h//2,w,h)),
]

def preprocess(im):
    im = im.resize((448,448), Image.Resampling.BICUBIC)
    x = np.asarray(im, np.float32) / 255.0
    mean = np.array([.485,.456,.406],np.float32)
    std = np.array([.229,.224,.225],np.float32)
    return ((x-mean)/std).transpose(2,0,1)

tiles = np.ascontiguousarray(
    np.stack([preprocess(im) for im in regions])[:a.batch],
    dtype=np.float32,
)

core = ov.Core()
path = (
    "models/InternVL2-1B-int4-ov/"
    "openvino_vision_embeddings_model.xml"
)

def compile_batch(n):
    model = core.read_model(path)
    model.reshape([n,3,448,448])

    start = time.perf_counter()
    c = core.compile_model(model,a.device)

    print(
        f"batch={n} compile_s="
        f"{time.perf_counter()-start:.3f}",
        "device=",c.get_property("EXECUTION_DEVICES"),
        flush=True
    )
    return c

# Same model/weights used for both comparisons.
single = compile_batch(1)
batched = single if a.batch == 1 else compile_batch(a.batch)

def run_single():
    outputs = []
    for tile in tiles:
        result = single([tile[None]])
        outputs.append(
            np.array(next(iter(result.values())),copy=True)
        )
    return np.concatenate(outputs,axis=0)

def run_batch():
    result = batched([tiles])
    return np.array(next(iter(result.values())),copy=True)

for _ in range(3):
    run_single()
    run_batch()

reference = run_single()
candidate = run_batch()

print("Single output:", reference.shape)
print("Batch output:", candidate.shape)

assert reference.shape == candidate.shape, (
    "Output shape mismatch: batching may not be compatible"
)

ref = reference.astype(np.float64).ravel()
got = candidate.astype(np.float64).ravel()

cosine = np.dot(ref,got)/(
    np.linalg.norm(ref)*np.linalg.norm(got)
)

rel_l2 = np.linalg.norm(ref-got)/np.linalg.norm(ref)

print("Feature cosine:",cosine)
print("Relative L2:",rel_l2)

rng = np.random.default_rng(20261103)
rows = []

# Alternate order to limit systematic frequency/ramp bias.
for i in range(a.repeats):
    for mode in rng.permutation(["sequential","batch"]):
        t0 = time.perf_counter()

        if mode == "sequential":
            run_single()
        else:
            run_batch()

        dt = (time.perf_counter()-t0)*1000
        rows.append({
            "device":a.device,
            "batch":a.batch,
            "repeat":i,
            "mode":mode,
            "latency_ms":dt
        })

with out.open("w",newline="") as f:
    writer = csv.DictWriter(f,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for mode in ("sequential","batch"):
    x = np.array([
        r["latency_ms"] for r in rows
        if r["mode"] == mode
    ])

    print(
        mode,
        "p50=",round(np.median(x),2),
        "p95=",round(np.percentile(x,95),2),
        "total-frame ms"
    )

print("Saved:",out)
