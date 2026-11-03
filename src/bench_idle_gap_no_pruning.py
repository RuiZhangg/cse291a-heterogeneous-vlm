import argparse
import csv
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--device", choices=["GPU", "NPU"], required=True)
p.add_argument("--repeats", type=int, default=12)
p.add_argument("--out", required=True)
a = p.parse_args()

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)

img = Image.open("data/test_dog.jpg").convert("RGB")
img = img.resize((448, 448), Image.Resampling.BICUBIC)

x = np.asarray(img, dtype=np.float32) / 255.
mean = np.array([.485, .456, .406], dtype=np.float32)
std = np.array([.229, .224, .225], dtype=np.float32)

x = ((x - mean) / std).transpose(2, 0, 1)
x = np.ascontiguousarray(x[None], dtype=np.float32)

core = ov.Core()
model = core.read_model(
    "models/InternVL2-1B-int4-ov/"
    "openvino_vision_embeddings_model.xml"
)
model.reshape([1, 3, 448, 448])

config = (
    {"NPU_DISABLE_IDLE_MEMORY_PRUNING": True}
    if a.device == "NPU" else {}
)
compiled = core.compile_model(model, a.device, config)
req = compiled.create_infer_request()

print("Device:", compiled.get_property("EXECUTION_DEVICES"),
      flush=True)

def infer():
    req.infer([x])

# Initial warmup
for _ in range(5):
    infer()

gaps = [0, .05, .10, .15, .20, .30, .50, 1.0]
rng = np.random.default_rng(20261103)
rows = []

for rep in range(a.repeats):
    for gap in rng.permutation(gaps):

        # Bring the device back to a recently active state.
        for _ in range(3):
            infer()

        last_end = time.perf_counter()
        if gap > 0:
            time.sleep(float(gap))

        start = time.perf_counter()
        infer()
        end = time.perf_counter()

        rows.append({
            "device": a.device,
            "repeat": rep,
            "target_gap_ms": round(float(gap) * 1000, 2),
            "actual_gap_ms": (start - last_end) * 1000,
            "latency_ms": (end - start) * 1000,
        })

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nIdle gap -> inference latency")
for gap in gaps:
    x = np.array([
        r["latency_ms"] for r in rows
        if r["target_gap_ms"] == round(gap * 1000, 2)
    ])

    print(
        f"{gap*1000:6.0f} ms:",
        f"p50={np.median(x):7.2f}",
        f"p95={np.percentile(x,95):7.2f}",
        flush=True
    )

print("Saved:", out)
