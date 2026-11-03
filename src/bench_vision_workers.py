import argparse
import csv
import queue
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["serial", "parallel"], required=True)
p.add_argument("--rate", type=float, required=True)
p.add_argument("--requests", type=int, default=120)
p.add_argument("--seed", type=int, default=101)
p.add_argument("--out", required=True)
a = p.parse_args()

assert a.rate > 0 and a.requests > 0

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)

# Match the existing single-tile preprocessing.
img = Image.open("data/test_dog.jpg").convert("RGB")
img = img.resize((448, 448), Image.Resampling.BICUBIC)

x = np.asarray(img, dtype=np.float32) / 255.
mean = np.array([.485, .456, .406], np.float32)
std = np.array([.229, .224, .225], np.float32)
x = ((x - mean) / std).transpose(2, 0, 1)
x = np.ascontiguousarray(x[None], dtype=np.float32)

core = ov.Core()
requests = {}

for device in ("GPU", "NPU"):
    model = core.read_model(
        "models/InternVL2-1B-int4-ov/"
        "openvino_vision_embeddings_model.xml"
    )
    model.reshape([1, 3, 448, 448])

    compiled = core.compile_model(model, device)
    requests[device] = compiled.create_infer_request()

    print(
        device,
        compiled.get_property("EXECUTION_DEVICES"),
        flush=True,
    )

    for _ in range(5):
        requests[device].infer([x])

# A separate request object exists for each device.
# No InferRequest is used by more than one worker simultaneously.

shared = queue.Queue()
queues = {
    "GPU": queue.Queue(),
    "NPU": queue.Queue(),
}

rows = []
errors = []

def worker(q):
    try:
        while True:
            job = q.get()

            if job is None:
                return

            device = job["device"]

            job["start"] = time.perf_counter()
            requests[device].infer([x])
            job["end"] = time.perf_counter()

            job["wait_ms"] = (
                job["start"] - job["arrival"]
            ) * 1000

            job["service_ms"] = (
                job["end"] - job["start"]
            ) * 1000

            job["e2e_ms"] = (
                job["end"] - job["arrival"]
            ) * 1000

            rows.append(job)

    except BaseException as e:
        errors.append(repr(e))

if a.mode == "serial":
    active_queues = [shared]
else:
    active_queues = [queues["GPU"], queues["NPU"]]

threads = [
    threading.Thread(target=worker, args=(q,))
    for q in active_queues
]

for t in threads:
    t.start()

rng = np.random.default_rng(a.seed)
due = time.perf_counter()

for i in range(a.requests):
    if i > 0:
        due += rng.exponential(1.0 / a.rate)

    delay = due - time.perf_counter()
    if delay > 0:
        time.sleep(delay)

    # Deterministic 50/50 placement.
    device = "GPU" if i % 2 == 0 else "NPU"

    job = {
        "id": i,
        "device": device,
        "arrival": time.perf_counter(),
    }

    q = shared if a.mode == "serial" else queues[device]
    q.put(job)

for q in active_queues:
    q.put(None)

for t in threads:
    t.join()

if errors or len(rows) != a.requests:
    raise RuntimeError(
        f"errors={errors}, completed={len(rows)}/{a.requests}"
    )

rows.sort(key=lambda r: r["id"])

fields = [
    "id", "device", "arrival", "start", "end",
    "wait_ms", "service_ms", "e2e_ms",
]

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def stats(name, x):
    x = np.asarray(x, dtype=float)
    print(
        name,
        f"mean={x.mean():.2f}",
        f"p50={np.median(x):.2f}",
        f"p95={np.percentile(x,95):.2f}",
    )

span = max(r["end"] for r in rows) - min(
    r["arrival"] for r in rows
)

stats("End-to-end ms", [r["e2e_ms"] for r in rows])
stats("Queue wait ms", [r["wait_ms"] for r in rows])

for device in ("GPU", "NPU"):
    subset = [r for r in rows if r["device"] == device]
    stats(
        f"{device} service ms",
        [r["service_ms"] for r in subset]
    )

# Total time during which both devices were doing vision work.
gpu = [r for r in rows if r["device"] == "GPU"]
npu = [r for r in rows if r["device"] == "NPU"]

overlap = 0.0
for g in gpu:
    for n in npu:
        overlap += max(
            0.0,
            min(g["end"], n["end"])
            - max(g["start"], n["start"])
        )

print("Mode:", a.mode)
print("Achieved throughput:", round(len(rows)/span, 3), "req/s")
print("GPU/NPU overlap:", round(overlap, 3), "seconds")
print("Results:", out)
