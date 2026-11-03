import argparse
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--device", default="GPU")
p.add_argument("--rate", type=float, default=5)
p.add_argument("--duration", type=float, default=35)
p.add_argument("--ready", required=True)
p.add_argument("--go", required=True)
a = p.parse_args()

ready = Path(a.ready)
go = Path(a.go)

model_path = (
    "models/InternVL2-1B-int4-ov/"
    "openvino_vision_embeddings_model.xml"
)

img = Image.open("data/test_dog.jpg").convert("RGB")
img = img.resize((448, 448), Image.Resampling.BICUBIC)

x = np.asarray(img, np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)

x = ((x - mean) / std).transpose(2, 0, 1)
x = np.ascontiguousarray(x[None], dtype=np.float32)

core = ov.Core()
model = core.read_model(model_path)
model.reshape([1, 3, 448, 448])

compiled = core.compile_model(model, a.device)

print(
    "Execution devices:",
    compiled.get_property("EXECUTION_DEVICES"),
    flush=True
)

for _ in range(5):
    compiled([x])

ready.touch()
print("READY", flush=True)

while not go.exists():
    time.sleep(0.05)

start = time.monotonic()
end = start + a.duration

count = 0
times = []

print("LOAD_START", time.time(), flush=True)

while True:
    due = start + count / a.rate
    remaining = due - time.monotonic()

    if remaining > 0:
        time.sleep(remaining)

    if time.monotonic() >= end:
        break

    t = time.perf_counter()
    compiled([x])
    times.append((time.perf_counter() - t) * 1000)
    count += 1

print("LOAD_END", time.time(), flush=True)
print("Requests:", count, flush=True)

if times:
    print("Mean vision latency:", np.mean(times))
    print("p50 vision latency:", np.median(times))
    print("p95 vision latency:", np.percentile(times, 95))
