import csv
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

out = Path("runs/npu-pacing-20261103")
out.mkdir(parents=True, exist_ok=True)

img = Image.open("data/test_dog.jpg").convert("RGB")
img = img.resize((448,448), Image.Resampling.BICUBIC)

x = np.asarray(img, dtype=np.float32) / 255.
mean = np.array([.485,.456,.406], np.float32)
std = np.array([.229,.224,.225], np.float32)
x = ((x-mean)/std).transpose(2,0,1)
x = np.ascontiguousarray(x[None], np.float32)

core = ov.Core()
model = core.read_model(
    "models/InternVL2-1B-int4-ov/"
    "openvino_vision_embeddings_model.xml"
)
model.reshape([1,3,448,448])
compiled = core.compile_model(model, "NPU")

print("Device:", compiled.get_property("EXECUTION_DEVICES"))

for _ in range(5):
    compiled([x])

with (out / "results.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["phase","rate","id","latency_ms"])

    for phase,rate in enumerate([2,5,2]):
        samples = []
        start = time.perf_counter()

        for i in range(30):
            due = start + i / rate
            delay = due-time.perf_counter()
            if delay > 0:
                time.sleep(delay)

            t0 = time.perf_counter()
            compiled([x])
            ms = (time.perf_counter()-t0)*1000
            samples.append(ms)
            writer.writerow([phase+1,rate,i,ms])

        print(
            f"Phase {phase+1}, {rate} req/s:",
            f"p50={np.median(samples):.2f}",
            f"first10={np.median(samples[:10]):.2f}",
            f"last10={np.median(samples[-10:]):.2f}",
            flush=True
        )

print("Results:",out)
