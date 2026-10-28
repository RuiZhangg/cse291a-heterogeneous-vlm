import time
import numpy as np
import openvino as ov
from PIL import Image

MODEL = "models/InternVL2-1B-int4-ov/openvino_vision_embeddings_model.xml"
IMAGE = "data/test_dog.jpg"

# InternVL2 preprocessing for a single 448x448 image tile.
img = Image.open(IMAGE).convert("RGB")
img = img.resize((448, 448), Image.Resampling.BICUBIC)

x = np.asarray(img, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

x = ((x - mean) / std).transpose(2, 0, 1)
x = np.ascontiguousarray(x[None], dtype=np.float32)

core = ov.Core()
outputs = {}

for device in ["GPU", "NPU"]:
    print(f"\n===== {device} =====", flush=True)

    model = core.read_model(MODEL)
    model.reshape([1, 3, 448, 448])

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    print("Compilation:", round(time.perf_counter()-t0, 3), "s",
          flush=True)
    print("Actual device:",
          compiled.get_property("EXECUTION_DEVICES"),
          flush=True)

    # Warm up
    for _ in range(3):
        result = compiled([x])

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        result = compiled([x])
        times.append((time.perf_counter()-t0)*1000)

    y = np.array(next(iter(result.values())), copy=True)
    outputs[device] = y

    print("Output shape:", y.shape, flush=True)
    print("Median latency:", round(np.median(times), 2), "ms",
          flush=True)
    print("p95 latency:", round(np.percentile(times, 95), 2), "ms",
          flush=True)
    print("Mean latency:", round(np.mean(times), 2), "ms",
          flush=True)

a = outputs["GPU"].astype(np.float64).ravel()
b = outputs["NPU"].astype(np.float64).ravel()

cosine = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
relative_error = np.linalg.norm(a-b) / np.linalg.norm(a)

print("\n===== FEATURE AGREEMENT =====")
print("GPU/NPU cosine similarity:", float(cosine))
print("Relative L2 difference:", float(relative_error))
print("All outputs finite:",
      np.isfinite(a).all() and np.isfinite(b).all())
