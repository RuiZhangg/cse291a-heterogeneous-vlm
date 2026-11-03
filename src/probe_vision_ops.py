import argparse
import time

import numpy as np
import openvino as ov
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--device", choices=["GPU","NPU"],
               required=True)
p.add_argument("--gap", type=float, default=0)
a = p.parse_args()

img = Image.open("data/test_dog.jpg").convert("RGB")
img = img.resize((448,448), Image.Resampling.BICUBIC)

x = np.asarray(img, np.float32) / 255.
x = (x - np.array([.485,.456,.406],np.float32)) / \
        np.array([.229,.224,.225],np.float32)
x = np.ascontiguousarray(
    x.transpose(2,0,1)[None], dtype=np.float32
)

core = ov.Core()
model = core.read_model(
    "models/InternVL2-1B-int4-ov/"
    "openvino_vision_embeddings_model.xml"
)
model.reshape([1,3,448,448])

try:
    compiled = core.compile_model(
        model, a.device, {"PERF_COUNT": "YES"}
    )
except Exception as e:
    print("Profiling compilation unsupported:", e)
    raise SystemExit(1)

req = compiled.create_infer_request()

for _ in range(5):
    req.infer([x])

time.sleep(a.gap)

t0 = time.perf_counter()
req.infer([x])
wall_ms = (time.perf_counter()-t0)*1000

print("Device:", compiled.get_property("EXECUTION_DEVICES"))
print("Call wall time:", round(wall_ms,2), "ms")

try:
    info = req.get_profiling_info()
except Exception as e:
    print("Profiling unavailable:", e)
    raise SystemExit(0)

def ms(v):
    if hasattr(v, "total_seconds"):
        return v.total_seconds() * 1000
    return float(v) / 1000

executed = [
    x for x in info
    if x.status == ov.ProfilingInfo.Status.EXECUTED
]

executed.sort(
    key=lambda x: ms(x.real_time), reverse=True
)

print("Executed entries:",len(executed))

for item in executed[:15]:
    print(
        f"{ms(item.real_time):9.3f} ms",
        item.node_type,
        item.exec_type,
        item.node_name[:80]
    )
