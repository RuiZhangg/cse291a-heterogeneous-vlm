import argparse
import csv
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--device", choices=["GPU","NPU"], required=True)
p.add_argument("--repeats", type=int, default=15)
p.add_argument("--out", required=True)
a = p.parse_args()

out = Path(a.out)
out.parent.mkdir(parents=True,exist_ok=True)

image = Image.open("data/test_dog.jpg").convert("RGB")
w,h = image.size

regions = [
    image,
    image.crop((0,0,w//2,h//2)),
    image.crop((w//2,0,w,h//2)),
    image.crop((0,h//2,w,h)),
]

def preprocess(im):
    im = im.resize((448,448),Image.Resampling.BICUBIC)
    x = np.asarray(im,dtype=np.float32)/255.0
    mean=np.array([.485,.456,.406],np.float32)
    std=np.array([.229,.224,.225],np.float32)
    return np.ascontiguousarray(
        ((x-mean)/std).transpose(2,0,1)[None],
        dtype=np.float32
    )

tiles = [preprocess(im) for im in regions]

core=ov.Core()
model=core.read_model(
    "models/InternVL2-1B-int4-ov/"
    "openvino_vision_embeddings_model.xml"
)
model.reshape([1,3,448,448])

compiled=core.compile_model(model,a.device)
req=compiled.create_infer_request()

print("Actual device:",
      compiled.get_property("EXECUTION_DEVICES"),
      flush=True)

for _ in range(5):
    req.infer([tiles[0]])

rng=np.random.default_rng(20261103)
rows=[]

# Vary the time the accelerator has been idle before
# processing a frame. Within each frame, tiles are
# processed back-to-back.
cases=[(n,gap) for n in (1,2,4)
       for gap in (0.0,0.15,0.5)]

for rep in range(a.repeats):
    for n,gap in rng.permutation(cases):

        # Restore an active state before each case.
        for _ in range(3):
            req.infer([tiles[0]])

        if gap:
            time.sleep(float(gap))

        frame_start=time.perf_counter()
        per_tile=[]

        for tile in tiles[:int(n)]:
            start=time.perf_counter()
            req.infer([tile])
            per_tile.append(
                (time.perf_counter()-start)*1000
            )

        total=(time.perf_counter()-frame_start)*1000

        rows.append({
            "device":a.device,
            "repeat":rep,
            "tiles":int(n),
            "idle_gap_ms":float(gap)*1000,
            "frame_ms":total,
            "first_tile_ms":per_tile[0],
            "later_tiles_mean_ms":(
                float(np.mean(per_tile[1:]))
                if len(per_tile)>1 else ""
            )
        })

with out.open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

for n in (1,2,4):
    for gap in (0.0,0.15,0.5):
        sub=[r for r in rows
             if r["tiles"]==n
             and r["idle_gap_ms"]==gap*1000]

        frame=np.array([r["frame_ms"] for r in sub])
        first=np.array([r["first_tile_ms"] for r in sub])

        print(
            f"{n} tiles, gap={gap*1000:.0f}ms:",
            f"frame_p50={np.median(frame):.1f}ms",
            f"first_p50={np.median(first):.1f}ms",
            flush=True
        )

print("Saved:",out)
