import argparse
import time
from pathlib import Path
import numpy as np
import openvino as ov
import openvino_genai as ovg
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--ready", required=True)
parser.add_argument("--go", required=True)
parser.add_argument("--duration", type=int, default=45)
args = parser.parse_args()

ready = Path(args.ready)
go = Path(args.go)

image = np.asarray(
    Image.open("data/test_dog.jpg").convert("RGB"),
    dtype=np.uint8
)[None, ...]

tensor = ov.Tensor(image)

print("Loading GPU VLM...", flush=True)
pipe = ovg.VLMPipeline(
    "models/InternVL2-1B-int4-ov", "GPU"
)

prompt = (
    "Describe this scene in considerable detail, including "
    "the foreground, background, objects, visual properties, "
    "and their spatial relationships."
)

print("Warming up GPU VLM...", flush=True)
pipe.generate(prompt, image=tensor, max_new_tokens=128)

ready.touch()
print("READY: GPU model resident but idle", flush=True)

while not go.exists():
    time.sleep(0.05)

deadline = time.monotonic() + args.duration
count = 0

while time.monotonic() < deadline:
    print("GENERATION_START", count, flush=True)
    t0 = time.perf_counter()

    result = pipe.generate(
        prompt, image=tensor, max_new_tokens=128
    )

    elapsed = time.perf_counter() - t0
    count += 1
    print(
        f"Generation {count}: {elapsed:.3f}s",
        flush=True
    )

print("Completed generation requests:", count, flush=True)
