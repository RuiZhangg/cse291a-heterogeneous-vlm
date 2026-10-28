import argparse
import time
import json
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ovg
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="GPU", choices=["CPU", "GPU", "NPU"])
parser.add_argument("--image", default="data/test_dog.jpg")
args = parser.parse_args()

model_dir = "models/InternVL2-1B-int4-ov"

print("Device:", args.device, flush=True)
print("Model:", model_dir, flush=True)
print("OpenVINO:", ov.__version__, flush=True)

# Load and compile the VLM.
t0 = time.perf_counter()
pipe = ovg.VLMPipeline(model_dir, args.device)
load_time = time.perf_counter() - t0

print(f"Pipeline initialization: {load_time:.3f} s", flush=True)

# Prepare the image.
image = Image.open(args.image).convert("RGB")
array = np.asarray(image, dtype=np.uint8)[None, ...]
tensor = ov.Tensor(array)

print("Image shape:", array.shape, flush=True)

prompt = "Describe the main objects and activities in this image in one concise sentence."

# First inference; includes warmup costs.
t0 = time.perf_counter()
result = pipe.generate(
    prompt,
    image=tensor,
    max_new_tokens=48,
)
first_time = time.perf_counter() - t0

caption = result.texts[0] if hasattr(result, "texts") else str(result)

print("\n===== CAPTION =====", flush=True)
print(caption, flush=True)
print(f"\nFirst generation: {first_time:.3f} s", flush=True)

# Warm inference using the same input.
t0 = time.perf_counter()
result2 = pipe.generate(
    prompt,
    image=tensor,
    max_new_tokens=48,
)
warm_time = time.perf_counter() - t0

print(f"Warm generation: {warm_time:.3f} s", flush=True)

Path("logs").mkdir(exist_ok=True)

print("\nSUCCESS", flush=True)
