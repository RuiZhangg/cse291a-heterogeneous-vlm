import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", type=int, default=50)
parser.add_argument("--warmups", type=int, default=5)
parser.add_argument("--load-seconds", type=int, default=75)
parser.add_argument("--out", required=True)
args = parser.parse_args()

out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)

ready = out / "background.ready"
go = out / "background.go"

ready.unlink(missing_ok=True)
go.unlink(missing_ok=True)

model_path = (
    "models/InternVL2-1B-int4-ov/"
    "openvino_vision_embeddings_model.xml"
)

# Identical image preprocessing for GPU and NPU.
img = Image.open("data/test_dog.jpg").convert("RGB")
img = img.resize((448, 448), Image.Resampling.BICUBIC)

x = np.asarray(img, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

x = ((x - mean) / std).transpose(2, 0, 1)
x = np.ascontiguousarray(x[None], dtype=np.float32)

core = ov.Core()
compiled = {}
metadata = {
    "openvino": ov.__version__,
    "pairs": args.pairs,
    "warmups": args.warmups,
    "background_seconds": args.load_seconds,
    "devices": {},
}

print("Compiling both vision encoders...", flush=True)

for device in ("GPU", "NPU"):
    model = core.read_model(model_path)
    model.reshape([1, 3, 448, 448])

    start = time.perf_counter()
    compiled[device] = core.compile_model(model, device)

    compile_s = time.perf_counter() - start
    actual = str(
        compiled[device].get_property("EXECUTION_DEVICES")
    )

    metadata["devices"][device] = {
        "compile_s": compile_s,
        "execution_devices": actual,
    }

    print(
        device,
        "compile_s =", round(compile_s, 3),
        "actual =", actual,
        flush=True,
    )

    for _ in range(args.warmups):
        compiled[device]([x])

with (out / "metadata.json").open("w") as f:
    json.dump(metadata, f, indent=2)

rng = np.random.default_rng(20261103)

csv_path = out / "vision_latency.csv"
fields = [
    "condition",
    "pair",
    "device",
    "start_ns",
    "end_ns",
    "latency_ms",
]

background = None
background_log = None

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    def measure(condition):
        print(f"\n===== {condition} =====", flush=True)

        samples = {"GPU": [], "NPU": []}

        for pair in range(args.pairs):
            # Randomize the order within each measurement pair.
            order = rng.permutation(["GPU", "NPU"])

            for device in order:
                t0 = time.perf_counter_ns()
                compiled[device]([x])
                t1 = time.perf_counter_ns()

                ms = (t1 - t0) / 1e6
                samples[device].append(ms)

                writer.writerow({
                    "condition": condition,
                    "pair": pair,
                    "device": device,
                    "start_ns": t0,
                    "end_ns": t1,
                    "latency_ms": ms,
                })

        f.flush()

        for device in ("GPU", "NPU"):
            values = np.asarray(samples[device])

            print(
                f"{device}: "
                f"mean={values.mean():.2f} "
                f"p50={np.median(values):.2f} "
                f"p95={np.percentile(values,95):.2f} ms",
                flush=True,
            )

    try:
        # A: No background model.
        measure("baseline")

        # Start the second GPU VLM, but leave it idle.
        background_log = (
            out / "background_gpu_vlm.log"
        ).open("w")

        background = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "src/gpu_caption_load.py",
                "--ready", str(ready),
                "--go", str(go),
                "--duration", str(args.load_seconds),
            ],
            stdout=background_log,
            stderr=subprocess.STDOUT,
        )

        # Wait for the full VLM to finish loading/warmup.
        deadline = time.monotonic() + 120

        while not ready.exists():
            if background.poll() is not None:
                raise RuntimeError(
                    "Background process exited before READY. "
                    "See background_gpu_vlm.log"
                )

            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Background GPU VLM initialization timed out"
                )

            time.sleep(0.2)

        print("\nBackground VLM loaded and idle.", flush=True)

        # B: Background model resident but idle.
        measure("resident_idle")

        # C: Begin continuous GPU VLM generation.
        go.touch()

        # Ensure background generation has actually begun.
        deadline = time.monotonic() + 20
        while True:
            background_log.flush()

            log_text = (
                out / "background_gpu_vlm.log"
            ).read_text(errors="replace")

            if "GENERATION_START" in log_text:
                break

            if background.poll() is not None:
                raise RuntimeError(
                    "Background process exited before generation"
                )

            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Background generation did not begin"
                )

            time.sleep(0.1)

        measure("active_gpu_load")

        if background.poll() is not None:
            print(
                "WARNING: background load ended before active "
                "measurements finished; inspect timestamps/logs",
                flush=True,
            )

        # Allow the background generation process to finish.
        background.wait(timeout=args.load_seconds + 30)

        if background.returncode != 0:
            raise RuntimeError(
                "Background GPU VLM exited nonzero; "
                "see background_gpu_vlm.log"
            )

        # D: Check whether latency returns after load exits.
        measure("recovery")

    finally:
        if background is not None and background.poll() is None:
            background.terminate()
            try:
                background.wait(timeout=10)
            except subprocess.TimeoutExpired:
                background.kill()
                background.wait()

        if background_log is not None:
            background_log.close()

print("\nSaved:", csv_path)
print("Experiment complete.")
