import time
import numpy as np
import openvino as ov

core = ov.Core()
path = "models/InternVL2-1B-int4-ov/openvino_vision_embeddings_model.xml"

model = core.read_model(path)
print("Original input:", model.input(0).partial_shape, flush=True)

model.reshape([1, 3, 448, 448])

print("Static input:", model.input(0).partial_shape, flush=True)
print("Static output:", model.output(0).partial_shape, flush=True)

for device in ["NPU"]:
    print(f"\nTesting {device}", flush=True)

    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    print("Compile time:", time.perf_counter() - t0, flush=True)
    print("Execution devices:",
          compiled.get_property("EXECUTION_DEVICES"), flush=True)

    # Synthetic input for execution compatibility only.
    # This is NOT a caption-quality test.
    x = np.zeros((1, 3, 448, 448), dtype=np.float32)

    t0 = time.perf_counter()
    result = compiled([x])
    print("First inference:", time.perf_counter() - t0, flush=True)

    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        result = compiled([x])
        times.append(time.perf_counter() - t0)

    output = next(iter(result.values()))
    print("Output shape:", output.shape, flush=True)
    print("Output finite:", bool(np.isfinite(output).all()), flush=True)
    print("Warm mean:", np.mean(times), flush=True)
    print("SUCCESS", flush=True)
