import time
import numpy as np
import openvino as ov
from openvino import opset8 as ops

core = ov.Core()

# Small FP16 convolutional model
x = ops.parameter([1, 3, 64, 64], dtype=ov.Type.f16)
weights = ops.constant(
    np.ones((8, 3, 3, 3), dtype=np.float16) * 0.01
)
conv = ops.convolution(
    x, weights, [1, 1], [1, 1], [1, 1], [1, 1]
)
output = ops.relu(conv)
model = ov.Model([output], [x], "device_test")

test_input = np.ones((1, 3, 64, 64), dtype=np.float16)

for device in ["CPU", "GPU", "NPU"]:
    print(f"\n===== {device} =====", flush=True)
    try:
        start = time.perf_counter()
        compiled = core.compile_model(model, device)
        compile_time = time.perf_counter() - start

        print("Compile time:", round(compile_time, 3), "s")

        try:
            print("Execution devices:",
                  compiled.get_property("EXECUTION_DEVICES"))
        except Exception as e:
            print("Execution-device property unavailable:", e)

        start = time.perf_counter()
        result = compiled([test_input])
        first_time = time.perf_counter() - start

        timings = []
        for _ in range(5):
            start = time.perf_counter()
            result = compiled([test_input])
            timings.append(time.perf_counter() - start)

        arr = next(iter(result.values()))

        print("First inference:", round(first_time * 1000, 2), "ms")
        print("Warm mean:", round(np.mean(timings) * 1000, 2), "ms")
        print("Output shape:", arr.shape)
        print("Output mean:", float(arr.mean()))
        print("Status: SUCCESS")

    except Exception as e:
        print("Status: FAILED")
        print(type(e).__name__, str(e)[:1500])
