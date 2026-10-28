import time
import openvino as ov

path = "models/InternVL2-1B-int4-ov/openvino_vision_embeddings_model.xml"
core = ov.Core()

print("OpenVINO:", ov.__version__, flush=True)
model = core.read_model(path)

print("\n===== INPUTS =====", flush=True)
for p in model.inputs:
    print(p.any_name, p.partial_shape, p.element_type, flush=True)

print("\n===== OUTPUTS =====", flush=True)
for p in model.outputs:
    print(p.any_name, p.partial_shape, p.element_type, flush=True)

for device in ["CPU", "GPU", "NPU"]:
    print(f"\n===== {device} =====", flush=True)
    try:
        start = time.perf_counter()
        compiled = core.compile_model(model, device)
        elapsed = time.perf_counter() - start

        print("Compile time:", round(elapsed, 3), "s", flush=True)
        print("Execution devices:",
              compiled.get_property("EXECUTION_DEVICES"),
              flush=True)
        print("COMPILATION SUCCESS", flush=True)
    except Exception as e:
        print("COMPILATION FAILED", flush=True)
        print(type(e).__name__, str(e)[:2500], flush=True)
