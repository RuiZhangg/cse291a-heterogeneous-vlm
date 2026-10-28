import time
import numpy as np
import openvino as ov

root = "models/InternVL2-1B-int4-ov/"
core = ov.Core()

def compile_vision(device):
    model = core.read_model(
        root + "openvino_vision_embeddings_model.xml"
    )
    model.reshape([1, 3, 448, 448])
    return core.compile_model(model, device)

gpu_vision = compile_vision("GPU")
npu_vision = compile_vision("NPU")

# Keep the language decoder resident on GPU as well.
decoder_model = core.read_model(
    root + "openvino_language_model.xml"
)
gpu_decoder = core.compile_model(decoder_model, "GPU")

gpu_req = gpu_vision.create_infer_request()
npu_req = npu_vision.create_infer_request()

x = np.zeros((1, 3, 448, 448), dtype=np.float32)

for name, req in [("GPU", gpu_req), ("NPU", npu_req)]:
    req.infer({"pixel_values": x})

for name, req in [
    ("GPU", gpu_req),
    ("NPU", npu_req),
    ("GPU", gpu_req),
    ("NPU", npu_req),
]:
    times = []
    for _ in range(5):
        t = time.perf_counter()
        req.infer({"pixel_values": x})
        times.append((time.perf_counter()-t)*1000)
    print(name, "median ms", round(np.median(times), 2))

print("Both vision models and GPU decoder resident: SUCCESS")
