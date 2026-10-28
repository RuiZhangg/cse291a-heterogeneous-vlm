import json
from pathlib import Path
import openvino as ov

folder = Path("models/InternVL2-1B-int4-ov")
core = ov.Core()

for name in [
    "openvino_vision_embeddings_model",
    "openvino_text_embeddings_model",
    "openvino_language_model",
]:
    print(f"\n========== {name} ==========", flush=True)

    model = core.read_model(str(folder / f"{name}.xml"))

    print("Inputs:", len(model.inputs))
    for p in list(model.inputs)[:15]:
        print(" ", p.any_name, p.partial_shape, p.element_type)

    print("Outputs:", len(model.outputs))
    for p in list(model.outputs)[:15]:
        print(" ", p.any_name, p.partial_shape, p.element_type)

    print("State sinks:", len(model.get_sinks()))

for filename in ["config.json", "preprocessor_config.json"]:
    path = folder / filename
    if not path.exists():
        continue

    print(f"\n========== {filename} ==========")
    data = json.loads(path.read_text())

    for key in [
        "model_type", "hidden_size", "image_context_token",
        "image_size", "crop_size", "patch_size",
        "vision_config", "text_config",
    ]:
        if key in data:
            value = data[key]
            if isinstance(value, dict):
                print(key, ":", {
                    k: value[k] for k in [
                        "model_type", "hidden_size", "image_size",
                        "patch_size", "architectures"
                    ] if k in value
                })
            else:
                print(key, ":", value)
