import argparse
import time
import numpy as np
import openvino as ov
import openvino_genai as ovg
from PIL import Image
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--vision-device", choices=["CPU", "GPU", "NPU"],
                    default="NPU")
parser.add_argument("--max-tokens", type=int, default=32)
args = parser.parse_args()

folder = Path("models/InternVL2-1B-int4-ov")
core = ov.Core()

def compile_component(name, device, static_vision=False):
    model = core.read_model(str(folder / f"openvino_{name}.xml"))
    if static_vision:
        model.reshape([1, 3, 448, 448])
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    print(f"{name} on {device}: compiled in "
          f"{time.perf_counter()-t0:.2f}s", flush=True)
    print("  Actual device:",
          compiled.get_property("EXECUTION_DEVICES"), flush=True)
    return compiled

print("Loading tokenizer...", flush=True)
tokenizer = ovg.Tokenizer(str(folder))

print("Compiling models...", flush=True)
vision = compile_component("vision_embeddings_model",
                           args.vision_device, True)
text = compile_component("text_embeddings_model", "GPU")
language = compile_component("language_model", "GPU")

# Single-tile InternVL2 image preprocessing
image = Image.open("data/test_dog.jpg").convert("RGB")
image = image.resize((448, 448), Image.Resampling.BICUBIC)

x = np.asarray(image, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x = ((x - mean) / std).transpose(2, 0, 1)
x = np.ascontiguousarray(x[None], dtype=np.float32)

# Encode image
t0 = time.perf_counter()
vision_result = vision({"pixel_values": x})
visual = np.array(next(iter(vision_result.values())), copy=True)
vision_time = time.perf_counter() - t0

print("Vision output:", visual.shape, flush=True)

# InternVL image placeholder token expansion
question = "Describe the main objects in this image in one sentence."

raw_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "<image>\n" + question}],
    add_generation_prompt=True
)

if raw_prompt.count("<image>") != 1:
    raise ValueError(
        "Unexpected chat-template image markers: "
        + repr(raw_prompt[:300])
    )

image_marker = "<IMG_CONTEXT>"
expanded = "<img>" + image_marker * 256 + "</img>"
prompt = raw_prompt.replace("<image>", expanded)

ids = np.array(
    tokenizer.encode(prompt, add_special_tokens=False).input_ids.data,
    copy=True
).astype(np.int64)

marker_ids = np.array(
    tokenizer.encode(image_marker, add_special_tokens=False).input_ids.data
).ravel()

if len(marker_ids) != 1:
    raise ValueError(
        f"Image marker is not a single token: {marker_ids}"
    )

image_positions = ids == int(marker_ids[0])
count = int(image_positions.sum())
print("Image placeholder count:", count, flush=True)

if count != visual.shape[1]:
    raise ValueError(
        f"Expected {visual.shape[1]} image placeholders, got {count}. "
        "Stopping to avoid incorrect visual embedding insertion."
    )

# Text embeddings
text_result = text({"input": ids})
embeddings = np.array(next(iter(text_result.values())), copy=True)

if embeddings.shape[-1] != visual.shape[-1]:
    raise ValueError("Vision/text hidden dimensions differ")

embeddings[image_positions] = visual.reshape(-1, visual.shape[-1])
print("Combined embeddings:", embeddings.shape, flush=True)

# Stateful autoregressive language generation
request = language.create_infer_request()
request.reset_state()

length = ids.shape[1]
attention_mask = np.ones((1, length), dtype=np.int64)
position_ids = np.arange(length, dtype=np.int64)[None, :]
beam_idx = np.array([0], dtype=np.int32)

current_embeds = embeddings
generated = []
decode_start = time.perf_counter()
first_token_time = None

# InternVL2 may use both EOS and chat end-of-turn markers.
stop_ids = {int(tokenizer.get_eos_token_id())}
for marker in ("<|im_end|>", "</s>"):
    marker_tokens = np.asarray(
        tokenizer.encode(marker, add_special_tokens=False).input_ids.data
    ).ravel()
    if len(marker_tokens) == 1:
        stop_ids.add(int(marker_tokens[0]))

for step in range(args.max_tokens):
    outputs = request.infer({
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "inputs_embeds": current_embeds,
        "beam_idx": beam_idx,
    })

    logits = next(iter(outputs.values()))
    token = int(np.argmax(logits[0, -1]))

    if first_token_time is None:
        first_token_time = time.perf_counter() - decode_start

    if token in stop_ids:
        break

    generated.append(token)

    if step == args.max_tokens - 1:
        break

    # Embed the newly generated token for the next decoding step.
    token_ids = np.array([[token]], dtype=np.int64)
    result = text({"input": token_ids})
    current_embeds = np.array(next(iter(result.values())), copy=True)

    position_ids = np.array([[length + len(generated) - 1]],
                            dtype=np.int64)
    attention_mask = np.ones((1, length + len(generated)),
                             dtype=np.int64)

decode_time = time.perf_counter() - decode_start
caption = tokenizer.decode(generated)

print("\n===== RESULT =====")
print("Vision device:", args.vision_device)
print("Caption:", caption)
print("Vision encoding:", round(vision_time * 1000, 2), "ms")
print("First token:", round(first_token_time * 1000, 2), "ms")
print("Decoder duration:", round(decode_time, 3), "s")
print("Generated tokens:", len(generated))
print("SUCCESS", flush=True)
