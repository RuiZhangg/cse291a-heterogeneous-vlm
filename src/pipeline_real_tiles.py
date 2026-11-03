import argparse, csv, queue, threading, time
from pathlib import Path
import numpy as np
import openvino as ov
import openvino_genai as ovg
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--vision-device', choices=['GPU','NPU','ORACLE','ONLINE'],
                required=True)
ap.add_argument('--rate', type=float, default=0,
                help='requests/sec, 0 = burst')
ap.add_argument('--requests', type=int, default=24)
ap.add_argument('--warmups', type=int, default=2)
ap.add_argument('--max-tokens', type=int, default=32)
ap.add_argument('--out', required=True)
ap.add_argument('--phase-rates', default='4,5,4')
ap.add_argument('--phase-requests', type=int, default=40)
ap.add_argument('--seed', type=int, default=20261028)
ap.add_argument('--frames-dir', default='data/frames')
ap.add_argument("--worker-mode", choices=["serial","parallel"], default="serial")
ap.add_argument('--tiles', type=int, choices=[1, 2, 4], default=1)
ap.add_argument('--thumbnail', action='store_true',
                help='Add full-image thumbnail after grid patches')
a = ap.parse_args()
phase_rates = [float(r) for r in a.phase_rates.split(',')]
assert a.phase_requests > 0 and all(r > 0 for r in phase_rates)
a.requests = len(phase_rates) * a.phase_requests
assert a.rate >= 0 and a.requests > 0

folder = Path('/data/project-03/models/InternVL2-1B-int4-ov')
core = ov.Core()

def compile_model(stem, device, reshape=False):
    m = core.read_model(str(folder / ('openvino_' + stem + '.xml')))
    if reshape:
        m.reshape([1,3,448,448])
    t = time.perf_counter()
    c = core.compile_model(m, device)
    print(stem, device, 'compile_s',
          round(time.perf_counter()-t,3),
          'execution_devices',
          c.get_property('EXECUTION_DEVICES'), flush=True)
    return c

visions = {
    dev: compile_model('vision_embeddings_model', dev, True)
    for dev in ('GPU', 'NPU')
}
text = compile_model('text_embeddings_model', 'GPU')
lang = compile_model('language_model', 'GPU')

vr = {dev: m.create_infer_request() for dev, m in visions.items()}
tr = text.create_infer_request()
lr = lang.create_infer_request()

tok = ovg.Tokenizer(str(folder))


# Prepare all frames before timed inference.
frame_paths = sorted(Path(a.frames_dir).rglob('*.jpg'))
if not frame_paths:
    raise RuntimeError(f'No JPG frames in {a.frames_dir}')

def preprocess(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size

    if a.tiles == 1:
        crops = [img]
        grid = (1, 1)
    else:
        # Exact number of spatial tiles, keeping grid aspect ratio
        # as close to the original image as possible.
        candidates = [(cols, a.tiles // cols)
                      for cols in range(1, a.tiles + 1)
                      if a.tiles % cols == 0]
        cols, rows = min(candidates,
                         key=lambda z: abs((z[0] / z[1]) - (w / h)))
        grid = (cols, rows)
        large = img.resize((cols * 448, rows * 448),
                           Image.Resampling.BICUBIC)
        crops = [large.crop((col * 448, row * 448,
                             (col + 1) * 448, (row + 1) * 448))
                 for row in range(rows) for col in range(cols)]
        if a.thumbnail:
            crops.append(img)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    output = []
    for crop in crops:
        x = np.asarray(crop.resize((448, 448), Image.Resampling.BICUBIC),
                       dtype=np.float32) / 255.0
        x = ((x - mean) / std).transpose(2, 0, 1)
        output.append(np.ascontiguousarray(x[None], dtype=np.float32))
    return output

frames = [preprocess(p) for p in frame_paths]
print('Loaded video frames:', len(frames), flush=True)

question = 'Describe the main objects in this image in one sentence.'
raw = tok.apply_chat_template(
    [{'role':'user','content':'<image>\n'+question}],
    add_generation_prompt=True
)
assert raw.count('<image>') == 1

tile_count = len(frames[0])
assert all(len(frame) == tile_count for frame in frames)
print('Image tiles per frame:', tile_count, 'grid tiles:', a.tiles,
      'thumbnail:', bool(a.thumbnail), flush=True)
prompt = raw.replace(
    '<image>', '<img>' + '<IMG_CONTEXT>' * (256 * tile_count) + '</img>'
)

ids = np.asarray(
    tok.encode(prompt,add_special_tokens=False).input_ids.data
).copy().astype(np.int64)

marker = np.asarray(
    tok.encode('<IMG_CONTEXT>',add_special_tokens=False).input_ids.data
).ravel()
assert marker.size == 1
positions = ids == int(marker[0])
assert int(positions.sum()) == 256 * tile_count
length = ids.shape[1]

stop_ids = {int(tok.get_eos_token_id())}
for marker_str in ('<|im_end|>', '</s>'):
    st = np.asarray(
        tok.encode(marker_str,add_special_tokens=False).input_ids.data
    ).ravel()
    if st.size == 1:
        stop_ids.add(int(st[0]))

def encode(frame, device):
    start = time.perf_counter()
    visual_tiles = []
    for tile in frame:
        out = vr[device].infer({'pixel_values': tile})
        features = np.array(next(iter(out.values())), copy=True)
        assert features.shape == (1, 256, 896), features.shape
        visual_tiles.append(features)
    visual = np.concatenate(visual_tiles, axis=0)
    assert visual.shape == (tile_count, 256, 896)

    return visual, (time.perf_counter()-start)*1000

def decode(visual):
    start = time.perf_counter()
    embout = tr.infer({'input':ids})
    emb = np.array(next(iter(embout.values())),copy=True)
    emb[positions] = visual.reshape(-1,896)
    prep_ms = (time.perf_counter()-start)*1000

    lr.reset_state()
    cur = emb
    mask = np.ones((1,length),dtype=np.int64)
    pos = np.arange(length,dtype=np.int64)[None,:]
    generated = []
    decode_start = time.perf_counter()
    ttft_ms = 0

    for k in range(a.max_tokens):
        out = lr.infer({
            'attention_mask':mask,
            'position_ids':pos,
            'inputs_embeds':cur,
            'beam_idx':np.array([0],dtype=np.int32)
        })
        token = int(np.argmax(
            next(iter(out.values()))[0,-1]
        ))

        if k == 0:
            ttft_ms = (time.perf_counter()-decode_start)*1000
        if token in stop_ids:
            break

        generated.append(token)
        if k == a.max_tokens-1:
            break

        next_out = tr.infer({
            'input':np.array([[token]],dtype=np.int64)
        })
        cur = np.array(next(iter(next_out.values())),copy=True)
        pos = np.array(
            [[length+len(generated)-1]],dtype=np.int64
        )
        mask = np.ones(
            (1,length+len(generated)),dtype=np.int64
        )

    return {
        'prep_ms':prep_ms,
        'decode_ms':(time.perf_counter()-decode_start)*1000,
        'ttft_ms':ttft_ms,
        'tokens':len(generated),
        'caption':tok.decode(generated)
    }

# Warmup before starting timed workload
print('Warmups:', a.warmups, flush=True)
for _ in range(a.warmups):
    for dev in ('GPU', 'NPU'):
        v,_ = encode(frames[0], dev)
        d = decode(v)
if a.warmups:
    print('Warmup caption:',d['caption'],flush=True)


# Experiment-only architecture comparison.
# GPU/NPU assignment is predetermined and independent of
# queue lengths, worker state or decoder state.

shared_q = queue.Queue()
device_q = {
    "GPU": queue.Queue(),
    "NPU": queue.Queue(),
}
vout = queue.Queue()

results = []
errors = []

producer_count = 1 if a.worker_mode == "serial" else 2

def vision_worker(q):
    try:
        while True:
            job = q.get()

            if job is None:
                return

            job['vision_start'] = time.perf_counter()

            job['visual'], job['vision_ms'] = encode(
                frames[job['id'] % len(frames)],
                job['device']
            )

            job['vision_end'] = time.perf_counter()
            vout.put(job)

    except BaseException as e:
        errors.append(('vision', repr(e)))

    finally:
        vout.put(None)

def decoder_worker():
    try:
        finished = 0

        while finished < producer_count:
            job = vout.get()

            if job is None:
                finished += 1
                continue

            job['decode_start'] = time.perf_counter()
            job.update(decode(job.pop('visual')))
            job['complete'] = time.perf_counter()

            results.append(job)

    except BaseException as e:
        errors.append(('decoder', repr(e)))

active_queues = (
    [shared_q]
    if a.worker_mode == "serial"
    else [device_q["GPU"], device_q["NPU"]]
)

threads = [
    threading.Thread(target=vision_worker, args=(q,))
    for q in active_queues
]

dt = threading.Thread(target=decoder_worker)

for t in threads:
    t.start()

dt.start()

start = time.perf_counter()
due = start
rng = np.random.default_rng(a.seed)

print(
    "Worker mode:", a.worker_mode,
    "Assignment: alternating GPU/NPU",
    flush=True
)

for i in range(a.requests):
    phase = i // a.phase_requests
    rate = phase_rates[phase]

    if i > 0:
        due += rng.exponential(1.0 / rate)

    delay = due - time.perf_counter()

    if delay > 0:
        time.sleep(delay)

    device = "GPU" if i % 2 == 0 else "NPU"

    job = {
        'id': i,
        'arrival': time.perf_counter(),
        'device': device,
        'phase': phase,
        'offered_rate': rate,
    }

    if a.worker_mode == "serial":
        shared_q.put(job)
    else:
        device_q[device].put(job)

for q in active_queues:
    q.put(None)

for t in threads:
    t.join()

dt.join()

if errors or len(results) != a.requests:
    raise RuntimeError(
        f'errors={errors} completed={len(results)}/{a.requests}'
    )

for r in results:
    r['e2e_ms'] = (
        r['complete'] - r['arrival']
    ) * 1000

    r['vision_wait_ms'] = (
        r['vision_start'] - r['arrival']
    ) * 1000

    r['decoder_wait_ms'] = (
        r['decode_start'] - r['vision_end']
    ) * 1000

out = Path(a.out)
out.parent.mkdir(parents=True,exist_ok=True)
fields = [
    'id','phase','offered_rate','device','e2e_ms','vision_wait_ms','vision_ms',
    'decoder_wait_ms','prep_ms','ttft_ms',
    'decode_ms','tokens','caption','arrival','vision_start','vision_end','decode_start','complete'
]
with out.open('w',newline='') as f:
    writer = csv.DictWriter(
        f,fieldnames=fields,extrasaction='ignore'
    )
    writer.writeheader()
    writer.writerows(
        sorted(results,key=lambda r:r['id'])
    )

span = (
    max(r['complete'] for r in results) -
    min(r['arrival'] for r in results)
)
print('Completed:',len(results),
      'Throughput:',round(len(results)/span,3),'requests/s')

for metric in [
    'e2e_ms','vision_wait_ms','vision_ms',
    'decoder_wait_ms','prep_ms','ttft_ms','decode_ms'
]:
    values = np.array([r[metric] for r in results])
    print(metric,
          'p50',round(np.median(values),2),
          'p95',round(np.percentile(values,95),2))

print('Captions identical:',
      len(set(r['caption'] for r in results))==1)
print('CSV:',out,flush=True)
