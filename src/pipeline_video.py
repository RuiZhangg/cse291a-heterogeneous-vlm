import argparse, csv, queue, threading, time
from pathlib import Path
import numpy as np
import openvino as ov
import openvino_genai as ovg
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--vision-device', choices=['GPU','NPU','CPU'],
                required=True)
ap.add_argument('--rate', type=float, default=0,
                help='requests/sec, 0 = burst')
ap.add_argument('--requests', type=int, default=24)
ap.add_argument('--warmups', type=int, default=2)
ap.add_argument('--max-tokens', type=int, default=32)
ap.add_argument('--out', required=True)
ap.add_argument('--frames-dir', default='data/frames')
a = ap.parse_args()
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

vision = compile_model('vision_embeddings_model',
                       a.vision_device, True)
text = compile_model('text_embeddings_model', 'GPU')
lang = compile_model('language_model', 'GPU')

vr = vision.create_infer_request()
tr = text.create_infer_request()
lr = lang.create_infer_request()

tok = ovg.Tokenizer(str(folder))


# Prepare all frames before timed inference.
frame_paths = sorted(Path(a.frames_dir).rglob('*.jpg'))
if not frame_paths:
    raise RuntimeError(f'No JPG frames in {a.frames_dir}')

def preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((448, 448), Image.Resampling.BICUBIC)
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    x = ((x - mean) / std).transpose(2, 0, 1)
    return np.ascontiguousarray(x[None])

frames = [preprocess(p) for p in frame_paths]
print('Loaded video frames:', len(frames), flush=True)

question = 'Describe the main objects in this image in one sentence.'
raw = tok.apply_chat_template(
    [{'role':'user','content':'<image>\n'+question}],
    add_generation_prompt=True
)
assert raw.count('<image>') == 1

prompt = raw.replace(
    '<image>', '<img>'+'<IMG_CONTEXT>'*256+'</img>'
)
ids = np.asarray(
    tok.encode(prompt,add_special_tokens=False).input_ids.data
).copy().astype(np.int64)

marker = np.asarray(
    tok.encode('<IMG_CONTEXT>',add_special_tokens=False).input_ids.data
).ravel()
assert marker.size == 1
positions = ids == int(marker[0])
assert int(positions.sum()) == 256
length = ids.shape[1]

stop_ids = {int(tok.get_eos_token_id())}
for marker_str in ('<|im_end|>', '</s>'):
    st = np.asarray(
        tok.encode(marker_str,add_special_tokens=False).input_ids.data
    ).ravel()
    if st.size == 1:
        stop_ids.add(int(st[0]))

def encode(frame):
    start = time.perf_counter()
    out = vr.infer({'pixel_values': frame})
    visual = np.array(next(iter(out.values())), copy=True)
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
    v,_ = encode(frames[0])
    d = decode(v)
if a.warmups:
    print('Warmup caption:',d['caption'],flush=True)

vin = queue.Queue()
vout = queue.Queue()
results = []
errors = []

def vision_worker():
    try:
        while True:
            job = vin.get()
            if job is None:
                vout.put(None)
                return
            job['vision_start'] = time.perf_counter()
            job['visual'],job['vision_ms'] = encode(frames[job['id'] % len(frames)])
            job['vision_end'] = time.perf_counter()
            vout.put(job)
    except BaseException as e:
        errors.append(('vision',repr(e)))
        vout.put(None)

def decoder_worker():
    try:
        while True:
            job = vout.get()
            if job is None:
                return
            job['decode_start'] = time.perf_counter()
            job.update(decode(job.pop('visual')))
            job['complete'] = time.perf_counter()
            results.append(job)
    except BaseException as e:
        errors.append(('decoder',repr(e)))

vt = threading.Thread(target=vision_worker)
dt = threading.Thread(target=decoder_worker)
vt.start()
dt.start()

start = time.perf_counter()
for i in range(a.requests):
    if a.rate:
        delay = start+i/a.rate-time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    vin.put({'id':i, 'arrival':time.perf_counter()})

vin.put(None)
vt.join()
dt.join()

if errors or len(results) != a.requests:
    raise RuntimeError(
        f'errors={errors} completed={len(results)}/{a.requests}'
    )

for r in results:
    r['e2e_ms'] = (r['complete']-r['arrival'])*1000
    r['vision_wait_ms'] = (
        r['vision_start']-r['arrival']
    )*1000
    r['decoder_wait_ms'] = (
        r['decode_start']-r['vision_end']
    )*1000

out = Path(a.out)
out.parent.mkdir(parents=True,exist_ok=True)
fields = [
    'id','e2e_ms','vision_wait_ms','vision_ms',
    'decoder_wait_ms','prep_ms','ttft_ms',
    'decode_ms','tokens','caption'
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
