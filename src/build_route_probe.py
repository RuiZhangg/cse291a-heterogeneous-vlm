from pathlib import Path

source = Path("src/pipeline_video.py").read_text()

def replace_once(old, new):
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one match, got {count}: {old[:100]}")
    source = source.replace(old, new)

replace_once(
    "choices=['GPU','NPU','CPU']",
    "choices=['GPU','NPU','ORACLE']"
)

replace_once(
    "ap.add_argument('--out', required=True)",
    """ap.add_argument('--out', required=True)
ap.add_argument('--phase-rates', default='4,5,4')
ap.add_argument('--phase-requests', type=int, default=40)"""
)

replace_once(
    "a = ap.parse_args()",
    """a = ap.parse_args()
phase_rates = [float(r) for r in a.phase_rates.split(',')]
assert a.phase_requests > 0 and all(r > 0 for r in phase_rates)
a.requests = len(phase_rates) * a.phase_requests"""
)

replace_once(
    "vision = compile_model('vision_embeddings_model',\n"
    "                       a.vision_device, True)",
    """visions = {
    dev: compile_model('vision_embeddings_model', dev, True)
    for dev in ('GPU', 'NPU')
}"""
)

replace_once(
    "vr = vision.create_infer_request()",
    "vr = {dev: m.create_infer_request() for dev, m in visions.items()}"
)

replace_once("def encode(frame):", "def encode(frame, device):")

replace_once(
    "vr.infer({'pixel_values': frame})",
    "vr[device].infer({'pixel_values': frame})"
)

replace_once(
    """for _ in range(a.warmups):
    v,_ = encode(frames[0])
    d = decode(v)""",
    """for _ in range(a.warmups):
    for dev in ('GPU', 'NPU'):
        v,_ = encode(frames[0], dev)
        d = decode(v)"""
)

replace_once(
    "encode(frames[job['id'] % len(frames)])",
    "encode(frames[job['id'] % len(frames)], job['device'])"
)

replace_once(
    """start = time.perf_counter()
for i in range(a.requests):
    if a.rate:
        delay = start+i/a.rate-time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    vin.put({'id':i, 'arrival':time.perf_counter()})""",
    """start = time.perf_counter()
due = start

for i in range(a.requests):
    phase = i // a.phase_requests
    rate = phase_rates[phase]

    if i > 0:
        due += 1.0 / rate

    delay = due - time.perf_counter()
    if delay > 0:
        time.sleep(delay)

    if a.vision_device == 'ORACLE':
        device = 'GPU' if rate <= 4.5 else 'NPU'
    else:
        device = a.vision_device

    vin.put({
        'id': i,
        'arrival': time.perf_counter(),
        'device': device,
        'phase': phase,
        'offered_rate': rate,
    })"""
)

replace_once(
    "'id','e2e_ms','vision_wait_ms','vision_ms',",
    "'id','phase','offered_rate','device',"
    "'e2e_ms','vision_wait_ms','vision_ms',"
)

Path("src/pipeline_route_probe.py").write_text(source)
print("Created src/pipeline_route_probe.py")
