from pathlib import Path
import re

src = Path("src/pipeline_route_online.py")
s = src.read_text()

# Add a comparison-mode flag before argument parsing.
m = re.search(
    r'(?m)^a\s*=\s*([A-Za-z_]\w*)\.parse_args\(\)\s*$',
    s
)
assert m is not None, "Could not find argument parser"

parser_name = m.group(1)
insertion = (
    f'{parser_name}.add_argument('
    '"--worker-mode", '
    'choices=["serial","parallel"], '
    'default="serial")\n'
)

s = s[:m.start()] + insertion + s[m.start():]

start = s.find("\nvin = queue.Queue()")
end = s.find("\nout = Path(a.out)", start)

assert start >= 0 and end > start, (
    "Could not locate worker/arrival section; "
    "original source was not modified"
)

replacement = '''
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
'''

s = s[:start] + "\n" + replacement + s[end:]

# Save raw timestamps so actual GPU/NPU overlap can be measured.
fields_start = s.find("fields = [", s.find("out = Path(a.out)"))
fields_end = s.find("\n]", fields_start)

assert fields_start >= 0 and fields_end > fields_start

block = s[fields_start:fields_end]

assert "'tokens','caption'" in block, (
    "Unexpected output fields; inspect original source"
)

block = block.replace(
    "'tokens','caption'",
    "'tokens','caption',"
    "'arrival','vision_start','vision_end',"
    "'decode_start','complete'",
    1
)

s = s[:fields_start] + block + s[fields_end:]

out = Path("src/pipeline_parallel_compare.py")
out.write_text(s)

print("Created:", out)
print("Original file unchanged:", src)
