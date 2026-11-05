"""Derive a profiling-only variant of the verified NUC pipeline.

Run in /data/project-03 with Python 3.12. Does NOT modify the original.
Records startup/setup boundaries and host-side feature staging.
No claim that these timings isolate accelerator bus transfers.
"""
from pathlib import Path

src = Path('src/pipeline_real_tiles_fixed.py')
s = src.read_text()

def one_replace(old, new):
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f'Expected exactly one match ({n} found): {old[:80]!r}')
    s = s.replace(old, new, 1)

s = "import time as _phase1_clock\n_phase1_boot_t0 = _phase1_clock.perf_counter()\n" + s
one_replace(
    "tok = ovg.Tokenizer(str(folder))",
    "tok = ovg.Tokenizer(str(folder))\n"
    "print('PROFILE init_to_tokenizer_seconds', "
    "round(_phase1_clock.perf_counter()-_phase1_boot_t0, 4), flush=True)"
)
one_replace(
    '# Warmup before starting timed workload',
    "print('PROFILE init_through_frame_preprocessing_seconds', "
    "round(_phase1_clock.perf_counter()-_phase1_boot_t0, 4), flush=True)\n"
    "# Warmup before starting timed workload"
)
one_replace(
    '''    visual_tiles = []
    for tile in frame:
        out = vr[device].infer({'pixel_values': tile})
        features = np.array(next(iter(out.values())), copy=True)
        assert features.shape == (1, 256, 896), features.shape
        visual_tiles.append(features)
    visual = np.concatenate(visual_tiles, axis=0)''',
    '''    visual_tiles = []
    _infer_total = 0.0
    _feature_copy_total = 0.0
    for tile in frame:
        _t = time.perf_counter()
        out = vr[device].infer({'pixel_values': tile})
        _infer_total += (time.perf_counter() - _t) * 1000
        _t = time.perf_counter()
        features = np.array(next(iter(out.values())), copy=True)
        _feature_copy_total += (time.perf_counter() - _t) * 1000
        assert features.shape == (1, 256, 896), features.shape
        visual_tiles.append(features)
    _t = time.perf_counter()
    visual = np.concatenate(visual_tiles, axis=0)
    _concat_ms = (time.perf_counter() - _t) * 1000
    print('PROFILE vision', device, 'tiles', len(frame),
          'infer_ms', round(_infer_total, 3),
          'result_copy_ms', round(_feature_copy_total, 3),
          'concat_ms', round(_concat_ms, 3), flush=True)'''
)
one_replace(
    '''    embout = tr.infer({'input':ids})
    emb = np.array(next(iter(embout.values())),copy=True)
    emb[positions] = visual.reshape(-1,896)
    prep_ms = (time.perf_counter()-start)*1000''',
    '''    _t = time.perf_counter()
    embout = tr.infer({'input':ids})
    _embed_infer_ms = (time.perf_counter() - _t) * 1000
    _t = time.perf_counter()
    emb = np.array(next(iter(embout.values())),copy=True)
    _embedding_copy_ms = (time.perf_counter() - _t) * 1000
    _t = time.perf_counter()
    emb[positions] = visual.reshape(-1,896)
    _visual_insert_ms = (time.perf_counter() - _t) * 1000
    prep_ms = (time.perf_counter()-start)*1000
    print('PROFILE bridge', 'embedding_request_ms', round(_embed_infer_ms,3),
          'embedding_copy_ms',round(_embedding_copy_ms,3),
          'visual_insert_ms',round(_visual_insert_ms,3),
          'total_prep_ms',round(prep_ms,3),flush=True)'''
)
dst = Path('src/pipeline_profile_overheads.py')
dst.write_text(s)
import ast
ast.parse(s)
print('Generated',dst,'; verified Python syntax; source unchanged.')
