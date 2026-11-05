#!/usr/bin/env python3
"""Fresh-process OV vision compiler timing; one device/run.
Usage: python profile_fresh_process.py GPU|NPU /data/project-03/models/InternVL2-1B-int4-ov

A fresh process and unique OpenVINO CACHE_DIR do NOT guarantee a cold GPU/NPU
firmware/driver or global hardware cache. Report as fresh-cache-dir compile.
"""
import sys
import time
import tempfile
import pathlib
import platform

if len(sys.argv) != 3 or sys.argv[1] not in ('GPU','NPU'):
    raise SystemExit('Usage: python profile_fresh_process.py GPU|NPU MODEL_DIR')
device=sys.argv[1]
folder=pathlib.Path(sys.argv[2]);assert folder.is_dir()
t0=time.perf_counter()
import openvino as ov
import numpy as np
print('PROFILE device',device,'python',platform.python_version(),'ov',ov.__version__,flush=True)
print('PROFILE from_script_start_to_import_s',round(time.perf_counter()-t0,3),flush=True)
t_core=time.perf_counter()
core=ov.Core()
print('PROFILE core_creation_s',round(time.perf_counter()-t_core,3),flush=True)
t_read=time.perf_counter()
m=core.read_model(str(folder/'openvino_vision_embeddings_model.xml'))
m.reshape([1,3,448,448])
print('PROFILE read_and_static_reshape_s',round(time.perf_counter()-t_read,3),flush=True)
with tempfile.TemporaryDirectory(prefix='phase1_ov_unique_cache_') as cd:
    t_compile=time.perf_counter()
    # This attempts a distinct OpenVINO on-disk cache directory for each run.
    try:
        cm=core.compile_model(m,device,{'CACHE_DIR':cd})
        cache_note='explicit_unique_cache_dir'
    except Exception as e:
        print('PROFILE cache_config_error',repr(e),flush=True)
        cm=core.compile_model(m,device)
        cache_note='default_cache_policy_not_cold'
    compile_s=time.perf_counter()-t_compile
    print('PROFILE compiled_in_s',round(compile_s,3),'cache_policy',cache_note,'devices',cm.get_property('EXECUTION_DEVICES'),flush=True)
    req=cm.create_infer_request()
    x=np.zeros([1,3,448,448],dtype=np.float32)
    t_first=time.perf_counter()
    y=req.infer({'pixel_values':x})
    print('PROFILE first_infer_ms',round(1000*(time.perf_counter()-t_first),2),flush=True)
    t_warm=time.perf_counter()
    for _ in range(5): req.infer({'pixel_values':x})
    print('PROFILE warm_5_avg_ms',round(1000*(time.perf_counter()-t_warm)/5,2),flush=True)
print('PROFILE script_elapsed_s',round(time.perf_counter()-t0,3),flush=True)
