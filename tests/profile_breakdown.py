"""
Profile szifi_jax to identify where time is spent.
"""

import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["XLA_FLAGS"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import sys
sys.path = [p for p in sys.path if p not in ('', '.')]

import numpy as np
import time
import copy
import jax
jax.config.update("jax_enable_x64", True)

print(f"JAX devices: {jax.devices()}")

import szifi_jax
import szifi_jax.mmf as mmf_mod
import szifi_jax.maps as maps_mod
import szifi_jax.spec as spec_mod
import szifi_jax.model as model_mod

# Instrument key functions
_timings = {}

def make_timer(name, orig_fn):
    def timed_fn(*args, **kwargs):
        t0 = time.time()
        result = orig_fn(*args, **kwargs)
        dt = time.time() - t0
        if name not in _timings:
            _timings[name] = []
        _timings[name].append(dt)
        return result
    return timed_fn

# Patch key functions to measure timing
# MMF functions
orig_get_mmf_q_map = mmf_mod.get_mmf_q_map
mmf_mod.get_mmf_q_map = make_timer("get_mmf_q_map", orig_get_mmf_q_map)

# Power spectrum
orig_get_cross_spec = spec_mod.cross_spec.get_cross_spec
spec_mod.cross_spec.get_cross_spec = make_timer("get_cross_spec", orig_get_cross_spec)

orig_get_inv_cov = spec_mod.cross_spec.get_inv_cov
spec_mod.cross_spec.get_inv_cov = make_timer("get_inv_cov", orig_get_inv_cov)

# Inpainting
orig_inpaint_freq = maps_mod.inpaint_freq
maps_mod.inpaint_freq = make_timer("inpaint_freq", orig_inpaint_freq)

# Model template
orig_get_t_map_convolved = model_mod.gnfw.get_t_map_convolved
model_mod.gnfw.get_t_map_convolved = make_timer("get_t_map_convolved", orig_get_t_map_convolved)

# Filter tmap
orig_filter_tmap = maps_mod.filter_tmap
maps_mod.filter_tmap = make_timer("filter_tmap", orig_filter_tmap)

# Diffusive inpaint
orig_diffusive_inpaint_jax = maps_mod.diffusive_inpaint_jax
maps_mod.diffusive_inpaint_jax = make_timer("diffusive_inpaint_jax", orig_diffusive_inpaint_jax)

# Detection
orig_make_detections = mmf_mod.make_detections
mmf_mod.make_detections = make_timer("make_detections", orig_make_detections)

orig_make_detections2 = mmf_mod.make_detections2
mmf_mod.make_detections2 = make_timer("make_detections2", orig_make_detections2)

# Setup
params_szifi = copy.deepcopy(szifi_jax.params_szifi_default)
params_data = copy.deepcopy(szifi_jax.params_data_default)
params_model = copy.deepcopy(szifi_jax.params_model_default)

params_szifi["path"] = "/scratch/scratch-izubeldia/szifi_project/szifi/"
params_szifi["path_data"] = "/scratch/scratch-izubeldia/szifi_project/szifi/data/"
params_szifi["survey_file"] = "/scratch/scratch-izubeldia/szifi_project/szifi/surveys/data_planck.py"
params_szifi["save_and_load_template"] = False
params_szifi["snr_weighting"] = False
params_data["field_ids"] = [0]

# Run once for warmup
print("Warmup run...")
data = szifi_jax.input_data(params_szifi=params_szifi, params_data=params_data)
cf = szifi_jax.cluster_finder(params_szifi=params_szifi, params_model=params_model, data_file=data, rank=0)
cf.find_clusters()
_timings.clear()

# Profiled run
print("\nProfiled run...")
data = szifi_jax.input_data(params_szifi=params_szifi, params_data=params_data)
cf = szifi_jax.cluster_finder(params_szifi=params_szifi, params_model=params_model, data_file=data, rank=0)

t0 = time.time()
cf.find_clusters()
t_total = time.time() - t0

print(f"\nTotal find_clusters: {t_total:.2f}s")
print("\n--- Per-function breakdown ---")
for name, times in sorted(_timings.items(), key=lambda x: -sum(x[1])):
    total = sum(times)
    n_calls = len(times)
    avg = total / n_calls
    print(f"  {name:30s}: {total:8.2f}s  ({n_calls:3d} calls, {avg:.3f}s avg, {100*total/t_total:.1f}%)")

accounted = sum(sum(t) for t in _timings.values())
print(f"\n  {'Accounted':30s}: {accounted:8.2f}s  ({100*accounted/t_total:.1f}%)")
print(f"  {'Unaccounted':30s}: {t_total-accounted:8.2f}s  ({100*(t_total-accounted)/t_total:.1f}%)")
