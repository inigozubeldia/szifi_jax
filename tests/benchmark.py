"""
Benchmark: szifi (NumPy/CPU) vs szifi_jax (JAX/GPU)
Runs warmup for JIT compilation, then times multiple runs.
Data is reloaded for each run since find_clusters() consumes the data object.
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

# =============================================
# 1. Benchmark original szifi (CPU)
# =============================================

print("=" * 60)
print("Benchmarking original szifi (NumPy/CPU)")
print("=" * 60)

import szifi

params_szifi_orig = copy.deepcopy(szifi.params_szifi_default)
params_data_orig = copy.deepcopy(szifi.params_data_default)
params_model_orig = copy.deepcopy(szifi.params_model_default)

params_szifi_orig["path"] = "/scratch/scratch-izubeldia/szifi_project/szifi/"
params_szifi_orig["path_data"] = "/scratch/scratch-izubeldia/szifi_project/szifi/data/"
params_szifi_orig["survey_file"] = "/scratch/scratch-izubeldia/szifi_project/szifi/surveys/data_planck.py"
params_szifi_orig["save_and_load_template"] = False
params_szifi_orig["snr_weighting"] = False
params_data_orig["field_ids"] = [0]

n_runs_orig = 3
times_orig = []
for run in range(n_runs_orig):
    data_orig = szifi.input_data(params_szifi=params_szifi_orig, params_data=params_data_orig)
    cf = szifi.cluster_finder(
        params_szifi=params_szifi_orig,
        params_model=params_model_orig,
        data_file=data_orig, rank=0
    )
    t0 = time.time()
    cf.find_clusters()
    t_run = time.time() - t0
    times_orig.append(t_run)
    print(f"  Run {run+1}/{n_runs_orig}: {t_run:.2f}s")

print(f"\nOriginal szifi (CPU): {np.mean(times_orig):.2f}s +/- {np.std(times_orig):.2f}s")
print(f"  Best: {np.min(times_orig):.2f}s")

# =============================================
# 2. Benchmark szifi_jax (GPU)
# =============================================

print("\n" + "=" * 60)
print("Benchmarking szifi_jax (JAX/GPU)")
print("=" * 60)

import jax
jax.config.update("jax_enable_x64", True)

print(f"JAX devices: {jax.devices()}")

import szifi_jax

params_szifi_jax = copy.deepcopy(szifi_jax.params_szifi_default)
params_data_jax = copy.deepcopy(szifi_jax.params_data_default)
params_model_jax = copy.deepcopy(szifi_jax.params_model_default)

params_szifi_jax["path"] = "/scratch/scratch-izubeldia/szifi_project/szifi/"
params_szifi_jax["path_data"] = "/scratch/scratch-izubeldia/szifi_project/szifi/data/"
params_szifi_jax["survey_file"] = "/scratch/scratch-izubeldia/szifi_project/szifi/surveys/data_planck.py"
params_szifi_jax["save_and_load_template"] = False
params_szifi_jax["snr_weighting"] = False
params_data_jax["field_ids"] = [0]

# Warmup run (JIT compilation)
print("\nWarmup run (JIT compilation)...")
data_jax = szifi_jax.input_data(params_szifi=params_szifi_jax, params_data=params_data_jax)
cf_warmup = szifi_jax.cluster_finder(
    params_szifi=params_szifi_jax,
    params_model=params_model_jax,
    data_file=data_jax, rank=0
)
t0 = time.time()
cf_warmup.find_clusters()
t_warmup = time.time() - t0
print(f"Warmup: {t_warmup:.2f}s")

# Timed runs
n_runs_jax = 3
times_jax = []
for run in range(n_runs_jax):
    data_jax = szifi_jax.input_data(params_szifi=params_szifi_jax, params_data=params_data_jax)
    cf = szifi_jax.cluster_finder(
        params_szifi=params_szifi_jax,
        params_model=params_model_jax,
        data_file=data_jax, rank=0
    )
    t0 = time.time()
    cf.find_clusters()
    t_run = time.time() - t0
    times_jax.append(t_run)
    print(f"  Run {run+1}/{n_runs_jax}: {t_run:.2f}s")

print(f"\nszifi_jax (GPU): {np.mean(times_jax):.2f}s +/- {np.std(times_jax):.2f}s")
print(f"  Best: {np.min(times_jax):.2f}s")

# =============================================
# 3. Summary
# =============================================

print("\n" + "=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)
print(f"szifi (NumPy/CPU):   {np.mean(times_orig):.2f}s +/- {np.std(times_orig):.2f}s  (best: {np.min(times_orig):.2f}s)")
print(f"szifi_jax (JAX/GPU): {np.mean(times_jax):.2f}s +/- {np.std(times_jax):.2f}s  (best: {np.min(times_jax):.2f}s)")
print(f"szifi_jax warmup:    {t_warmup:.2f}s (includes JIT compilation)")
print(f"Speedup (mean):      {np.mean(times_orig)/np.mean(times_jax):.2f}x")
print(f"Speedup (best):      {np.min(times_orig)/np.min(times_jax):.2f}x")
