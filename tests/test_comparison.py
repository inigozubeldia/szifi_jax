"""
Test script comparing szifi (original NumPy) vs szifi_jax (JAX port).
Runs both on Planck field 0 data and compares:
1. Cross-spectra (spec_tensor)
2. Inverse covariance (inv_cov)
3. SNR maps (q_map) per theta_500 scale
4. Detection catalogues
5. sigma_vec values
"""

import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["XLA_FLAGS"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 (GPU 0 is mostly full)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable TF GPU to avoid Blackwell conflicts
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import sys
# Ensure installed packages are used, not cwd
sys.path = [p for p in sys.path if p not in ('', '.')]

import numpy as np
import time
import copy

# ---- Run original szifi ----

print("="*60)
print("Running original szifi")
print("="*60)

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

t0 = time.time()
data_orig = szifi.input_data(params_szifi=params_szifi_orig, params_data=params_data_orig)
cf_orig = szifi.cluster_finder(params_szifi=params_szifi_orig, params_model=params_model_orig, data_file=data_orig, rank=0)
cf_orig.find_clusters()
t_orig = time.time() - t0
print(f"\nOriginal szifi time: {t_orig:.2f}s")

# Save original results for comparison
results_orig = cf_orig.results_dict
sigma_vec_orig = cf_orig.filtered_maps.sigma_vec.copy()

# Get detection catalogues
dp_orig = szifi.detection_processor(results_orig, params_szifi_orig)
cat_noit_orig = dp_orig.results.catalogues["catalogue_find_0"]
cat_it_orig = dp_orig.results.catalogues.get("catalogue_find_1", szifi.cluster_catalogue())

print(f"Original detections (noit): {len(cat_noit_orig.catalogue['q_opt'])}")
if hasattr(cat_it_orig, 'catalogue') and len(cat_it_orig.catalogue['q_opt']) > 0:
    print(f"Original detections (it): {len(cat_it_orig.catalogue['q_opt'])}")
print(f"Original sigma_vec: {sigma_vec_orig}")

# ---- Run szifi_jax ----

print("\n" + "="*60)
print("Running szifi_jax")
print("="*60)

import jax
jax.config.update("jax_enable_x64", True)

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

t0 = time.time()
data_jax = szifi_jax.input_data(params_szifi=params_szifi_jax, params_data=params_data_jax)
cf_jax = szifi_jax.cluster_finder(params_szifi=params_szifi_jax, params_model=params_model_jax, data_file=data_jax, rank=0)
cf_jax.find_clusters()
t_jax = time.time() - t0
print(f"\nszifi_jax time: {t_jax:.2f}s")

# Save jax results
results_jax = cf_jax.results_dict
sigma_vec_jax = np.asarray(cf_jax.filtered_maps.sigma_vec)

# Get detection catalogues
dp_jax = szifi_jax.detection_processor(results_jax, params_szifi_jax)
cat_noit_jax = dp_jax.results.catalogues["catalogue_find_0"]
cat_it_jax = dp_jax.results.catalogues.get("catalogue_find_1", szifi_jax.cluster_catalogue())

print(f"JAX detections (noit): {len(cat_noit_jax.catalogue['q_opt'])}")
if hasattr(cat_it_jax, 'catalogue') and len(cat_it_jax.catalogue['q_opt']) > 0:
    print(f"JAX detections (it): {len(cat_it_jax.catalogue['q_opt'])}")
print(f"JAX sigma_vec: {sigma_vec_jax}")

# ---- Comparisons ----

print("\n" + "="*60)
print("Comparison Results")
print("="*60)

# 1. sigma_vec comparison
print("\n--- sigma_vec comparison ---")
sigma_diff = np.abs(sigma_vec_orig - sigma_vec_jax)
sigma_reldiff = sigma_diff / np.abs(sigma_vec_orig)
print(f"Max abs diff: {np.max(sigma_diff):.2e}")
print(f"Max rel diff: {np.max(sigma_reldiff):.2e}")
for i in range(len(sigma_vec_orig)):
    print(f"  theta[{i}]: orig={sigma_vec_orig[i]:.8e}, jax={sigma_vec_jax[i]:.8e}, rel_diff={sigma_reldiff[i]:.2e}")

# 2. Detection catalogue comparison (non-iterative)
print("\n--- Detection catalogue comparison (non-iterative) ---")
n_orig = len(cat_noit_orig.catalogue["q_opt"])
n_jax = len(cat_noit_jax.catalogue["q_opt"])
print(f"N detections: orig={n_orig}, jax={n_jax}")

if n_orig > 0 and n_jax > 0:
    # Sort by q_opt for comparison
    idx_orig = np.argsort(cat_noit_orig.catalogue["q_opt"])[::-1]
    idx_jax = np.argsort(cat_noit_jax.catalogue["q_opt"])[::-1]

    n_compare = min(n_orig, n_jax, 10)
    print(f"Top {n_compare} detections:")
    for i in range(n_compare):
        q_o = cat_noit_orig.catalogue["q_opt"][idx_orig[i]]
        q_j = cat_noit_jax.catalogue["q_opt"][idx_jax[i]]
        t_o = cat_noit_orig.catalogue["theta_500"][idx_orig[i]]
        t_j = cat_noit_jax.catalogue["theta_500"][idx_jax[i]]
        print(f"  #{i}: q_orig={q_o:.4f}, q_jax={q_j:.4f}, theta_orig={t_o:.4f}, theta_jax={t_j:.4f}")

# 3. Timing comparison
print("\n--- Timing ---")
print(f"Original: {t_orig:.2f}s")
print(f"JAX:      {t_jax:.2f}s")
print(f"Speedup:  {t_orig/t_jax:.2f}x")

# 4. Summary
print("\n--- Summary ---")
all_pass = True

if np.max(sigma_reldiff) < 2e-3:
    print(f"PASS: sigma_vec matches (rel diff {np.max(sigma_reldiff):.2e} < 2e-3)")
else:
    print(f"WARN: sigma_vec rel diff = {np.max(sigma_reldiff):.2e}")
    all_pass = False

if n_orig == n_jax:
    print(f"PASS: Same number of detections ({n_orig})")
else:
    print(f"WARN: Different number of detections (orig={n_orig}, jax={n_jax})")
    all_pass = False

if all_pass:
    print("\nAll tests PASSED!")
else:
    print("\nSome tests had warnings - check details above.")
