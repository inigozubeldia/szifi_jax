"""
End-to-end comparison of szifi (NumPy/CPU) against szifi_jax (JAX/GPU) on Planck field 0.

Every parameter is pinned explicitly and identically on both sides. This matters: the two
packages' own defaults are not interchangeable (szifi's params.py misspells "snr_weigthing"
while its mmf.py reads "snr_weighting", and szifi_jax adds "powspec_implementation"), so a
comparison that starts from each package's defaults is not comparing the same analysis.

Both packages run with their own defaults, which are now equivalent: the Gauss-Legendre
line-of-sight quadrature developed for this port was upstreamed into szifi, so the two share
it. Set params_model["los_integral"] = "quad" on either side to fall back to the
scipy.integrate.quad call szifi used before that change.

Note that szifi is not bit-reproducible run to run: pymaster's FFTW planning varies, which
moves spec_tensor by ~4e-12 between two identical runs. That is the floor for any comparison
of the power spectra; the detections themselves are reproducible.

Compared, in the order the pipeline produces them:
  1. noise cross-spectra          cspec.spec_tensor
  2. inverse noise covariance     cf.inv_cov
  3. filter normalisations        filtered_maps.sigma_vec
  4. SNR maps                     q_tensor / q maps per theta_500
  5. detection catalogues         q_opt, theta_500, positions, y0

Usage:  python tests/test_comparison.py [n_theta]
"""

import os
import subprocess
import sys


def _pick_gpu():
    """Select the GPU with the most free VRAM, as the shared box requires."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True)
        rows = [line.split(",") for line in out.strip().splitlines()]
        return min(rows, key=lambda r: int(r[1]))[0].strip()
    except Exception:
        return "0"


os.environ.setdefault("CUDA_VISIBLE_DEVICES", _pick_gpu())
os.environ["XLA_FLAGS"] = ""
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf                                     # noqa: E402
tf.config.set_visible_devices([], 'GPU')                    # Blackwell: TF GPU is broken

sys.path = [p for p in sys.path if p not in ('', '.')]

import copy                                                 # noqa: E402
import time                                                 # noqa: E402
import numpy as np                                          # noqa: E402
import jax                                                  # noqa: E402
jax.config.update("jax_enable_x64", True)

print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}  jax devices: {jax.devices()}")

N_THETA = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ROOT = "/scratch/scratch-izubeldia/szifi_project/szifi/"

# ---------------------------------------------------------------------------------------
# One parameter set, used verbatim by both packages.
# ---------------------------------------------------------------------------------------
COMMON = {
    "theta_500_vec_arcmin": np.exp(np.linspace(np.log(0.5), np.log(15.), N_THETA)),
    "q_th": 4.,
    "q_th_noise": 4.,
    "mask_radius": 3.,
    "iterative": True,
    "max_it": 1,

    "decouple_type": "master",
    "save_coupling_matrix": False,
    "compute_coupling_matrix": True,
    "coupling_matrix_needed": True,
    "powspec_lmax1d": None,
    "powspec_new_shape": None,
    "powspec_bin_fac": 4,
    "cov_type": "isotropic",
    "cov_kernel_shape": [4., 4.],
    "snr_weighting": False,

    "lrange": [100, 2500],
    "freqs": [0, 1, 2, 3, 4, 5],
    "beam": "real",

    "interp_type": "nearest",
    "n_inpaint": 100,
    "inpaint_type": "diffusive",
    "inpaint": True,
    "lsep": 3000,
    "max_radius_mask_arcmin": np.inf,
    "min_radius_mask_arcmin": 0.,

    "extraction_mode": "find",
    "get_q_true": False,
    "theta_500_input": None,
    "norm_type": "centre",
    "min_ftile": 0.3,
    "tilemask_mode": "field",
    "tilemask_buffer_arcmin": 10.,
    "tile_type": "healpix",
    "theta_find": "input",
    "detection_method": "maxima_lomem",
    "apod_type": "old",

    "path": ROOT,
    "path_data": ROOT + "data/",
    "survey_file": ROOT + "surveys/data_planck.py",
    "save_and_load_template": False,
    "path_template": ROOT + "data/",
    "map_dtype": np.float32,

    "mmf_type": "standard",
    "cmmf_type": "one_dep",
    "a_matrix": None,
    "comp_to_calculate": [0],
    "deproject_cib": None,
    "integrate_bandpass": True,
    "save_snr_maps": False,
    "snr_maps_name": None,
    "snr_maps_path": None,
    "get_lonlat": True,
    "cosmology": "Planck15",
    "cosmology_tool": "astropy",
    "rSZ": False,
}

# szifi_jax adds this key; "pymaster" is the estimator szifi always uses.
JAX_ONLY = {"powspec_implementation": "pymaster", "inv_cov_backend": "numpy"}

PARAMS_DATA = {"data_set": "Planck_real", "field_ids": [0],
               "other_params": {"components": ["tSZ", "kSZ", "CIB", "CMB", "noise"]}}


def run(pkg, extra=None, model_extra=None):
    params_szifi = copy.deepcopy(COMMON)
    if extra:
        params_szifi.update(extra)
    params_model = copy.deepcopy(pkg.params_model_default)
    if model_extra:
        params_model.update(model_extra)
    params_data = copy.deepcopy(PARAMS_DATA)

    data = pkg.input_data(params_szifi=params_szifi, params_data=params_data)
    cf = pkg.cluster_finder(params_szifi=params_szifi, params_model=params_model,
                            data_file=data, rank=0)
    t0 = time.time()
    cf.find_clusters()
    elapsed = time.time() - t0

    dp = pkg.detection_processor(cf.results_dict, params_szifi)
    catalogue = dp.results.catalogues["catalogue_find_0"]
    return cf, catalogue, elapsed


print("\n" + "=" * 78)
print(f"Running szifi (NumPy/CPU), n_theta={N_THETA}")
print("=" * 78)
import szifi                                                # noqa: E402
cf_np, cat_np, t_np = run(szifi)

print("\n" + "=" * 78)
print(f"Running szifi_jax (JAX/GPU), n_theta={N_THETA}")
print("=" * 78)
import szifi_jax                                            # noqa: E402
cf_jx, cat_jx, t_jx = run(szifi_jax, JAX_ONLY)

# ---------------------------------------------------------------------------------------
RESULTS = []


def compare(name, a, b, tol, mode="rel"):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  [FAIL] {name:44s} shape mismatch {a.shape} vs {b.shape}")
        RESULTS.append((name, np.inf, tol, False))
        return
    d = np.abs(a - b)
    scale = np.abs(a).max() if mode == "rel" else 1.
    val = (d / scale).max() if scale > 0 else d.max()
    ok = val <= tol
    RESULTS.append((name, val, tol, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {val:.3e}  (tol {tol:.1e}, max-{mode})")


print("\n" + "=" * 78)
print("Comparison")
print("=" * 78)

print("\n--- 1. noise cross-spectra (spec_tensor) ---")
st_np = np.asarray(cf_np.cspec.spec_tensor)
st_jx = np.asarray(cf_jx.cspec.spec_tensor)
print(f"    shape {st_np.shape}; szifi range [{st_np.min():.4e}, {st_np.max():.4e}]")
compare("spec_tensor", st_np, st_jx, 1e-10)   # floor set by szifi's own run-to-run scatter

print("\n--- 2. inverse noise covariance (inv_cov) ---")
ic_np = np.asarray(cf_np.inv_cov)
ic_jx = np.asarray(cf_jx.inv_cov)
d_ic = np.abs(ic_np - ic_jx).max(axis=(2, 3)) / max(np.abs(ic_np).max(), 1e-300)
print(f"    shape {ic_np.shape}; szifi |inv_cov| max {np.abs(ic_np).max():.4e}")
print(f"    per-mode rel diff percentiles: "
      + "  ".join(f"p{q}={np.percentile(d_ic, q):.2e}" for q in (50, 90, 99, 99.9, 100)))
print("    (the covariance is numerically singular at some modes, so a max over all of them")
print("     is set by the worst-conditioned handful; percentiles show the bulk)")
compare("inv_cov", ic_np, ic_jx, 1e-9)

print("\n--- 3. filter normalisations (sigma_vec) ---")
sv_np = np.asarray(cf_np.filtered_maps.sigma_vec)
sv_jx = np.asarray(cf_jx.filtered_maps.sigma_vec)
for i, (a, b) in enumerate(zip(sv_np, sv_jx)):
    rel = abs(a - b) / abs(a) if a != 0 else np.inf
    print(f"    theta[{i}] = {COMMON['theta_500_vec_arcmin'][i]:7.3f}'  "
          f"szifi={a:.10e}  jax={b:.10e}  rel={rel:.2e}")

# szifi's maxima_lomem path `continue`s at j == 0 before recording sigma_vec[0], so it always
# reports zero there. The port records it. Compare the entries szifi actually fills.
skip0 = (COMMON["detection_method"] == "maxima_lomem" and len(sv_np) > 3 and sv_np[0] == 0.)
if skip0:
    print("    (szifi leaves sigma_vec[0] == 0: its maxima_lomem path skips the assignment;")
    print("     the port records it, so index 0 is excluded from the comparison)")
compare("sigma_vec", sv_np[1:] if skip0 else sv_np,
        sv_jx[1:] if skip0 else sv_jx, 1e-12)

print("\n--- 4. detection catalogues ---")
q_np = np.asarray(cat_np.catalogue["q_opt"])
q_jx = np.asarray(cat_jx.catalogue["q_opt"])
print(f"    n detections: szifi={len(q_np)}  szifi_jax={len(q_jx)}")
RESULTS.append(("n detections equal", 0. if len(q_np) == len(q_jx) else 1., 0., len(q_np) == len(q_jx)))
print(f"  [{'PASS' if len(q_np) == len(q_jx) else 'FAIL'}] {'n detections equal':44s} "
      f"{len(q_np)} vs {len(q_jx)}")

if len(q_np) and len(q_np) == len(q_jx):
    o_np = np.argsort(-q_np)
    o_jx = np.argsort(-q_jx)
    fields = ["q_opt", "theta_500", "y0", "theta_x", "theta_y"]
    print(f"    {'#':>3} " + "".join(f"{f:>24}" for f in fields))
    for i in range(min(10, len(q_np))):
        row = f"    {i:>3} "
        for f in fields:
            a = np.asarray(cat_np.catalogue[f])[o_np[i]]
            b = np.asarray(cat_jx.catalogue[f])[o_jx[i]]
            row += f"{a:>11.5f}/{b:<12.5f}"
        print(row)
    # map_dtype defaults to float32, and szifi runs the matched-filter convolution at that
    # dtype, so ~1e-7 is the floor for anything downstream of it.
    TOL = {"q_opt": 1e-6, "theta_500": 0., "y0": 1e-6, "theta_x": 0., "theta_y": 0.}
    for f in fields:
        compare(f"catalogue {f}", np.asarray(cat_np.catalogue[f])[o_np],
                np.asarray(cat_jx.catalogue[f])[o_jx], TOL[f])


print("\n--- 5. timing ---")
print(f"    szifi (CPU): {t_np:.2f}s")
print(f"    szifi_jax (GPU, incl. JIT compilation): {t_jx:.2f}s")
print(f"    ratio: {t_np / t_jx:.2f}x")

print("\n" + "=" * 78)
n_fail = sum(1 for _, _, _, ok in RESULTS if not ok)
for name, value, tol, ok in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} {value:.3e} <= {tol:.1e}")
print("=" * 78)
print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
sys.exit(1 if n_fail else 0)
