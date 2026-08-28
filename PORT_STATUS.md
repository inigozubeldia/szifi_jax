# szifi_jax port status

Audit date: 2026-08-28. Reference: `szifi` @ `3f8b68b` ("Update README.md"),
working copy at `/scratch/scratch-izubeldia/szifi_project/szifi`.

The goal is a **carbon copy of `szifi` in JAX**: same public API, same numerics,
fully JIT-compiled and GPU-resident. This document records how far the current
code is from that, so the gaps are visible rather than assumed away.

## 1. Coverage

Module-level public symbols match `szifi` exactly, except `diffusive_inpaint`,
which is replaced by `diffusive_inpaint_jax`. Everything else is either identical
or an added private JIT helper.

| module | lines (szifi → jax) | `jnp.` | `np.` | `@jit` | state |
|---|---|---|---|---|---|
| `car.py`     | 156 → 156   | 0   | 18  | 0 | byte-identical, pure NumPy/pixell |
| `cat.py`     | 935 → 929   | 0   | 130 | 0 | comments stripped only; pure NumPy/sklearn |
| `config.py`  | 4 → 4       | 0   | 0   | 0 | byte-identical |
| `data.py`    | 15 → 15     | 0   | 0   | 0 | byte-identical |
| `paint.py`   | 64 → 64     | 0   | 7   | 0 | byte-identical |
| `params.py`  | 96 → 97     | 0   | 6   | 0 | **defaults changed** (§3.1) |
| `sphere.py`  | 269 → 267   | 0   | 36  | 0 | whitespace/comments only |
| `expt.py`    | 145 → 147   | 4   | 38  | 0 | arrays cast to `jnp` at the end of `__init__` |
| `sed.py`     | 204 → 204   | 11  | 2   | 0 | **quadrature changed** (§3.2) |
| `utils.py`   | 216 → 244   | 51  | 2   | 0 | ported; `invert_cov`/`div0` semantics differ (§3.6) |
| `model.py`   | 727 → 699   | 44  | 32  | 4 | **quadrature changed** (§3.2); Hankel path still SciPy |
| `spec.py`    | 454 → 757   | 59  | 77  | 2 | new "custom" JAX estimator; `interp_type` ignored (§3.3) |
| `maps.py`    | 1600 → 1439 | 153 | 156 | 3 | FFT/filter/inpaint on GPU; resample/EDT/interp still SciPy |
| `mmf.py`     | 1391 → 1664 | 80  | 139 | 4 | batched MMF on GPU (§2); several divergences (§3.4, §4) |

Totals: 402 `jnp.` vs **643 `np.`** call sites, 13 `@jit` decorators, 18 `vmap`
sites. Seven of fifteen modules contain no JAX at all. **"All jitted" is not yet
true**; the hot path (template FFT → matched filter → SNR maps → peak finding)
is, and that is where the wall-clock is.

## 2. What genuinely runs on the GPU

* `mmf.get_mmf_q_maps_batch` / `_get_mmf_q_maps_batch_jit` — all `theta_500`
  templates filtered and convolved in **one** `vmap`'d JIT call, replacing the
  per-theta Python loop over `scipy.signal.fftconvolve`. This is the main win.
* `mmf._get_maxima_mask_{2d,3d,3d_masked}_jit` — peak finding on device; only the
  small boolean peak mask returns to the host.
* `maps.diffusive_inpaint_jax` — `lax.fori_loop`, replacing a numba
  *object-mode* (`@jit(nopython=False)`, i.e. effectively pure-Python) double
  loop over masked pixels. Large speedup.
* `maps._edt_jfa_jax` — jump-flooding Euclidean distance transform, replacing
  `scipy.ndimage.distance_transform_edt`.
* `spec._cov_and_inv_aniso_gaussian_jit` — fused anisotropic-Gaussian covariance
  and inverse; Gaussian smoothing via `jax.scipy.signal.fftconvolve`.
* `model._get_p_cal_int_{arnaud,battaglia}_jit` + `vmap` — line-of-sight pressure
  integral for all radii at once.
* Template caching across noise iterations in `filter_maps` (see §4.1 for the
  caveat).

## 3. Numerical / behavioural divergences from `szifi`

These are places where `szifi_jax` will **not** reproduce `szifi`. None of them is
currently guarded by a check or a warning.

### 3.1 Default parameters differ (highest impact)

Verified at runtime by importing both packages:

| key | `szifi` | `szifi_jax` |
|---|---|---|
| `decouple_type` | `"master"` | `"fsky"` |
| `coupling_matrix_needed` | `True` | `False` |
| `powspec_implementation` | *(absent)* | `"custom"` |
| `snr_weigthing` / `snr_weighting` | `snr_weigthing` | `snr_weighting` |

`szifi`'s `mmf.py` never passes `implementation=` to `cross_spec.get_cross_spec`,
so it always uses the NaMaster/`pymaster` path with MASTER decoupling.
`szifi_jax`'s `mmf.py` passes `params_szifi["powspec_implementation"]`, default
`"custom"`. So **out of the box the two codes estimate the noise covariance with
a different estimator and a different decoupling**, and therefore produce
different filters and different `q` values. Any comparison that starts from each
package's own defaults — including the current `tests/test_comparison.py` and
`tests/benchmark.py` — is comparing two different analyses.

The `snr_weigthing` → `snr_weighting` rename is a genuine fix (`szifi`'s
`params.py` has the typo while its `mmf.py` reads the corrected spelling, so
stock `szifi` raises `KeyError` unless the user overrides it), but it does mean a
params dict written for one package is not valid for the other.

### 3.2 Quadrature rules changed

* `model.gnfw.get_p_cal_int`: `scipy.integrate.quad` (adaptive, `epsabs≈1.5e-10`)
  → fixed 256-node Gauss–Legendre with a cubic substitution `z = R t³`. The
  docstring claims `<1e-6` relative accuracy; **this has not been verified in
  this repo**.
* `model.gnfw.get_Y_aperture_numerical` and `sed.get_sed_bandpass`:
  `integrate.simpson` → `jnp.trapezoid`. Trapezoid is O(h²) where Simpson is
  O(h⁴); these are different answers, not a reformulation. The SED bandpass
  integral feeds `tsz_sed`, i.e. the MMF's spectral weights.

### 3.3 `spec.interp_cov` silently ignores `interp_type`

`interp_cov(..., interp_type=...)` still takes the argument but always does
nearest-neighbour via `_interp_cov_jit`. `szifi` used
`scipy.interpolate.interp1d(kind=interp_type, fill_value="extrapolate")`. The
project default is `"nearest"`, so the default path agrees, but `"linear"` /
`"cubic"` are silently downgraded rather than raising.

Related: extrapolation differs. SciPy extrapolates; `_interp_cov_jit` clamps to
the end bins.

### 3.4 Inpainting and blending use different algorithms

* `diffusive_inpaint` (`szifi`) updates masked pixels **sequentially in place**
  within each sweep — Gauss–Seidel — and divides by the *actual* neighbour count
  `c`. `diffusive_inpaint_jax` computes all 8 neighbours from the previous
  iterate — **Jacobi** — and always divides by 8 (edge-replicate padding). Both
  converge to the same harmonic solution, but at the default `n_inpaint=100` they
  are not the same map, and Jacobi converges roughly half as fast.
* `circular_cosine_blend_adaptive` (`szifi`) computes the distance transform
  **per hole, inside a padded bounding box**, so pixels of a *neighbouring* hole
  count as "outside". `circular_cosine_blend_adaptive_multi` computes **one
  global** EDT, in which neighbouring holes count as "inside". These differ
  wherever two masked regions are close. The new version also ignores the
  `min_radius` and `pad` arguments.
* `_edt_jfa_jax` is documented as an "exact match" with
  `scipy.ndimage.distance_transform_edt`. Jump flooding is an *approximate*
  algorithm and this one runs in `float32`; the claim is unverified.

### 3.5 Precision in the batched matched filter

`_get_mmf_q_maps_batch_jit` hard-casts the map and the filter to `float32` for
the convolution (`tmap_f32`, `filter_f32`), then returns `float64`. `szifi` uses
`tmap.dtype`, i.e. `params_szifi["map_dtype"]`, which defaults to `np.float32` —
so **the default agrees**. But if a user sets `map_dtype=np.float64` the
sequential fallback honours it and the batch path silently does not.

### 3.6 `utils` semantics

* `invert_cov`: `szifi` inverts every non-all-zero block and raises
  `LinAlgError` on a singular one. `szifi_jax` zeroes any block with
  `|det| == 0`. More robust, but different — and a well-conditioned block whose
  determinant underflows to exactly 0 in float64 is now zeroed instead of
  inverted.
* `div0`: `szifi` zeroes every non-finite result (including `nan` from `0/0` and
  from a `nan` numerator). `szifi_jax` returns `jnp.where(b != 0., a/b, 0.)`, so
  a `nan` numerator with a nonzero denominator now propagates.
* `get_cmmf_std` lost its unused `freqs` parameter. `szifi` passed 6 arguments to
  a 5-parameter function, so this call path was broken upstream; the port fixes
  it.
* `utils.get_mmf_y_map` convolved with `filter_fft` (a *Fourier-space*, complex
  array) in `szifi`; the port convolves with its real-space inverse FFT. Again a
  fix, but a change.

### 3.7 `mmf.make_detections` fall-through

`szifi` leaves `ret` unbound for an unrecognised `detection_method`, so it raises
`NameError`. The port `return ([],[],[])`. A typo in the config now yields "no
detections" instead of an error.

## 4. Bugs and risks specific to the port

### 4.1 `_template_cache` is not keyed on the field (CAR/SO correctness bug)

`cluster_finder.find_clusters` creates `self._template_cache` under
`if not hasattr(self, '_template_cache')`, inside the field loop. It is therefore
created on the first field and **reused for every subsequent field**, keyed only
on the `theta_500` index `j`.

For HEALPix tiles (Planck) every tile shares the same `pixel` object and `wcs` is
`None`, so all templates are genuinely identical and the cache is a pure win.
For `tile_type="car"` the template comes from `enmap.posmap()` on the *tile's own*
`wcs`, so templates differ from tile to tile — the cache will hand tile 0's
templates to every other tile. **This silently corrupts multi-tile CAR/SO runs**,
which is the Simons Observatory Cluster Finder Challenge use case.

Fix: key the cache on `(field_id, pix.nx, pix.ny, j)`, or clear it at the top of
each field iteration.

### 4.2 The batched path cannot handle a degraded power spectrum

`szifi`'s `get_tem_conv_fft` calls `maps.reshape_ell_matrix(tem_fft,
inv_cov.shape[:2])` to match a covariance computed on a degraded grid
(`powspec_lmax1d` / `powspec_new_shape`), then expands the filter back.
`_get_mmf_q_maps_batch_jit` does `jnp.einsum('ijk,ijkl->ijl', tem_fft_a,
inv_cov)` with no reshape, so it assumes `inv_cov` has the full map shape. With
`powspec_lmax1d` or `powspec_new_shape` set, the batch path will raise a shape
error or, if the shapes happen to be compatible, be wrong. There is no guard:
`can_batch` only checks `mmf_type`, `snr_weighting` and `save_snr_maps`.

### 4.3 `detection_method` is ignored on the batched path

`can_batch` does not consult `detection_method`, so the default
`"maxima_lomem"` is replaced by full 3D maxima. In practice these agree — the
only difference is a `>=` vs `>` tie-break along the `theta` axis, unreachable for
distinct floats — but the parameter is silently inert, and the memory saving that
`maxima_lomem` exists for is lost (the batch path materialises the full
`(nx, ny, n_theta)` `q` and `y` tensors on the GPU).

### 4.4 Anisotropic covariance skips power-spectrum setup

For `cov_type in ("anisotropic_gaussian", "anisotropic_boxcar")` the port skips
constructing `self.ps` and skips `get_cross_spec` entirely. The covariance itself
does not need them, so results are unaffected — but `self.ps` is then never set,
and `self.cspec.spec_tensor` stays `None`, so any downstream access raises.

### 4.5 Per-element GPU dispatch in the Hankel path

`model.gnfw.get_p_cal_int` returns `float(...)`, forcing a device→host sync, and
`get_y_map_hankel` / `get_t_map_convolved_hankel` call it through
`np.vectorize`. That is one JIT dispatch **plus** one host sync per radial
sample (~1024 with the default padding). `eval_type="hankel"` is now slower than
the NumPy original, not faster.

### 4.6 Unbounded module-level `_ell_cache`

`maps._ell_cache` is a module-level dict of GPU arrays keyed on
`(nx, ny, dx, dy)`, never evicted. Harmless for a single tile geometry;
it accumulates VRAM across a run with many distinct tile shapes.

### 4.7 Dead code

`spec._get_cross_spec_custom_jit`, `spec._gaussian_smooth_reflect_batch_jit`,
`spec._correct_for_inpainting_jit` are defined and never called.
`mmf.get_maxima_mask`, `mmf.get_maxima_mask_2d`, `mmf.get_maxima` and
`maps.circular_cosine_blend_adaptive` survive for API parity but are no longer
used by the port itself.

### 4.8 `maps.rmap.filter_lrange` return shape

`szifi`'s version discards the result of its own `reshape` and returns a flat
`(nx*ny,)` array (so `tmap.filter_lrange` cannot work). The port returns
`(nx, ny)`. A fix to dead code, but a divergence.

## 5. Still on the CPU in the hot path

* `maps.resample_fft`, `resample_fft_simple`, `degrade_map`,
  `reshape_ell_matrix` — pure NumPy, with a GPU→host→GPU round trip in
  `reshape_ell_matrix` whenever shapes do not already match. Called per template
  in `get_tem_conv_fft`.
* `spec.get_anisotropic_ps` — `scipy.ndimage` for the `boxcar` and `mixed_*`
  covariance types (only `anisotropic_gaussian` has a JAX path).
* `maps.get_noise_map` / `get_noise_tmap` — `np.random`, not `jax.random`, so
  simulation generation is neither jittable nor reproducible under a JAX key.
* `maps.smooth_inpainting_border`, `feather_mask_edge`, `smooth_mask_transition`,
  `repixelise_map`, `bin_radially`, `get_integrated_map_r`, `bin_harmonic_map` —
  SciPy.
* `scipy.ndimage.label` inside the blending routines (deliberate; the C
  implementation is ~2 ms).
* All of `cat.py` (`detection_processor`, merging, completeness) and `sphere.py`.

## 6. Validation gaps

* `tests/test_comparison.py` compares one Planck field and asserts only
  `max |Δσ|/σ < 2e-3` on `sigma_vec` and equal detection counts. It does **not**
  compare `q_opt`, positions, `theta_500`, the covariance, or the cross-spectra,
  despite its docstring listing them.
* Because both scripts start from each package's own `params_szifi_default`, the
  comparison runs `master`+`pymaster` against `fsky`+`custom` (§3.1). The
  configurations must be pinned identically on both sides before any agreement
  number means anything.
* Both scripts hard-code `CUDA_VISIBLE_DEVICES = "1"` instead of selecting the
  GPU with the most free VRAM at run time.
* No test covers CAR/SO tiles, multi-field runs, `mmf_type="spectrally_constrained"`,
  the iterative noise path (`max_it > 1`), or `powspec_lmax1d`.
* No recorded run of either script is checked in, so there is no evidence in this
  repository that the port has ever matched `szifi` on anything.

## 7. Suggested order of work

1. Pin every parameter explicitly and identically on both sides of
   `test_comparison.py`; re-run and record the numbers. Nothing else is
   meaningful until the two codes are configured the same way.
2. Fix `_template_cache` keying (§4.1) — silent wrong answers on CAR/SO.
3. Guard or fix the batched path for degraded power spectra (§4.2), and include
   `detection_method` in `can_batch` (§4.3).
4. Quantify the quadrature substitutions (§3.2) and the inpainting/blending
   algorithm changes (§3.4) against `szifi` rather than assuming they are
   negligible; either match `szifi` or document the tolerance.
5. Validate `_edt_jfa_jax` against `scipy.ndimage.distance_transform_edt` before
   relying on the "exact match" claim.
6. Extend coverage: CAR tiles, multi-field, spectrally-constrained MMF,
   `max_it > 1`.
7. Then push the remaining NumPy hot-path pieces (§5) onto the GPU.
