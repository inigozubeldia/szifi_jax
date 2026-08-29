# szifi_jax port status

Reference: `szifi` @ `3f8b68b`, working copy at
`/scratch/scratch-izubeldia/szifi_project/szifi`. Last updated 2026-08-28.

The goal is a **carbon copy of `szifi` in JAX**: same public API, same numerics, GPU-resident
and JIT-compiled wherever that is meaningful. This document records where the port stands.

Every number quoted here comes from `tests/test_parity_units.py` (routine by routine, against
`szifi` / scipy / a 40-digit mpmath reference) or `tests/test_comparison.py` (end-to-end on
Planck field 0). Both print the values they judge, so the tolerances can be checked rather
than taken on trust.

## 1. Where the port stands

Both packages on their own defaults, the port reproduces szifi on Planck field 0 to:

| quantity | agreement |
|---|---|
| noise cross-spectra `spec_tensor` | 9.7e-17 |
| inverse noise covariance `inv_cov` | 1.2e-10 (p50 6.1e-13) |
| filter normalisations `sigma_vec` | **3.3e-15** |
| detected `q_opt` | 3.8e-8 |
| detected `y0` | 3.2e-8 |
| detected `theta_500`, `theta_x`, `theta_y` | **exact** |
| number of detections | equal |

(`tests/test_comparison.py`, 9/9 passing: MASTER decoupling, 8 `theta_500` scales, one
iterative noise step, all parameters pinned identically on both sides.) Note that szifi is not
bit-reproducible run to run — pymaster's FFTW planning moves `spec_tensor` by ~4e-12 between
two identical runs — so that is the floor for the power spectra, not the port's doing.

3.8e-8 on `q_opt` is the float32 floor: `map_dtype` defaults to `np.float32` and szifi runs
the matched-filter convolution at that dtype, so nothing downstream of it can do better than
~1e-7. `inv_cov`'s 1.0e-10 tail is set by the modes where the noise covariance is numerically
singular (`|inv_cov|` reaches 2.9e14); the bulk agrees to 6e-13.

### Speed

The port is **11.4x** faster than szifi on everything it ported: 2.7 s against 30.2 s for
Planck field 0 with 8 scales and one noise iteration, with the MASTER coupling matrix reused
from disk (`compute_coupling_matrix=False`, `save_coupling_matrix=True` — the workflow szifi
is designed around). Cold, before the JIT cache is warm, the port takes 5.5 s.

An earlier revision of this document reported 1.06x. That was an artefact of benchmarking with
`compute_coupling_matrix=True`, which forces NaMaster to rebuild the mode-coupling matrix on
every run. Profiling settled it:

| stage | time | share |
|---|---|---|
| `nmt.compute_coupling_matrix` (2 calls) | 277.5 s | **97.2%** |
| cross-spectra, 42 frequency pairs | 2.3 s | 0.8% |
| templates | 1.1 s | 0.4% |
| `np.linalg.inv` of the covariance | 0.9 s | 0.3% |
| inpainting | 0.8 s | 0.3% |
| **the entire batched MMF** | **0.71 s** | **0.25%** |

Two of those numbers are the story. The matched filter — the reason the port exists — is
0.25% of the runtime, so no amount of further GPU work on it matters. And the coupling matrix
was being computed **twice per field with identical inputs**: it depends only on `mask_ps`,
the pixelisation and the binning, none of which change with the noise iteration, but it was
constructed inside the iteration loop. Hoisting it out (done here and upstreamed to szifi)
is exactly neutral and takes both codes from ~320 s to ~170 s on its own.

Three optimisations came out of this, all exact and all applied to szifi as well:

* **The coupling matrix is built once per field**, not once per noise iteration. ~1.9x.
* **NaMaster fields are built once per frequency**, not once per frequency *pair*. The
  cross-spectrum loop asks for every ordered pair, so it was constructing 42 `NmtFieldFlat`
  objects for 6 channels instead of 6.
* **The line-of-sight pressure integral is vectorised** (§3.1), which is 40x faster at
  tabulating a profile than 1000 separate `scipy.integrate.quad` calls and takes szifi from
  43.9 s to 31.6 s in the no-coupling-matrix configuration.

### The mode-coupling matrix on the GPU

`coupling_matrix_backend="jax"` computes the flat-sky MASTER matrix on the GPU instead of
calling NaMaster, removing the last large cost. Measured on Planck field 0:

| configuration | wall | `q_opt` vs szifi |
|---|---|---|
| szifi, matrix loaded from disk | 31.4 s | — |
| szifi, matrix computed by NaMaster | 167.3 s | — |
| szifi_jax, matrix loaded from disk | 5.5 s | 3.8e-8 |
| **szifi_jax, matrix computed on the GPU every run** | **3.9 s** | **2.4e-7** |

The matrix itself goes from 138.0 s to 1.04 s (133x). End to end that is **43x** against a
cold szifi run — and it is *faster than loading a precomputed matrix from disk*, so the whole
save/reuse workflow becomes unnecessary.

**It is not bit-compatible with NaMaster, and it is off by default.** The defining double sum

    S_bb' = sum_{l in b} sum_{l' in b'} |W(l - l')|^2,   M_bb' = S_bb' / N_b

is manifestly symmetric in (b, b'). Ours is symmetric to 5.6e-17. NaMaster's is asymmetric by
**4.6e-4**, concentrated in the high-ell bins, and that asymmetry is the entire disagreement
between the two — they track each other as the bins narrow:

| bins (128²) | ours vs NaMaster | NaMaster's asymmetry |
|---|---|---|
| 8   | 1.5e-13 | 1.9e-15 |
| 16  | 2.6e-4  | 2.9e-4  |
| 32  | 1.1e-3  | 4.6e-4  |
| 64  | 4.0e-3  | 5.6e-4  |

Where the asymmetry vanishes the two agree to **1e-13**. In NaMaster's source the row bin
index is guarded (`if(ik1>=0)`) while the column index (`i_band_nocut`) is not, so Fourier
modes falling outside every bin — the grid corners beyond the last bin edge, ~20% of modes at
these settings — are excluded from rows but not from columns. That is consistent with
everything measured, but it has not been confirmed with NaMaster's authors, so treat the
attribution as a hypothesis and the numbers as the fact.

The normalisation is exact, not fitted: `1/(nx·ny)³`, verified by the matrix coming out as the
identity (6.7e-16) for a structureless mask.

In practice the 1e-3 difference in the matrix washes out to **2.4e-7** in `q_opt`, because it
lives in the weakly weighted high-ell off-diagonals. If that is acceptable for your analysis,
turn it on; `tests/test_coupling_matrix.py` reports all of the above so you can re-measure.

The other lever remains `powspec_implementation="custom"`, the all-JAX estimator in `spec.py`,
which skips NaMaster entirely but does **not** implement MASTER decoupling — a different
estimator, not a drop-in, and still unvalidated.

## 2. Coverage

Module-level public symbols match `szifi` exactly. The port adds `diffusive_inpaint_jax`,
`circular_cosine_blend_adaptive_multi`, `get_mmf_q_maps_batch`, and private JIT helpers.

| module | lines (szifi → jax) | state |
|---|---|---|
| `car.py`     | 156 → 156   | byte-identical, pure NumPy/pixell |
| `cat.py`     | 935 → 929   | comments stripped only; pure NumPy/sklearn |
| `config.py`  | 4 → 4       | byte-identical |
| `data.py`    | 15 → 15     | byte-identical |
| `paint.py`   | 64 → 64     | byte-identical |
| `params.py`  | 96 → 133    | defaults match szifi; adds four port-only options (§5) |
| `sphere.py`  | 269 → 267   | whitespace/comments only |
| `expt.py`    | 145 → 147   | arrays cast to `jnp` at the end of `__init__` |
| `sed.py`     | 204 → 205   | Simpson quadrature, as in szifi |
| `utils.py`   | 216 → 249   | `div0` / `invert_cov` verified against szifi |
| `model.py`   | 727 → 754   | GNFW profile and LOS integral on GPU |
| `spec.py`    | 454 → 726   | adds an optional all-JAX power-spectrum estimator |
| `maps.py`    | 1600 → 1520 | FFTs, filtering, blending on GPU |
| `mmf.py`     | 1391 → 1704 | batched MMF on GPU |

Seven of fifteen modules contain no JAX. That is deliberate: they are I/O (`data`, `config`),
catalogue bookkeeping over a handful of detections (`cat`), or thin healpy/pixell wrappers
(`car`, `sphere`, `paint`). "Every line jitted" is not the state and is not the goal; §6 lists
what stays on the CPU and why.

## 3. What was wrong, and what fixed it

The port initially disagreed with szifi at the 3e-3 level in `q_opt`. Four separate causes,
found by bisecting the pipeline stage by stage:

**(a) Diffusive inpainting was Jacobi, szifi's is Gauss-Seidel.** szifi sweeps masked pixels
sequentially in raster order, writing in place. The port used the Jacobi form, which is
GPU-parallelisable but converges about half as fast. With `n_inpaint=100` *neither* is
converged for the larger point-source holes — Planck field 0 has 111 holes reaching 20 px in
radius, and szifi's own answer still moves by 2e5 between 100 and 400 sweeps. The two
therefore disagreed by ~4e3 times the map rms. Fixed by implementing szifi's exact
Gauss-Seidel sweep in numba (`maps.diffusive_inpaint`) and making it the default; it costs
~1.1 s per tile against a ~300 s run. The GPU Jacobi version remains available as
`inpaint_type="diffusive_jax"`. Effect: `spec_tensor` 3.3e-3 → 8.4e-6.

**(b) The blend used an approximate distance transform.** `_edt_jfa_jax` is a jump-flooding
transform and was documented as an "exact match" for `scipy.ndimage.distance_transform_edt`.
It is not: on scattered discs of radius ≤ 8 px it agrees to 4e-7 px, but on the real Planck
point-source mask it is wrong for **17680 of 29658 hole pixels, by up to 0.36 px**. That moves
the cosine blend weights. Fixed by using scipy's exact transform in the blending routines
(~63 ms per tile); the JFA primitive is kept with an honest docstring. Effect: `spec_tensor`
8.4e-6 → **9.7e-17**.

**(c) The ell map was built on the device.** This is the subtle one. The power-spectrum bin
centres sit at odd multiples of the fundamental Fourier mode and their boundaries at even
multiples, so a large set of ell values — every mode on the lx or ly axis, among others —
falls **exactly** on a bin boundary. `interp_cov` snaps each mode to its nearest bin, so at
those modes the assignment is a tie, and the ~5e-12 difference between the CPU and GPU
evaluations of `sqrt(lx² + ly²)` broke the tie the other way. Each flip shifts the covariance
at that mode by a whole bin (~2e4 out of ~1e6). Fixed by building `get_lxly` / `get_ell` in
NumPy and moving the result to the device — it is computed once per geometry and cached, so
this costs nothing. Effect: `sigma_vec` 1.16e-4 → 2.7e-6.

**(d) The nearest-bin rule differed from scipy's.** The port compared `|v - left|` against
`|v - right|`; scipy precomputes the boundaries as `(x[1:] + x[:-1]) / 2` and searchsorts
against them with `side="left"`. The two agree only where the midpoint is exactly
representable. Given (c), that is not a corner case. Fixed by replicating scipy's rule
exactly. Effect: `sigma_vec` 2.7e-6 → **1.2e-14**, `inv_cov` 3.1e-2 → 3.4e-12.

Two hypotheses were tested and **rejected** along the way, and are recorded so they are not
re-investigated: the batched MMF kernel (batched and sequential paths agree to 2.5e-8 and
diverged from szifi identically), and the `np.linalg.inv` / `jnp.linalg.inv` backend
(switching it changed nothing). szifi itself is well conditioned here — perturbing its input
maps by 1 ULP moves `sigma_vec` by 3.4e-15 and `q_opt` by exactly zero — so the residual could
never have been dismissed as numerical noise.

### 3.1 The line-of-sight quadrature, now shared with szifi

szifi integrated the pressure profile over the full `[-R_truncation, R_truncation]` with
`scipy.integrate.quad`, without telling `quad` where the integrand's truncation edge or its
central cusp are. Measured against a 40-digit mpmath reference, that call returns **`inf` at
r=0**, is wrong by 3.7e-6 at r=1e-8 and 9.0e-5 at r=1e-6 (Arnaud, the central cusp — and
r→0 is exactly what sets the template normalisation via `get_y_norm("centre")`), and by up to
**21% at r=4.99** for Battaglia, where the profile has fallen 21 orders of magnitude and
`quad`'s absolute tolerance is meaningless.

The replacement integrates over the support only, with a `z = Z(r)·t⁷` substitution and 512
Gauss-Legendre nodes: **1.1e-12** (Arnaud) and **2.5e-14** (Battaglia) against mpmath. It is
also 40x faster, because it evaluates all radii at once instead of making 1000 adaptive `quad`
calls per template.

**This has been upstreamed into szifi**, so both codes now use it and their defaults agree.
The change is invisible in the detections — szifi's `q_opt`, `theta_500` and positions are
bit-identical before and after, because `q` is invariant to template normalisation — and moves
`sigma_vec` and `y0` by 3.7e-6, which is the error being corrected. Set
`params_model["los_integral"] = "quad"` on either side to fall back to the old behaviour.

## 4. Verified agreement, routine by routine

`tests/test_parity_units.py`, 20/20 passing:

| routine | reference | measured |
|---|---|---|
| `gnfw.get_p_cal_int`, Arnaud | 40-digit mpmath | 1.1e-12 |
| `gnfw.get_p_cal_int`, Battaglia | 40-digit mpmath | 2.5e-14 |
| `maps.diffusive_inpaint` (default) | szifi's Gauss-Seidel | **0.0, exact** |
| `maps.diffusive_inpaint_jax` | same fixed point as szifi | 1.4e-16 |
| `circular_cosine_blend_adaptive_multi` | per-frequency blend | 2.8e-14 |
| global vs per-hole EDT in the blend | szifi's per-region bounding box | **0.0, exactly equal** |
| `spec.interp_cov`, `nearest` (incl. exact ties) | `scipy.interpolate.interp1d` | **0.0, bit-identical** |
| `spec.interp_cov`, `linear` | `scipy.interpolate.interp1d` | 3.6e-15 |
| `spec.interp_cov`, `cubic` | `scipy.interpolate.interp1d` | 0.0 (scipy fallback) |
| `utils._fftconvolve_same_sum` | `scipy.signal.fftconvolve` | 5.4e-16 |
| `utils.div0` | szifi, incl. `nan` and overflow | **0.0** |
| `utils.invert_cov` | szifi | 1.7e-16 |
| `params_szifi_default` | szifi, key by key | **identical** |

szifi's per-hole bounding-box EDT and the port's single global EDT are provably the same
thing: hole regions are 8-connected components, so any raster path from one hole to another
must cross a non-hole pixel, which is therefore at least as close as the other hole. Measured
difference: exactly zero.

## 5. Port-only parameters

All four default to szifi's behaviour. `params.normalise_params_szifi` also accepts szifi's
misspelled `snr_weigthing` key, so a params dict written against either package works.

| key | default | alternative |
|---|---|---|
| `powspec_implementation` | `"pymaster"` (what szifi always uses) | `"custom"`, an all-JAX estimator, **not yet validated** |
| `inv_cov_backend` | `"numpy"` (LAPACK, as szifi) | `"jax"` (cuSOLVER, stays on device) |
| `mmf_batched` | `True` | `False`, filter each `theta_500` sequentially |
| `coupling_matrix_backend` | `"pymaster"` (bit-compatible with szifi) | `"jax"`, 133x faster, differs by ~1e-3 in the matrix and 2.4e-7 in `q_opt` |
| `params_model["los_integral"]` | `"gauss_legendre"` (now also szifi's default) | `"quad"`, szifi's pre-fix call (§3.1) |

`inpaint_type` gains `"diffusive_jax"` alongside szifi's `"diffusive"` and `"orphics"`.

## 6. What runs on the GPU, and what does not

On the GPU:

* `mmf.get_mmf_q_maps_batch` — all `theta_500` templates filtered and convolved in one
  `vmap`'d JIT call, replacing the per-theta Python loop over `scipy.signal.fftconvolve`.
* `mmf._get_maxima_mask_{2d,3d,3d_masked}_jit` — peak finding on device; only the small
  boolean peak mask returns to the host.
* `spec._cov_and_inv_aniso_gaussian_jit` — fused anisotropic-Gaussian covariance.
* `model._get_p_cal_int_{arnaud,battaglia}_jit` + `vmap` — the LOS integral for all radii at
  once. This also removed a device dispatch and host sync per radial sample in the Hankel
  path, which used to call the scalar entry point through `np.vectorize`.
* Template caching across noise iterations, keyed on tile geometry (and, for CAR, on the tile
  WCS — see §7).

Still on the CPU, deliberately:

* `maps.diffusive_inpaint` (numba) and `scipy.ndimage`'s EDT and `label` in the blend — see
  §3(a) and §3(b). Exactness is worth ~1.2 s per tile.
* `maps.get_lxly` / `get_ell` — §3(c). Cached, so the cost is one evaluation per geometry.
* `np.linalg.inv` for the noise covariance — 0.42 s for a 1024×1024 grid of 6×6 matrices.
* `pymaster`, for the default MASTER-decoupled power spectra.
* `maps.resample_fft`, `degrade_map`, `reshape_ell_matrix` — only exercised when
  `powspec_lmax1d` / `powspec_new_shape` is set, and they force a GPU→host→GPU round trip.
* `spec.get_anisotropic_ps` — `scipy.ndimage` for the `boxcar` and `mixed_*` covariance types.
* `maps.get_noise_map` / `get_noise_tmap` — `np.random`, not `jax.random`, so simulated noise
  is neither jittable nor reproducible under a JAX key.
* `maps.smooth_inpainting_border`, `feather_mask_edge`, `smooth_mask_transition`,
  `repixelise_map`, `bin_radially`, `get_integrated_map_r`, `bin_harmonic_map`.
* All of `cat.py` and `sphere.py`.

## 7. Bugs fixed relative to the original port

* **`_template_cache` was keyed only on the `theta_500` index**, and created under a
  `hasattr` guard inside the field loop, so tile 0's templates were reused for every
  subsequent tile. Harmless for HEALPix (all Planck tiles share a pixelisation) but silently
  wrong for CAR, where the template is built from each tile's own posmap — i.e. exactly the
  Simons Observatory case. Now keyed on `(tile_type, nx, ny, dx, dy, wcs)`.
* **The batched MMF path contracted the template FFT directly against `inv_cov`**, with no
  `reshape_ell_matrix`, so it would break or silently misbehave with a degraded power spectrum
  (`powspec_lmax1d` / `powspec_new_shape`). `can_batch` now requires `inv_cov` to be on the
  full map grid and falls back to the sequential path otherwise.
* **`detection_method` was ignored on the batched path.** Now part of `can_batch`.
* **`self.ps` and the cross-spectra were skipped for anisotropic covariance types**, leaving
  `self.ps` undefined for any downstream caller. Restored.
* **`make_detections` returned no detections for an unrecognised `detection_method`**
  (szifi raises `NameError`). Now raises `ValueError` naming the allowed values.
* **The batched convolution hard-cast to float32**, silently ignoring `map_dtype=np.float64`.
  Now follows `map_dtype`, as szifi does.
* **`spec.interp_cov` accepted `interp_type` and ignored it**, always doing nearest. Now
  `nearest` and `linear` have JAX kernels that reproduce scipy exactly, and any other kind
  falls back to scipy rather than being silently replaced.
* **Quadrature had been swapped for lower-order rules**: `integrate.simpson` → `jnp.trapezoid`
  in `sed.integrate_sed_bandpass` (which feeds `tsz_sed`, the MMF's spectral weights) and in
  `gnfw.get_Y_aperture_numerical`. Simpson restored.
* **`utils.div0` let a `nan` numerator and overflow-to-inf through**, where szifi zeroes every
  non-finite result.
* **`_ell_cache` was unbounded**, accumulating device arrays across tile geometries.
* **Dead code removed**: `spec._get_cross_spec_custom_jit`,
  `spec._gaussian_smooth_reflect_batch_jit`, `spec._correct_for_inpainting_jit`.

## 8. Deliberate fixes to upstream szifi bugs

The port does not reproduce these, because szifi is wrong:

* **`snr_weigthing`** — szifi's `params.py` defines this misspelling while its `mmf.py` reads
  `snr_weighting`, so stock szifi raises `KeyError` unless the caller overrides it.
* **`sigma_vec[0]`** — szifi's `maxima_lomem` path `continue`s at `j == 0` before recording
  it, so it always reports `sigma_vec[0] == 0`. The port records it.
* **`utils.get_cmmf_std`** — szifi passes 6 arguments to a 5-parameter function.
* **`utils.get_mmf_y_map`** — szifi convolves the map with `filter_fft`, a *Fourier-space*
  complex array. The port convolves with its real-space inverse FFT.
* **`maps.rmap.filter_lrange`** — szifi discards the result of its own `reshape` and returns a
  flat `(nx·ny,)` array, so `tmap.filter_lrange` cannot work.

`maps.get_map_convolved_fft`'s use of `pix.nx` for both axes (wrong for non-square tiles) is
reproduced as-is: changing it would alter results rather than fix a crash.

Not a port issue, but worth knowing: **`szifi_jax.utils` is shadowed by `pixell.utils`**,
because `maps.py` does `from pixell import enmap, utils` and `__init__.py` does
`from .maps import *`. szifi has exactly the same shadowing. Reach the port's module with
`importlib.import_module("szifi_jax.utils")`.

## 9. Coverage gaps

Not yet exercised by any test:

* CAR / Simons Observatory tiles, and multi-field runs. The template-cache fix (§7) is
  unit-tested but has not been run end-to-end on CAR data. **This is the biggest gap.**
* `mmf_type="spectrally_constrained"` (both `cmmf_type="one_dep"` and `"general"`). Note that
  `utils.invert_cov`, which that path uses, inverts on the GPU regardless of
  `inv_cov_backend`.
* The iterative noise path with `max_it > 1`.
* `powspec_lmax1d` / `powspec_new_shape`, including the sequential fallback they now trigger.
* `cov_type` other than `"isotropic"`.
* `powspec_implementation="custom"` against `"pymaster"` on MASTER. On FLAMINGO homog **fsky**
  it was measured and **fails** (`spec_tensor` rel ~100, thousands of spurious detections).

## 10. This machine (2026-08-29)

GPU1 = RTX PRO 6000 Blackwell (sm_120), JAX 0.10. `tests/test_parity_units.py` 20/20.

**Batched MMF.** `vmap` over `theta_500` of the per-mode 6×6 einsum dies with
`JaxRuntimeError INTERNAL`. Not memory: `n_theta=1` at 1024² works; `n_theta=2` fails from
`nx≥256` or `n_freq≥2`; FFTs vmap. The kernel now flattens to a 3D gemv
`(n_theta·n_pix, n_freq)` × `(n_theta·n_pix, n_freq, n_freq)` and batches FFTs with frequency
on a leading axis. Same contraction, supported on this GPU. Theta is still a
`lax.scan` (cannot vmap that contraction here).

**FLAMINGO homog iMMF** (fsky, Gaussian beam, 8 scales 1′–10′, unmasked 768 tiles).
CNC path: `nmt_jax` + unique-bin LAPACK then GPU scatter
(`invert_spec_interp_tiles`) + raster Gauss-Seidel (not Jacobi) + stacked GPU
FFT. Do **not** invert the 1024² grid with `jnp.linalg.inv` (moves `q_opt` at
~3e-3 on singular modes) or use Jacobi inpaint.

| check | result |
|---|---|
| tiles 0–1 | **10/10**, median sep **0**, max \|Δq\| **6e-6** |
| full sky | **N=2364/2364**, all positions matched, median sep **0**, N(q) bins identical |
| max \|Δq\| | **1.0e-3** on one cluster (tile 190); the rest ≲ 4e-5 |

**GPU tile batch** (L1_m9, after JIT; 16-tile batches unless noted). Sequential
was ~3.8 s/tile (NaMaster + per-mode LAPACK).

| `gpu_tile_batch` | warm s/tile | notes |
|---|---|---|
| 8 | 0.71 | underfills the GPU |
| **16** | **0.22–0.52** | fastest; homog quiet tiles ~0.22, L1_m9 ~0.50 |
| 24 | 0.92 | slower (host↔device scatter of a larger inv stack) |
| 32 | 0.50–0.66 | slower than 16 |
| 40 | — | OOM allocating 26 GiB |

Default `gpu_tile_batch` is **16**. `nmt_jax` matches NaMaster coupled cells
(rfft visit, ~1e-16 on Cl > 1e-20). The `"custom"` periodogram is not a fsky
drop-in (median Cl rel ~0.25).

**25 search scales** (log-spaced 0.5′–32′, vs the 8-scale 1′–10′ grid). Only
the MMF `scan` over θ and 3D peak-find scale with n_θ; spectra, unique-bin
inv, and inpaint do not. Predicted **2–2.5×**. Measured on L1_m9, 16 tiles,
warm: **3.6 s → 8.1 s (2.25×)**, 0.22 → 0.51 s/tile.

**L1 feedback prescriptions** (same 8-scale CNC path, `gpu_tile_batch=16`).
Jax catalogues vs NumPy `fullsky_splitA_immf_q5.npz`; every position matched
(median sep 0), N(q) bins identical:

| prescription | N | max \|Δq\| | wall |
|---|---|---|---|
| L1_m9 | **2509/2509** | 3.1e-5 | 6.5 min |
| fgas−8σ | **2125/2125** | 2.8e-5 | 8.9 min |
| M\*−1σ | **2634/2634** | 4.9e-3 (one cluster) | 9.9 min |
| LS8 | **1561/1561** | 2.8e-5 | 9.0 min |

## 11. Suggested next steps

1. Extend `tests/test_comparison.py` to CAR tiles and a multi-field run.
2. Exercise the spectrally-constrained MMF and `max_it > 1`; give `utils.invert_cov` the same
   backend switch as `spec.get_inv_cov`.
3. Decide whether `coupling_matrix_backend="jax"` should become the default. It is 43x faster
   end to end and costs 2.4e-7 in `q_opt`, but it is not bit-compatible with NaMaster, and the
   reason is a suspected asymmetry in NaMaster itself that would be worth raising upstream
   before relying on it. For **fsky**, use `nmt_jax` rather than `"custom"`.
4. Port `reshape_ell_matrix` and `resample_fft` to JAX and lift the `inv_cov` shape
   restriction on the batched MMF.
5. Move noise generation to `jax.random` so simulations are reproducible and jittable.
