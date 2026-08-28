# szifi_jax

A JAX/GPU port of [szifi](https://github.com/inigozubeldia/szifi), the multi-frequency
matched-filter (MMF) cluster finder used for the Planck SZiFi catalogues and the
Simons Observatory Cluster Finder Challenge.

The module layout mirrors `szifi` one-to-one:

| module | role |
|---|---|
| `mmf.py`    | MMF core: `cluster_finder`, `filter_maps`, `cmmf_precomputation` |
| `maps.py`   | pixelisation, FFTs, masks, inpainting |
| `spec.py`   | cross-spectra and noise covariance estimation |
| `model.py`  | GNFW / Arnaud pressure profile and y-templates |
| `sed.py`    | tSZ / CIB SEDs |
| `cat.py`    | catalogues, `detection_processor` |
| `expt.py`   | experiment definitions |
| `sphere.py`, `car.py` | HEALPix / CAR tiling |
| `utils.py`, `params.py`, `paint.py`, `data.py`, `config.py` | support |

The public API is identical to `szifi`:

```python
data = szifi_jax.input_data(params_szifi=..., params_data=...)
cf   = szifi_jax.cluster_finder(params_szifi=..., params_model=..., data_file=data, rank=0)
cf.find_clusters()
results = szifi_jax.detection_processor(cf.results_dict, params_szifi)
```

## Status

`szifi_jax` reproduces `szifi` numerically. On Planck field 0, with both packages on their
own defaults and every parameter pinned identically:

| quantity | agreement |
|---|---|
| noise cross-spectra `spec_tensor` | 9.7e-17 |
| filter normalisations `sigma_vec` | 3.3e-15 |
| detected `q_opt` | 3.8e-8 |
| detected `theta_500`, positions | exact |

3.8e-8 on `q_opt` is the float32 floor: `map_dtype` defaults to `np.float32` and `szifi` runs
the matched-filter convolution at that dtype.

Speed, same field: 3.9 s against `szifi`'s 320.6 s. Part of that came from fixes upstreamed
into `szifi` itself, which is now 167.3 s; see `PORT_STATUS.md` for the breakdown, for what
is still on the CPU and why, and for the places where the port deliberately differs.

Not yet exercised: CAR / Simons Observatory tiles, multi-field runs, the
spectrally-constrained MMF, `max_it > 1`, and `powspec_lmax1d`. Read `PORT_STATUS.md` §9
before using this for science.

## Requirements

JAX with float64 enabled:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

plus `numpy`, `scipy`, `astropy`, `healpy`, `pixell`, `pymaster` (only for the
`pymaster` power-spectrum path), `scikit-learn`.

## Tests

**Run these whenever you compare `szifi_jax` against `szifi`, and whenever you change either
of them.** Getting the port to agree with `szifi` turned on four separate bugs, none of which
showed up as a crash or an obviously wrong map — they moved `q_opt` at the 1e-3 level and
nothing else. Agreement here is not something to assume; it is something to measure, and
these are what measure it.

Every check prints the numbers it is judging, not just pass/fail, so the tolerances can be
audited rather than trusted.

| script | checks | what it covers |
|---|---|---|
| `tests/test_parity_units.py` | 20 | Each routine the port reimplements, against `szifi`, scipy, or a 40-digit mpmath reference: the line-of-sight integral, diffusive inpainting, the blend and its distance transform, `interp_cov` for every interpolation kind (including exact bin-edge ties), `_fftconvolve_same_sum`, `div0`, `invert_cov`. Plus guards on the fixed bugs — the CAR template-cache key, `make_detections` raising on an unknown method, the `snr_weigthing` alias, and a key-by-key check that the defaults match `szifi`'s. |
| `tests/test_comparison.py` | 9 | End to end on Planck field 0, all parameters pinned identically on both sides: `spec_tensor`, `inv_cov`, `sigma_vec`, and the catalogue (`q_opt`, `theta_500`, `y0`, positions, detection count). |
| `tests/test_coupling_matrix.py` | 6 | The optional GPU mode-coupling matrix against NaMaster: normalisation, symmetry, and agreement where NaMaster's own asymmetry vanishes. |

Start with `test_parity_units.py`: it localises a disagreement to a single function instead
of a whole SNR map, and it runs in seconds. `test_comparison.py` is the one that decides
whether the two codes actually agree, and takes a few minutes.

```bash
python tests/test_parity_units.py      # fast, routine by routine
python tests/test_comparison.py 8      # end to end, 8 theta_500 scales
python tests/test_coupling_matrix.py   # only if using coupling_matrix_backend="jax"
```

These are runnable scripts with exit codes, not a `pytest` suite, and they need the real
Planck field-0 data under `szifi_project/szifi/`, so they do not run in CI as they stand.
`tests/benchmark.py` and `tests/profile_breakdown.py` time and profile a run.

Two things worth knowing when you interpret the output. `szifi` is **not bit-reproducible run
to run** — pymaster's FFTW planning moves `spec_tensor` by ~4e-12 between two identical runs,
which is the floor for any power-spectrum comparison. And `szifi` leaves `sigma_vec[0] == 0`
(its `maxima_lomem` path skips the assignment), so `test_comparison.py` excludes that entry
and says so.
