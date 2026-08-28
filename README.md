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

This is a **work in progress**, not yet a validated drop-in replacement for `szifi`.
See `PORT_STATUS.md` for a module-by-module audit of what is on the GPU, what is
still NumPy/SciPy, and the places where the port's numerics or defaults differ
from `szifi`. Read that file before using this code for science.

## Requirements

JAX with float64 enabled:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

plus `numpy`, `scipy`, `astropy`, `healpy`, `pixell`, `pymaster` (only for the
`pymaster` power-spectrum path), `scikit-learn`.

## Tests

`tests/test_comparison.py` runs `szifi` and `szifi_jax` on Planck field 0 and
compares `sigma_vec` and the detection catalogues. `tests/benchmark.py` times
both. Note that both currently start from each package's *own* defaults, which
are not identical — see `PORT_STATUS.md`.
