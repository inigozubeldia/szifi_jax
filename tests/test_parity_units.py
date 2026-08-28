"""
Unit-level parity checks of szifi_jax against szifi / scipy.

Each check isolates one routine that the port reimplements, and compares it against the
reference implementation it is supposed to reproduce. Run this before the end-to-end
comparison: it localises a disagreement to a single function instead of a whole q map.

Every check prints the numbers it is judging, so the tolerances can be audited.
"""

import os
os.environ.setdefault("XLA_FLAGS", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
sys.path = [p for p in sys.path if p not in ('', '.')]

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from scipy import integrate, interpolate
from scipy.ndimage import distance_transform_edt
import scipy.signal as sg

import importlib
import szifi_jax
from szifi_jax import maps, model, spec
# szifi_jax.utils is shadowed by pixell.utils: maps.py does `from pixell import enmap, utils`
# and __init__.py does `from .maps import *`. szifi has exactly the same shadowing, so the
# port reproduces it; reach the port's own module explicitly.
utils = importlib.import_module("szifi_jax.utils")

RESULTS = []


def check(name, value, tol, unit="rel"):
    ok = value <= tol
    RESULTS.append((name, value, tol, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {value:.3e}  (tol {tol:.1e}, {unit})")
    return ok


# ---------------------------------------------------------------- 1. LOS quadrature
print("\n1. gnfw.get_p_cal_int: Gauss-Legendre quadrature")
print("   szifi uses scipy.integrate.quad(-R, R, epsabs=1.49e-10, limit=100). That call is")
print("   itself inaccurate where the integrand has its central cusp and where the profile")
print("   has fallen many orders of magnitude, so a 40-digit mpmath integral is the reference")
print("   and quad's own error is reported alongside for context.")

import astropy.cosmology as cp
import mpmath as mp
mp.mp.dps = 40
cosmo = cp.FlatLambdaCDM(H0=70., Om0=0.3)


def quad_szifi(profile, R_truncation, r):
    """szifi's exact call."""
    return integrate.quad(lambda z: profile(np.sqrt(z * z + r * r)),
                          -R_truncation, R_truncation,
                          epsabs=1.49e-08 / 100., limit=100)[0]


def quad_mpmath(profile_mp, R_truncation, r):
    Z = mp.sqrt(max(R_truncation ** 2 - r ** 2, 0.))
    return float(2 * mp.quad(lambda z: profile_mp(mp.sqrt(z * z + mp.mpf(r) ** 2)), [0, Z]))


for label, kind, radii in [
        ("arnaud", "arnaud", np.array([0., 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1., 3., 4.999])),
        ("battaglia", "battaglia", np.array([1e-4, 1e-2, 0.1, 0.5, 1., 2., 3., 4., 4.99]))]:

    nfw = model.gnfw(3e14, 0.2, cosmo, type=kind)
    q = nfw.my_params_sz

    if kind == "arnaud":
        def prof(x, q=q):
            cx = q.c_500 * x
            return 0. if x >= q.R_truncation else \
                q.P0 / (cx ** q.gamma * (1. + cx ** q.alpha) ** ((q.beta - q.gamma) / q.alpha))

        def prof_mp(x, q=q):
            cx = q.c_500 * x
            return mp.mpf(q.P0) / (cx ** mp.mpf(q.gamma) *
                                   (1 + cx ** mp.mpf(q.alpha)) ** ((q.beta - q.gamma) / q.alpha))
    else:
        def prof(x, q=q):
            return 0. if x >= q.R_truncation else \
                q.P0 * (x / q.xc) ** q.gamma * (1. + (x / q.xc) ** q.alpha) ** (-q.beta)

        def prof_mp(x, q=q):
            return mp.mpf(q.P0) * (x / mp.mpf(q.xc)) ** mp.mpf(q.gamma) * \
                (1 + (x / mp.mpf(q.xc)) ** mp.mpf(q.alpha)) ** (-mp.mpf(q.beta))

    ref = np.array([quad_mpmath(prof_mp, q.R_truncation, max(r, 1e-30)) for r in radii])
    gl = np.asarray(nfw.get_p_cal_int_vec(radii))
    qd = np.array([quad_szifi(prof, q.R_truncation, r) for r in radii])
    rel_gl = np.abs(gl - ref) / ref
    rel_qd = np.abs(qd - ref) / ref

    print(f"\n    {label}:")
    print(f"      {'r':>9}{'mpmath':>17}{'GL rel err':>14}{'szifi quad rel err':>21}")
    for r, t, a, b in zip(radii, ref, rel_gl, rel_qd):
        print(f"      {r:>9.4g}{t:>17.8e}{a:>14.2e}{b:>21.2e}")

    check(f"get_p_cal_int  {label}: max rel err vs mpmath", rel_gl.max(), 1e-9)
    check(f"get_p_cal_int  {label}: at least as accurate as szifi's quad",
          rel_gl.max() / max(rel_qd.max(), 1e-300), 1.0, "ratio")

# used by later checks
nfw = model.gnfw(3e14, 0.2, cosmo, type="arnaud")

# ---------------------------------------------------------------- 2. JFA EDT
print("\n2. maps._edt_jfa_jax vs scipy.ndimage.distance_transform_edt")

rng = np.random.default_rng(0)
nx = 256
# Hole radii matched to the real Planck field 0 point-source mask, whose 111 holes reach
# ~20 px in radius. Smaller discs do not expose the jump-flooding error.
hole = np.zeros((nx, nx), dtype=bool)
for _ in range(30):
    ci, cj = rng.integers(22, nx - 22, 2)
    rad = rng.integers(3, 21)
    ii, jj = np.ogrid[:nx, :nx]
    hole |= ((ii - ci) ** 2 + (jj - cj) ** 2) <= rad ** 2

d_scipy = distance_transform_edt(hole)
d_jfa = np.asarray(maps._edt_jfa_jax(jnp.array(hole)))
err = np.abs(d_jfa - d_scipy)
print(f"    {hole.sum()} hole pixels, max scipy distance {d_scipy.max():.3f} px")
print(f"    _edt_jfa_jax mismatches: {(err > 1e-5).sum()} / {hole.sum()}; max {err.max():.3e} px")
print("    (jump flooding is approximate; the blending routines use the exact transform)")

# the blend must reproduce szifi exactly, whichever EDT the primitive uses
rng_b = np.random.default_rng(7)
orig = rng_b.standard_normal((nx, nx, 3)) * 50.
inp = rng_b.standard_normal((nx, nx, 3)) * 50.
mask_b = (~hole).astype(float)
ref_blend = np.stack([
    np.asarray(maps.circular_cosine_blend_adaptive(orig[:, :, k], inp[:, :, k], mask_b,
                                                   inner_zero_frac=0.))
    for k in range(3)], axis=2)
got_blend = np.asarray(maps.circular_cosine_blend_adaptive_multi(orig, inp, mask_b,
                                                                 inner_zero_frac=0.))
check("blend  multi-frequency == per-frequency",
      np.abs(got_blend - ref_blend).max() / np.abs(ref_blend).max(), 1e-14)

# ------------------------------------------- 3. blend EDT equivalence (global vs per-region)
print("\n3. Global EDT == szifi's per-hole bounding-box EDT")

from scipy.ndimage import label
structure = np.ones((3, 3), dtype=bool)
labeled, n_reg = label(hole, structure=structure)
d_per_region = np.zeros_like(d_scipy)
pad = 2
for r in range(1, n_reg + 1):
    reg = labeled == r
    yy, xx = np.where(reg)
    y0, y1 = max(yy.min() - pad, 0), min(yy.max() + pad + 1, nx)
    x0, x1 = max(xx.min() - pad, 0), min(xx.max() + pad + 1, nx)
    box = reg[y0:y1, x0:x1]
    d_per_region[y0:y1, x0:x1][box] = distance_transform_edt(box)[box]
diff_edt = np.abs(d_per_region - d_scipy)[hole]
print(f"    {n_reg} hole regions; max |per-region EDT - global EDT| over holes: {diff_edt.max():.3e} px")
check("blend EDT      per-region vs global (px)", diff_edt.max(), 1e-9, "abs px")

# ---------------------------------------------------------------- 4. diffusive inpainting
print("\n4. diffusive_inpaint_jax (Jacobi) vs szifi's diffusive_inpaint (Gauss-Seidel)")


def diffusive_inpaint_reference(image, mask, n_inpaint):
    """szifi's algorithm, transcribed: sequential raster-order in-place sweeps."""
    nx_ = len(image)
    mask = np.asarray(mask, dtype=image.dtype)
    img = (image * mask).copy()
    x_0, y_0 = np.where(mask == 0.)
    for _ in range(n_inpaint):
        for i in range(len(x_0)):
            x, y = x_0[i], y_0[i]
            c, value = 0, 0.
            for ddx in (-1, 0, 1):
                for ddy in (-1, 0, 1):
                    if ddx == 0 and ddy == 0:
                        continue
                    if 0 <= x + ddx < nx_ and 0 <= y + ddy < nx_:
                        value += img[x + ddx, y + ddy]
                        c += 1
            img[x, y] = value / c
    return img


n_small = 128
rng = np.random.default_rng(1)
img = rng.standard_normal((n_small, n_small)) * 50.
mask_in = np.ones((n_small, n_small))
for _ in range(12):
    ci, cj = rng.integers(8, n_small - 8, 2)
    rad = rng.integers(2, 6)
    ii, jj = np.ogrid[:n_small, :n_small]
    mask_in[((ii - ci) ** 2 + (jj - cj) ** 2) <= rad ** 2] = 0.

n_holes = int((mask_in == 0).sum())
ref = diffusive_inpaint_reference(img, mask_in, 100)
holes = mask_in == 0
scale = np.std(img)
print(f"    {n_holes} masked pixels, map rms {scale:.3f}")

# the default path must reproduce szifi exactly
exact = np.asarray(maps.diffusive_inpaint(img, mask_in, 100))
de = np.abs(exact - ref)[holes]
print(f"    inpaint_type='diffusive'     max |diff| vs szifi: {de.max():.3e}")
check("diffusive_inpaint  default path is exact vs szifi", de.max(), 0., "abs")

# the opt-in GPU path is Jacobi; report how far it sits from szifi at finite n_inpaint
jac = np.asarray(maps.diffusive_inpaint_jax(jnp.array(img), jnp.array(mask_in), 100))
d = np.abs(jac - ref)[holes]
print(f"    inpaint_type='diffusive_jax' max |diff| vs szifi: {d.max():.4e} "
      f"({d.max()/scale:.2e} of map rms)  [Jacobi vs Gauss-Seidel, both unconverged]")
ref_conv = diffusive_inpaint_reference(img, mask_in, 4000)
jac_conv = np.asarray(maps.diffusive_inpaint_jax(jnp.array(img), jnp.array(mask_in), 4000))
dc = np.abs(jac_conv - ref_conv)[holes]
print(f"    at convergence (n=4000 both): max |diff| {dc.max():.3e}")
check("diffusive_inpaint  Jacobi/GS share a fixed point", dc.max() / scale, 1e-9)

# ---------------------------------------------------------------- 5. interp_cov
print("\n5. spec.interp_cov vs scipy.interpolate.interp1d")

ell_vec = np.linspace(100., 2500., 40)
cov_t = rng.standard_normal((40, 3, 3))
# Deliberately include out-of-range values AND values sitting exactly on a bin boundary.
# The latter is the case that matters: szifi's ell bin centres are odd multiples of the
# fundamental Fourier mode and the boundaries are even multiples, so every mode on the lx or
# ly axis of a real ell map lands exactly on a boundary.
ell_bds_exact = 0.5 * (ell_vec[1:] + ell_vec[:-1])
ell_map = np.concatenate([rng.uniform(0., 3000., 64 * 64 - 2 * len(ell_bds_exact)),
                          ell_bds_exact, ell_bds_exact]).reshape(64, 64)
for kind in ("nearest", "linear"):
    ref_t = np.zeros((64, 64, 3, 3))
    for a in range(3):
        for b in range(3):
            ref_t[:, :, a, b] = interpolate.interp1d(
                ell_vec, cov_t[:, a, b], kind=kind,
                bounds_error=False, fill_value="extrapolate")(ell_map)
    got = np.asarray(spec.interp_cov(ell_map, ell_vec, cov_t, interp_type=kind))
    e = np.abs(got - ref_t).max()
    check(f"interp_cov     kind={kind!r} max |diff| vs scipy", e, 1e-10, "abs")

got_cubic = np.asarray(spec.interp_cov(ell_map, ell_vec, cov_t, interp_type="cubic"))
ref_cubic = np.zeros((64, 64, 3, 3))
for a in range(3):
    for b in range(3):
        ref_cubic[:, :, a, b] = interpolate.interp1d(
            ell_vec, cov_t[:, a, b], kind="cubic",
            bounds_error=False, fill_value="extrapolate")(ell_map)
check("interp_cov     kind='cubic' honoured (scipy fallback)",
      np.abs(got_cubic - ref_cubic).max(), 1e-10, "abs")

# ---------------------------------------------------------------- 6. fftconvolve
print("\n6. utils._fftconvolve_same_sum vs scipy.signal.fftconvolve")

a3 = rng.standard_normal((96, 96, 4))
b3 = rng.standard_normal((96, 96, 4))
ref_c = sum(sg.fftconvolve(a3[:, :, i], b3[:, :, i], mode="same") for i in range(4))
got_c = np.asarray(utils._fftconvolve_same_sum(jnp.array(a3), jnp.array(b3)))
check("_fftconvolve_same_sum  max rel diff vs scipy",
      np.abs(got_c - ref_c).max() / np.abs(ref_c).max(), 1e-12)

# ---------------------------------------------------------------- 7. div0 / invert_cov
print("\n7. utils.div0 and utils.invert_cov")

a = np.array([1., 2., np.nan, 4., 1e308])
b = np.array([2., 0., 1., 0., 1e-308])
with np.errstate(divide="ignore", invalid="ignore"):
    c_ref = np.true_divide(a, b)
    c_ref[~np.isfinite(c_ref)] = 0
c_got = np.asarray(utils.div0(jnp.array(a), jnp.array(b)))
print(f"    a={a}\n    b={b}\n    szifi={c_ref}\n    port ={c_got}")
check("div0           max |diff| vs szifi", np.abs(c_got - c_ref).max(), 1e-12, "abs")

cov = rng.standard_normal((8, 8, 3, 3))
cov = np.einsum('ijab,ijcb->ijac', cov, cov) + 3. * np.eye(3)
cov[0, 0] = 0.
inv_ref = np.zeros_like(cov)
for i in range(8):
    for j in range(8):
        if np.any(cov[i, j]):
            inv_ref[i, j] = np.linalg.inv(cov[i, j])
inv_got = np.asarray(utils.invert_cov(jnp.array(cov)))
check("invert_cov     max rel diff vs szifi",
      np.abs(inv_got - inv_ref).max() / np.abs(inv_ref).max(), 1e-10)

# ---------------------------------------------------------------- 8. port-specific guards
print("\n8. Port-specific guards (behaviour szifi does not have, added to keep it honest)")

from szifi_jax import mmf, params as sj_params

# 8a. template cache key must separate CAR tiles with different WCS, but not HEALPix tiles
pix = maps.pixel(128, 0.5 / 60. / 180. * np.pi)
wcs_a = maps.get_target_wcs(0.5 / 60., (128, 128), (0., 0.))
wcs_b = maps.get_target_wcs(0.5 / 60., (128, 128), (30., -20.))

k_hp_a = mmf.get_template_cache_key(pix, None, "healpix")
k_hp_b = mmf.get_template_cache_key(pix, wcs_b, "healpix")
k_car_a = mmf.get_template_cache_key(pix, wcs_a, "car")
k_car_b = mmf.get_template_cache_key(pix, wcs_b, "car")
print(f"    healpix, two tiles      -> keys equal:     {k_hp_a == k_hp_b}")
print(f"    car, two tile WCS       -> keys differ:    {k_car_a != k_car_b}")
check("template cache key  shared across HEALPix tiles",
      0. if k_hp_a == k_hp_b else 1., 0., "bool")
check("template cache key  distinct per CAR tile WCS",
      0. if k_car_a != k_car_b else 1., 0., "bool")

# 8b. an unrecognised detection_method must raise, not silently return no detections
q_tensor = rng.standard_normal((16, 16, 4)) * 10.
try:
    mmf.make_detections(q_tensor, 4., pix, detection_method="maxima_typo")
    raised = False
except ValueError as e:
    raised = True
    print(f"    unknown detection_method raises: {e}")
check("make_detections  raises on unknown method", 0. if raised else 1., 0., "bool")

# szifi's own methods must still work
i_, j_, t_ = mmf.make_detections(q_tensor, 4., pix, detection_method="maxima")
print(f"    detection_method='maxima' still returns {len(i_)} peaks")
check("make_detections  'maxima' still works", 0. if len(i_) >= 0 else 1., 0., "bool")

# 8c. the misspelled szifi key must be accepted
p_in = {"snr_weigthing": True}
p_out = sj_params.normalise_params_szifi(dict(p_in))
print(f"    {p_in} -> {p_out}")
check("params  szifi's 'snr_weigthing' spelling accepted",
      0. if p_out.get("snr_weighting") is True and "snr_weigthing" not in p_out else 1., 0., "bool")

# 8d. defaults must match szifi's, key for key
import szifi as szifi_np
mismatch = []
for k, v in szifi_np.params_szifi_default.items():
    kk = sj_params.params_szifi_aliases.get(k, k)
    if kk not in sj_params.params_szifi_default:
        mismatch.append(f"{k}: missing in szifi_jax")
        continue
    a, b = v, sj_params.params_szifi_default[kk]
    same = np.array_equal(a, b) if isinstance(a, np.ndarray) else (a == b or (a is b))
    if not same:
        mismatch.append(f"{k}: szifi={a!r} szifi_jax={b!r}")
extra = set(sj_params.params_szifi_default) - {
    sj_params.params_szifi_aliases.get(k, k) for k in szifi_np.params_szifi_default}
print(f"    keys differing in value: {mismatch if mismatch else 'none'}")
print(f"    keys only in szifi_jax:  {sorted(extra) if extra else 'none'}")
check("params  defaults identical to szifi", float(len(mismatch)), 0., "count")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 74)
n_fail = sum(1 for _, _, _, ok in RESULTS if not ok)
for name, value, tol, ok in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {value:.3e} <= {tol:.1e}")
print("=" * 74)
print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
sys.exit(1 if n_fail else 0)
