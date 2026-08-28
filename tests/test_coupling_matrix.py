"""
The GPU flat-sky MASTER coupling matrix (spec.get_coupling_matrix_jax) against NaMaster.

This is NOT a parity test: the two do not agree to machine precision, and the reason is on
NaMaster's side. The defining double sum

    S_bb' = sum_{l in b} sum_{l' in b'} |W(l - l')|^2,   M_bb' = S_bb' / N_b

is manifestly symmetric in (b, b'), so N_b * M_bb' must equal N_b' * M_b'b. NaMaster's matrix
fails that by ~5e-4, concentrated in the high-ell bins. In its source the row bin index is
guarded (`if(ik1>=0)`) while the column index (`i_band_nocut`) is not, so Fourier modes that
fall outside every bin -- about 20% of the grid at these settings, the corners beyond the last
bin edge -- are excluded from rows but not from columns.

This test therefore reports three things and asserts only the ones that are ours:
  * exactness in the cases where NaMaster's asymmetry vanishes (structureless mask, few bins);
  * that our matrix is symmetric, as the formula requires;
  * the size of NaMaster's asymmetry, alongside our disagreement with it, so the two can be
    compared. If they track each other, the disagreement is explained.
"""

import os
import subprocess
import sys

os.environ["XLA_FLAGS"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
try:
    _gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES",
                          min([g.split(",") for g in _gpus], key=lambda r: int(r[1]))[0].strip())
except Exception:
    pass

sys.path = [p for p in sys.path if p not in ('', '.')]

import time                                                  # noqa: E402
import numpy as np                                           # noqa: E402
import pymaster as nmt                                       # noqa: E402
import jax                                                   # noqa: E402
jax.config.update("jax_enable_x64", True)

from szifi_jax import maps, spec                              # noqa: E402

RESULTS = []


def check(name, value, tol):
    ok = value <= tol
    RESULTS.append((name, value, tol, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:56s} {value:.3e}  (tol {tol:.1e})")


DX = 1.7177432059087017 / 60. / 180. * np.pi


def build_mask(nx, structured=True, seed=0):
    rng = np.random.default_rng(seed)
    m = np.ones((nx, nx))
    m[0, :] = m[:, 0] = m[-1, :] = m[:, -1] = 0.
    if structured:
        for _ in range(max(3, nx // 24)):
            ci, cj = rng.integers(4, nx - 4, 2)
            r = rng.integers(1, max(2, nx // 30))
            ii, jj = np.ogrid[:nx, :nx]
            m[((ii - ci) ** 2 + (jj - cj) ** 2) <= r * r] = 0.
    return nmt.mask_apodization_flat(m, nx * DX, nx * DX, 0.2, apotype="C1")


def nmt_matrix(mask, pix, l0, lf):
    """NaMaster's binned matrix, recovered by probing decouple_cell with basis vectors
    (NmtWorkspaceFlat exposes no accessor; decouple_cell applies the inverse)."""
    nb = len(l0)
    f = nmt.NmtFieldFlat(pix.nx * pix.dx, pix.ny * pix.dy, mask, [np.zeros(mask.shape)])
    w = nmt.NmtWorkspaceFlat()
    w.compute_coupling_matrix(f, f, nmt.NmtBinFlat(l0, lf))
    m_inv = np.zeros((nb, nb))
    for b in range(nb):
        e = np.zeros(nb)
        e[b] = 1.
        m_inv[:, b] = w.decouple_cell([e])[0]
    return np.linalg.inv(m_inv)


def bins_for(nx, bin_fac):
    n = int(nx / bin_fac)
    return (np.arange(n) * bin_fac * np.pi / (nx * DX),
            (np.arange(n) + 1) * bin_fac * np.pi / (nx * DX))


# ------------------------------------------------------------ 1. normalisation is absolute
print("\n1. Absolute normalisation (no fitted constant): identity for a structureless mask")
nx = 128
pix = maps.pixel(nx, DX)
l0, lf = bins_for(nx, 8)
M = spec.get_coupling_matrix_jax(np.ones((nx, nx)), pix, l0, lf)
print(f"    diag(M)[:4] = {np.diag(M)[:4]}")
check("all-ones mask: M == identity", np.abs(M - np.eye(len(l0))).max(), 1e-12)

# ------------------------------------------------------------ 2. our matrix is symmetric
print("\n2. N_b * M must be symmetric -- the defining double sum is")
mask = build_mask(128)
l0, lf = bins_for(128, 4)
M = spec.get_coupling_matrix_jax(mask, pix, l0, lf)
chi = spec.get_bin_indicators(pix, l0, lf)
n_b = chi.reshape(len(l0), -1).sum(axis=1)
S = M * n_b[:, None]
check("ours: max |S - S^T| / max|S|", np.abs(S - S.T).max() / np.abs(S).max(), 1e-12)

M_nmt = nmt_matrix(mask, pix, l0, lf)
S_nmt = M_nmt * n_b[:, None]
asym = np.abs(S_nmt - S_nmt.T).max() / np.abs(S_nmt).max()
disagree = np.abs(M - M_nmt).max() / np.abs(M_nmt).max()
print(f"    NaMaster: max |S - S^T| / max|S|  = {asym:.3e}   <- not ours to fix")
print(f"    ours vs NaMaster                  = {disagree:.3e}")
print(f"    the two track each other: ratio     {disagree / asym:.2f}")
check("disagreement is the size of NaMaster's asymmetry", disagree / asym, 5.0)

# ------------------------------------------------------------ 3. agrees where that vanishes
print("\n3. Agreement with NaMaster where its asymmetry is negligible")
for nx_, bf, label in [(64, 32, "few bins"), (128, 16, "few bins"), (128, 128, "one bin")]:
    p = maps.pixel(nx_, DX)
    a, b = bins_for(nx_, bf)
    if len(a) < 1:
        continue
    mk = build_mask(nx_)
    Mg = spec.get_coupling_matrix_jax(mk, p, a, b)
    Mn = nmt_matrix(mk, p, a, b)
    d = np.abs(Mg - Mn).max() / np.abs(Mn).max()
    print(f"    {nx_}^2, {len(a):3d} bins ({label}): {d:.3e}")
    check(f"{nx_}^2 with {len(a)} bins matches NaMaster", d, 1e-9)

print("\n   how it degrades as the bins narrow (NaMaster's asymmetry grows):")
print(f"    {'grid':>8}{'bins':>6}{'ours vs nmt':>14}{'nmt asymmetry':>16}")
for nx_, bf in [(128, 16), (128, 8), (128, 4), (128, 2)]:
    p = maps.pixel(nx_, DX)
    a, b = bins_for(nx_, bf)
    mk = build_mask(nx_)
    Mg = spec.get_coupling_matrix_jax(mk, p, a, b)
    Mn = nmt_matrix(mk, p, a, b)
    nb_ = spec.get_bin_indicators(p, a, b).reshape(len(a), -1).sum(axis=1)
    Sn = Mn * nb_[:, None]
    print(f"    {nx_:>6}^2{len(a):>6}{np.abs(Mg-Mn).max()/np.abs(Mn).max():>14.3e}"
          f"{np.abs(Sn-Sn.T).max()/np.abs(Sn).max():>16.3e}")

# ------------------------------------------------------------ 4. speed at szifi's real size
print("\n4. Speed at szifi's real size (1024^2, 256 bins)")
nx = 1024
pix = maps.pixel(nx, DX)
l0, lf = bins_for(nx, 4)
mask = build_mask(nx)
f = nmt.NmtFieldFlat(nx * DX, nx * DX, mask, [np.zeros((nx, nx))])
t0 = time.time()
w = nmt.NmtWorkspaceFlat()
w.compute_coupling_matrix(f, f, nmt.NmtBinFlat(l0, lf))
t_nmt = time.time() - t0
spec.get_coupling_matrix_jax(mask, pix, l0, lf)          # warm the JIT
t0 = time.time()
spec.get_coupling_matrix_jax(mask, pix, l0, lf)
t_jax = time.time() - t0
print(f"    NaMaster: {t_nmt:7.1f}s")
print(f"    JAX/GPU:  {t_jax:7.2f}s   -> {t_nmt / t_jax:.0f}x")

print("\n" + "=" * 78)
n_fail = sum(1 for _, _, _, ok in RESULTS if not ok)
for name, value, tol, ok in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:56s} {value:.3e} <= {tol:.1e}")
print("=" * 78)
print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
sys.exit(1 if n_fail else 0)
