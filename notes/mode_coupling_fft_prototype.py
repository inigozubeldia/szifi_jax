"""
PROTOTYPE -- NOT USED BY THE PACKAGE, NOT VALIDATED FOR SCIENCE.

A GPU implementation of the flat-sky MASTER mode-coupling matrix, which is the single
dominant cost in szifi: profiling a Planck field-0 run puts NaMaster's
`NmtWorkspaceFlat.compute_coupling_matrix` at 97% of the wall clock (~139 s per tile),
against 0.71 s for the entire batched matched filter.

There is no JAX NaMaster. The JAX ecosystem has s2fft and jax-healpy, but those are
curved-sky spherical harmonic transforms; szifi uses the flat-sky estimator, which they do
not cover. The known literature speed-up is the Toeplitz approximation
(Louis et al., arXiv:2010.14344), which is curved-sky and approximate.

THE FORMULATION
---------------
Flat-sky MASTER couples bandpowers as

    M_bb' = (1/N_b) sum_{l in b} sum_{l' in b'} |W(l - l')|^2

with W the Fourier transform of the mask and the difference taken modulo the grid (verified:
removing the periodic wrap by zero-padding makes agreement with pymaster 100x worse). Writing
chi_b for the indicator of bin b on the 2D Fourier grid and P = |W|^2, the double sum is a
correlation,

    sum_q P(q) (chi_b corr chi_b')(q)

and correlations are FFTs, so the whole matrix is one Gram contraction:

    M_bb' ~ Re sum_x conj(Phat(x)) F_b(x) conj(F_b'(x)) / N_b,
    F_b = FFT(chi_b),  Phat = FFT(P).

That is n_bins + 1 FFTs plus an (n_bins x N) @ (N x n_bins) product -- ideal for a GPU.

WHAT IS ESTABLISHED
-------------------
* The formula is exact. A brute-force periodic double sum reproduces pymaster's matrix to
  3.4e-15 on a 32x32 grid.
* pymaster does use the periodic (wrapped) difference.
* Speed at szifi's real problem size (1024x1024, 256 bins, one GPU):
      pymaster   138.8 s
      this code    0.36 s   (0.75 s including JIT compilation)
  i.e. ~385x.

WHAT IS NOT
-----------
At 256x256 the agreement with pymaster is ~6e-7 for most elements but 1.5e-4 for one bin.
That is far too coarse to be a drop-in: the port currently reproduces szifi's q_opt to 3.8e-8.
Three candidate causes were tested and REJECTED:
  * bin-edge ties -- an exact integer bin assignment (4*(ix^2+iy^2) vs (b*bin_fac)^2, which
    removes all floating-point comparison) gives the identical residual;
  * the periodic wrap -- removing it is 100x worse;
  * catastrophic cancellation in the Gram sum -- the measured cancellation factor is 1.00.
The brute-force sum tracks this FFT implementation to the expected factor of N, so whatever
the discrepancy is, it is shared by both and lies in how pymaster's matrix was recovered
(by probing `decouple_cell` with basis vectors and inverting) or in an extra step pymaster
applies. That has to be understood before this is worth finishing.

DO NOT wire this into the pipeline until it matches pymaster to ~1e-12 on a full-size mask.
"""

import numpy as np
import jax
import jax.numpy as jnp


def bin_indicators(nx, ny, bin_fac, n_bins):
    """Indicator of each ell bin on the 2D Fourier grid, by exact integer comparison.

    szifi's bins are l0_b = b*bin_fac*pi/(nx*dx) and the grid is l = f*sqrt(ix^2 + iy^2) with
    f = 2*pi/(nx*dx), so `l >= l0_b` is exactly `4*(ix^2 + iy^2) >= (b*bin_fac)^2`."""

    ix = (np.fft.fftfreq(nx) * nx).round().astype(np.int64)
    iy = (np.fft.fftfreq(ny) * ny).round().astype(np.int64)
    n2 = 4 * (ix[:, None] ** 2 + iy[None, :] ** 2)
    edges = (np.arange(n_bins + 1) * bin_fac) ** 2

    return np.stack([((n2 >= edges[b]) & (n2 < edges[b + 1])).astype(np.float64)
                     for b in range(n_bins)])


@jax.jit
def coupling_matrix_fft(mask, chi):
    """Un-normalised flat-sky mode-coupling matrix, up to one overall constant."""

    p_hat = jnp.fft.fft2(jnp.abs(jnp.fft.fft2(mask)) ** 2).ravel()
    f = jnp.fft.fft2(chi, axes=(1, 2)).reshape(chi.shape[0], -1)
    g = f * jnp.conj(p_hat)[None, :]
    n_b = chi.reshape(chi.shape[0], -1).sum(axis=1)

    return jnp.real(g @ jnp.conj(f).T) / n_b[:, None]


def reference_matrix_from_pymaster(w, n_bins):
    """Recover pymaster's binned coupling matrix by probing decouple_cell with basis vectors.

    NmtWorkspaceFlat exposes no accessor for the matrix, but decouple_cell applies its
    inverse, so decouple_cell(e_b) is the b-th column of M^-1."""

    m_inv = np.zeros((n_bins, n_bins))

    for b in range(n_bins):

        e = np.zeros(n_bins)
        e[b] = 1.
        m_inv[:, b] = w.decouple_cell([e])[0]

    return np.linalg.inv(m_inv)
