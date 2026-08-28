import jax.numpy as jnp
import jax
import numpy as np
from szifi_jax import maps

#Various utility functions

#Calculate covariance (or ILC) bias on signal-to-noise

def get_q_bias(pix,tem,signal,cov_mf_true,cov_noi_true,n_modes_per_bin_map,type="q"):

    tem_fft = maps.get_fft(tem,pix)
    signal_fft = maps.get_fft(signal,pix)

    n0 = filter_sum(tem_fft,tem_fft,cov_mf_true)

    map1 = (signal_fft+jnp.conjugate(signal_fft))*cov_noi_true*jnp.conjugate(tem_fft)
    map2 = (signal_fft+jnp.conjugate(signal_fft))*cov_noi_true*jnp.conjugate(tem_fft)*jnp.abs(tem_fft)**2
    map1 = div0(map1,cov_mf_true**2*n_modes_per_bin_map)
    map2 = div0(map2,cov_mf_true**3*n_modes_per_bin_map)

    sum1 = jnp.sum(map1).real
    sum2 = jnp.sum(map2).real

    if type == "q":

        bias1 = -sum1/n0**0.5
        bias2 = 0.5*sum2/n0**1.5

    elif type == "y0":

        bias1 = -sum1/n0
        bias2 = sum2/n0**2

    return bias1,bias2

#Calculate covariance (or ILC) bias on signal-to-noise variance

def get_q_std_bias(pix,tem,signal,cov_mf_true,cov_noi_true):

    tem_fft = maps.get_fft(tem,pix)
    signal_fft = maps.get_fft(signal,pix)
    n0 = filter_sum(tem_fft,tem_fft,cov_mf_true)
    n_bias = filter_sum(tem_fft,tem_fft*cov_noi_true,(cov_mf_true)**2)
    n_bias = filter_sum(tem_fft,tem_fft*cov_noi_true,(cov_mf_true)**2)
    sigma_unbiased = 1./jnp.sqrt(n0)
    sigma_biased = 1./jnp.sqrt(n_bias)

    return sigma_unbiased/sigma_biased

#Returns MMF y0 at centre (just one cluster)

def get_mmf_y0(tmap,tem,inv_cov,pix,nfreq=6,ell_filter=None,mmf_type="standard",
cmmf_prec=None):

    if ell_filter == None:

        ell_filter = [0.,1./pix.dx*10.]

    n_freqs = tmap.shape[2]
    y_map = jnp.zeros((pix.nx,pix.ny))
    norm_map = jnp.zeros((pix.nx,pix.ny))

    tem_fft = maps.get_fft_f(tem,pix)
    tem_fft = tem_fft

    tem = maps.get_ifft_f(maps.filter_fft_f(tem_fft,pix,ell_filter),pix).real

    y0_vec = jnp.zeros(nfreq+1)
    std_vec = jnp.zeros(nfreq+1)

    from szifi_jax import mmf as mmf_module
    filter_fft = mmf_module.get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,cmmf_prec=cmmf_prec)
    filter_fft = maps.filter_fft_f(filter_fft,pix,ell_filter)

    for i in range(nfreq):

        y = _fftconvolve_same(tmap[:,:,i],jnp.real(maps.get_ifft(filter_fft[:,:,i],pix)))*pix.dx*pix.dy
        sigma2 = _fftconvolve_same(tem[:,:,i],jnp.real(maps.get_ifft(filter_fft[:,:,i],pix)))*pix.dx*pix.dy

        y_map = y_map + y
        norm_map = norm_map + sigma2

        y0_vec = y0_vec.at[i].set(y[pix.nx//2,pix.ny//2]/jnp.max(sigma2))
        std_vec = std_vec.at[i].set(1/jnp.sqrt(jnp.max(sigma2)))

    norm = jnp.max(norm_map)

    y0_vec = y0_vec.at[-1].set(y_map[pix.nx//2,pix.ny//2]/jnp.max(norm_map))
    std_vec = std_vec.at[-1].set(1/jnp.sqrt(jnp.max(norm_map)))

    if mmf_type == "spectrally_constrained":

        std_vec = std_vec.at[-1].set(get_cmmf_std(pix,mmf_type,inv_cov,filter_fft,norm))

    return y0_vec,std_vec

def get_mf_map(obs,tem,noi,pix):

    tem_fft = maps.get_fft(tem,pix)
    tem_conv = maps.get_ifft(tem_fft/noi,pix).real

    map_convolution = _fftconvolve_same(obs,tem_conv)*pix.dx*pix.dy
    norm = jnp.max(_fftconvolve_same(tem,tem_conv))*pix.dx*pix.dy
    est_map = map_convolution/norm
    std = 1./jnp.sqrt(norm)

    return est_map,std

def filter_sum(rec,tem,noi):

    map1 = jnp.conjugate(rec)*tem
    map1 = div0(map1,noi)
    map2 = jnp.conjugate(tem)*tem
    map2 = div0(map2,noi)

    sum_1 = jnp.sum(map1).real
    sum_2 = jnp.sum(map2).real

    normalisation_estimate = sum_1/sum_2
    normalisation_variance = 1./sum_2

    return (normalisation_estimate,normalisation_variance)

def div0(a,b):

    """a/b with every non-finite result replaced by zero, as in szifi's
    np.true_divide + c[~np.isfinite(c)] = 0. Testing b != 0 is not equivalent: it lets a
    nan numerator through, and it keeps an overflow-to-inf that szifi would have zeroed."""

    c = jnp.where(b != 0., a / jnp.where(b != 0., b, 1.), 0.)

    return jnp.where(jnp.isfinite(c), c, 0.)

def get_cmmf_std(pix,mmf_type,inv_cov,filter_fft,norm):

    freqs = jnp.arange(filter_fft.shape[2])

    cov = invert_cov(inv_cov)

    filter_fft_2 = get_inv_cov_conjugate(filter_fft,cov)

    filter_2 = maps.get_ifft_f(filter_fft_2,pix).real
    filter_val = maps.get_ifft_f(filter_fft,pix).real

    var_map = jnp.zeros((pix.nx,pix.ny),dtype=complex)

    for i in range(len(freqs)):

        var_map = var_map + _fftconvolve_same(filter_val[:,:,freqs[i]],filter_2[:,:,i])*pix.dx*pix.dy

    var_map = var_map/norm**2
    var = var_map[pix.nx//2,pix.nx//2].real
    std = jnp.sqrt(var)

    return std

def invert_cov(cov):
    # Batched inversion: cov is (nx, ny, nf, nf)
    # Use jnp.linalg.inv which handles batched dims natively
    det = jnp.linalg.det(cov)
    inv = jnp.linalg.inv(cov)
    # Zero out where determinant is zero (singular matrices)
    valid = jnp.abs(det) > 0.
    inv = jnp.where(valid[..., None, None], inv, 0.)
    return inv

def get_inv_cov_conjugate(tem_fft,inv_cov):

    return jnp.einsum('dhi,dhij->dhj',tem_fft,inv_cov)

def get_inv_cov_dot(a,inv_cov,b):

    conjugate = get_inv_cov_conjugate(a,inv_cov)

    return jnp.einsum('dhi,dhi->dh',conjugate,b)

def gaussian_1d(x,mu,sigma):

    y = (x-mu)/sigma

    return jnp.exp(-0.5*y**2)/(jnp.sqrt(2.*jnp.pi*sigma**2))

def extract(dct,name,field_id=None, dtype=None):

    """Return a data entry and delete it from this object for memory management"""

    if field_id is None:

        obj = dct.pop(name)

    else:

        obj = dct[name][field_id]

        del dct[name][field_id]

    if isinstance(obj, np.ndarray) and (dtype is not None):

        obj = np.asarray(obj,dtype)

    return obj

def _fftconvolve_same(a, b):
    """Replicate scipy.signal.fftconvolve(a, b, mode='same') using JAX FFT."""
    s1 = jnp.array(a.shape)
    s2 = jnp.array(b.shape)
    shape = s1 + s2 - 1
    fa = jnp.fft.fft2(a, s=tuple(shape))
    fb = jnp.fft.fft2(b, s=tuple(shape))
    ret = jnp.fft.ifft2(fa * fb).real
    # Crop to 'same' size (centered)
    start0 = (s2[0] - 1) // 2
    start1 = (s2[1] - 1) // 2
    return jax.lax.dynamic_slice(ret, (start0, start1), (s1[0], s1[1]))


def _next_fast_fft_len(n):
    """Find next FFT-friendly size >= n (product of small primes 2,3,5)."""
    target = n
    while True:
        m = target
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return target
        target += 1


def _fftconvolve_same_sum(a_3d, b_3d):
    """Vectorized fftconvolve(a[:,:,i], b[:,:,i], mode='same') summed over i.

    a_3d, b_3d: (nx, ny, n_freq) arrays
    Returns: (nx, ny) sum of per-frequency convolutions
    """
    nx, ny, n_freq = a_3d.shape
    # Padded shape for linear convolution, rounded up to FFT-friendly size
    pnx = _next_fast_fft_len(2 * nx - 1)
    pny = _next_fast_fft_len(2 * ny - 1)
    # Batched FFT over frequency axis
    fa = jax.vmap(lambda m: jnp.fft.fft2(m, s=(pnx, pny)), in_axes=2)(a_3d)  # (n_freq, pnx, pny)
    fb = jax.vmap(lambda m: jnp.fft.fft2(m, s=(pnx, pny)), in_axes=2)(b_3d)
    # Multiply and sum over frequencies, then single IFFT
    product_sum = jnp.sum(fa * fb, axis=0)  # (pnx, pny)
    ret = jnp.fft.ifft2(product_sum).real
    # Crop to 'same' size
    start0 = (nx - 1) // 2
    start1 = (ny - 1) // 2
    return jax.lax.dynamic_slice(ret, (start0, start1), (nx, ny))
