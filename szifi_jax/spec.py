import numpy as np
import jax
import jax.numpy as jnp
import functools
import os
from scipy import interpolate
import pymaster as nmt
from szifi_jax import maps
import scipy.ndimage as ndimage


# ============================================================
# JIT-compiled functions for custom power spectrum estimation
# ============================================================


@functools.partial(jax.jit, static_argnums=(4,))
def _bin_cross_spec_jit(cross_spec_maps, ell_map, bins_low, bins_high, n_bins):
    """Bin cross-spectra into ell bins. Vectorized over all freq pairs.

    cross_spec_maps: (nx, ny, n_freq, n_freq) real cross-power maps
    ell_map: (nx, ny) ell values
    bins_low, bins_high: (n_bins,) bin edges
    Returns: (n_bins, n_freq, n_freq) binned spectra, (n_bins,) modes per bin
    """
    n_freq = cross_spec_maps.shape[2]
    ell_flat = ell_map.ravel()

    # Assign each pixel to a bin using digitize
    # digitize returns index i such that bins_low[i-1] <= ell < bins_low[i]
    # We want: bin i contains pixels where bins_low[i] <= ell < bins_high[i]
    bin_idx = jnp.digitize(ell_flat, bins_low) - 1  # 0-indexed

    # Also need ell < bins_high[bin_idx] check
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    # Check upper bound
    valid = valid & (ell_flat < bins_high[jnp.clip(bin_idx, 0, n_bins-1)])

    # Flatten cross-spec maps: (n_pix, n_freq, n_freq)
    n_pix = ell_flat.shape[0]
    cross_flat = cross_spec_maps.reshape(n_pix, n_freq, n_freq)

    # Sum into bins using segment_sum-like approach
    # Use jnp.zeros + scatter_add equivalent
    safe_idx = jnp.where(valid, bin_idx, 0)
    mask_valid = valid.astype(cross_flat.dtype)

    # Count modes per bin
    n_modes = jnp.zeros(n_bins, dtype=cross_flat.dtype)
    n_modes = n_modes.at[safe_idx].add(mask_valid)

    # Sum spectra per bin — vectorize over freq pairs
    spec_sum = jnp.zeros((n_bins, n_freq, n_freq), dtype=cross_flat.dtype)
    spec_sum = spec_sum.at[safe_idx].add(cross_flat * mask_valid[:, None, None])

    # Mean = sum / count
    spec_binned = spec_sum / jnp.maximum(n_modes[:, None, None], 1.)

    return spec_binned, n_modes


@jax.jit
def _interp_cov_jit(ell_map, ell_vec, spec_tensor):
    """Interpolate 1D binned spectra onto 2D ell map, nearest-neighbour.

    Reproduces scipy.interpolate.interp1d(kind="nearest"): ties go to the lower bin
    (scipy searchsorts the bin midpoints with side="left"), and out-of-range values
    clamp to the end bins.

    ell_map: (nx, ny) 2D ell values
    ell_vec: (n_ell,) 1D ell bin centres
    spec_tensor: (n_ell, n_freq, n_freq) binned cross-spectra
    Returns: (nx, ny, n_freq, n_freq) interpolated covariance
    """
    n_ell, n_freq, _ = spec_tensor.shape
    ell_flat = ell_map.ravel()

    # scipy's interp1d(kind="nearest") builds the bin boundaries once, as the midpoints
    # between consecutive centres, and searchsorts against them with side="left":
    #
    #     x_bds = (x[1:] + x[:-1]) / 2
    #     idx = searchsorted(x_bds, x_new, side="left").clip(0, len(x) - 1)
    #
    # Do the same rather than comparing |v - left| against |v - right|. The two agree only
    # when the midpoint is exactly representable; where it is not, a value sitting on a
    # boundary snaps to a different bin under the two rules. That is not a corner case here:
    # the bin centres sit at odd multiples of the fundamental mode and the boundaries at even
    # multiples, so every mode on the lx or ly axis lands exactly on one.
    ell_bds = 0.5 * (ell_vec[1:] + ell_vec[:-1])
    nearest_idx = jnp.clip(jnp.searchsorted(ell_bds, ell_flat, side="left"), 0, n_ell - 1)

    # Gather: (n_pix, n_freq, n_freq)
    result = spec_tensor[nearest_idx]

    return result.reshape(ell_map.shape[0], ell_map.shape[1], n_freq, n_freq)


@jax.jit
def _interp_cov_linear_jit(ell_map, ell_vec, spec_tensor):
    """Linear interpolation of 1D binned spectra onto a 2D ell map, with linear
    extrapolation beyond the end bins (scipy's fill_value="extrapolate" behaviour;
    jnp.interp clamps instead, so the slope has to be applied by hand)."""
    n_ell = spec_tensor.shape[0]
    ell_flat = ell_map.ravel()

    idx = jnp.clip(jnp.searchsorted(ell_vec, ell_flat) - 1, 0, n_ell - 2)

    x0 = ell_vec[idx]
    x1 = ell_vec[idx + 1]
    y0 = spec_tensor[idx]
    y1 = spec_tensor[idx + 1]

    t = (ell_flat - x0) / (x1 - x0)
    result = y0 + (y1 - y0) * t[:, None, None]

    return result.reshape(ell_map.shape[0], ell_map.shape[1], *spec_tensor.shape[1:])



# ============================================================
# Classes
# ============================================================

class power_spectrum:

    def __init__(self, pix, mask=None, cm_compute=False, cm_save=False,
    cm_name=None, bins=None, bin_fac=4., new_shape=None, cm_compute_scratch=False):

        if new_shape is not None:
            pix = maps.degrade_pix(pix, new_shape)
            mask = maps.degrade_map(mask, new_shape)

        if mask is None:

            mask = np.ones((pix.nx,pix.ny))

        if bins is None:

            l0_bins = np.arange(pix.nx/bin_fac)*bin_fac*np.pi/(pix.nx*pix.dx)
            lf_bins = (np.arange(pix.nx/bin_fac)+1)*bin_fac*np.pi/(pix.nx*pix.dx)
            bins = nmt.NmtBinFlat(l0_bins,lf_bins)

        self.l0_bins = l0_bins
        self.lf_bins = lf_bins
        self.pix = pix
        self.mask = mask
        self.bins = bins
        self.new_shape = new_shape
        self.n_modes_per_bin = np.zeros(len(self.l0_bins))

        self.ell_eff = self.bins.get_effective_ells()
        self.w00 = None

        if cm_compute == True:

            if cm_name == None:

                print("Coupling matrix not found")
                self.get_coupling_matrix()

                if cm_save == True:

                    self.w00.write_to(cm_name)

            elif os.path.isfile(cm_name) == False:

                print("Coupling matrix not found")
                self.get_coupling_matrix()

                if cm_save == True:

                    self.w00.write_to(cm_name)

            elif os.path.isfile(cm_name) == True:

                print("Coupling matrix found")

                if cm_compute_scratch == True:

                    os.remove(cm_name)
                    print("Coupling matrix recomputation")
                    self.get_coupling_matrix()

                    if cm_save == True:

                        self.w00.write_to(cm_name)

                else:

                    self.w00 = nmt.NmtWorkspaceFlat()
                    self.w00.read_from(cm_name)


    def get_coupling_matrix(self):

        f0 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,self.mask,[np.zeros((self.pix.nx,self.pix.ny))])
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f0,f0,self.bins)
        self.w00 = w00

    #Note: map1 is UNMASKED

    def get_power_spectrum(self,map1,map2=None,decouple_type="none",implementation="pymaster"):
        if self.new_shape is not None:
            map1 = maps.degrade_map(map1, self.new_shape)
            map2 = maps.degrade_map(map2, self.new_shape)
        if map2 is None:

            map2 = map1

        # Convert JAX arrays to numpy for pymaster
        map1_np = np.asarray(map1)
        map2_np = np.asarray(map2)
        mask_np = np.asarray(self.mask)

        if implementation == "pymaster":

            f1 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,mask_np,[map1_np])
            f2 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,mask_np,[map2_np])
            cl00_coupled = nmt.compute_coupled_cell_flat(f1,f2,self.bins)

            if decouple_type == "none":

                ret = (self.ell_eff,cl00_coupled[0])

            elif decouple_type == "master":

                if self.w00 is None:

                    self.get_coupling_matrix()

                ret = (self.ell_eff,self.w00.decouple_cell(cl00_coupled)[0])

            elif decouple_type == "fsky":

                fsky = np.sum(mask_np)/(self.pix.nx*self.pix.ny)
                m1 = np.sum(mask_np**2)/(self.pix.nx*self.pix.ny)
                m2 = np.sum(mask_np**4)/(self.pix.nx*self.pix.ny)
                fsky = fsky*m1**2/m2

                ret = (self.ell_eff,cl00_coupled[0]/fsky)

        elif implementation == "custom":

            map1_fft = maps.get_fft(jnp.asarray(map1),self.pix)
            map2_fft = maps.get_fft(jnp.asarray(map2),self.pix)
            spec_map = jnp.conjugate(map1_fft)*map2_fft
            ell_map = maps.rmap(self.pix).get_ell()
            (ell_vec_eff,spec_tensor_binned,n_modes_per_bin,n_modes_per_bin_map) = get_binned_spec(spec_map,ell_map,self.l0_bins,self.lf_bins)
            self.ell_eff = ell_vec_eff
            self.n_modes_per_bin = n_modes_per_bin
            self.n_modes_per_bin_map = n_modes_per_bin_map
            self.ell_map = ell_map

            ret = (ell_vec_eff,spec_tensor_binned)

        return ret

def get_binned_spec(spec_map,ell_map,bins_low,bins_high):

    spec_map_np = np.asarray(spec_map)
    ell_map_np = np.asarray(ell_map)

    spec_tensor_binned = np.zeros(len(bins_low))
    n_modes_per_bin = np.zeros(len(bins_low))
    n_modes_per_bin_map = np.zeros(ell_map_np.shape[0]*ell_map_np.shape[1])
    spec_map_shape = spec_map_np.shape

    for i in range(0,len(bins_low)):

        ell_flat = ell_map_np.flatten()
        indices = np.where((ell_flat < bins_high[i]) & (ell_flat >= bins_low[i]))[0]

        spec_tensor_binned[i] = np.mean(spec_map_np.flatten()[indices])
        n_modes_per_bin[i] = len(indices)
        n_modes_per_bin_map[indices] = len(indices)

    n_modes_per_bin_map = n_modes_per_bin_map.reshape(spec_map_shape)

    ell_vec_eff = 0.5*(bins_low+bins_high)

    return (ell_vec_eff,spec_tensor_binned,n_modes_per_bin,jnp.array(n_modes_per_bin_map))


class cross_spec:

    def __init__(self,freqs):

        self.freqs = freqs
        self.n_freqs = len(self.freqs)
        self.spec_tensor = None
        self.ell_vec = None

    def get_cross_spec(self,pix,t_map=None,t_map_2=None,mask=None,bin_fac=4., new_shape=None, ps=None,
    decouple_type="master",inpaint_flag=False,mask_point=None,
    lsep=3000,implementation="pymaster"):

        pix = maps.degrade_pix(pix, new_shape)

        if mask is None:

            mask = np.ones((pix.nx,pix.ny))

        if mask_point is None:

            mask_point = np.ones((pix.nx,pix.ny))

        if t_map_2 is None:

            t_map_2 = t_map

        if ps is None:

            ps = power_spectrum(pix, mask=mask, cm_compute=False, bin_fac=bin_fac, new_shape=new_shape)

        self.ps = ps

        if ps.new_shape != new_shape:

            raise ValueError(f"cross_spec new_shape={new_shape} and ps {ps.new_shape} should be equal")

        if implementation == "custom":

            return self._get_cross_spec_custom(pix, t_map, t_map_2, mask,
                                                mask_point, ps, inpaint_flag, lsep,
                                                decouple_type)

        # Original pymaster path
        ell = ps.ell_eff[1:]
        n_freq = self.n_freqs
        spec_tensor = np.zeros((len(ell),n_freq,n_freq))

        t_map_np = np.asarray(t_map)
        t_map_2_np = np.asarray(t_map_2)

        for i in range(0,n_freq):

            for j in range(0,n_freq):

                if j >= i:

                    ell,cl = ps.get_power_spectrum(t_map_np[:,:,i],
                    map2=t_map_2_np[:,:,j],decouple_type=decouple_type,
                    implementation=implementation)
                    if inpaint_flag == True:

                        ell,cl = correct_for_inpainting((ell,cl),np.asarray(mask_point),lsep=lsep)

                    ell = ell[1:]
                    cl = cl[1:]
                    spec_tensor[:,i,j] = cl

                else:

                    spec_tensor[:,i,j] = spec_tensor[:,j,i]

        self.spec_tensor = spec_tensor
        self.ell_vec = ell

        return ell,spec_tensor

    def _get_cross_spec_custom(self, pix, t_map, t_map_2, mask,
                                mask_point, ps, inpaint_flag, lsep,
                                decouple_type):
        """Fully vectorized JAX cross-spectrum estimation (no pymaster)."""
        n_freq = self.n_freqs
        mask_jnp = jnp.asarray(mask)
        t_map_jnp = jnp.asarray(t_map)
        t_map_2_jnp = jnp.asarray(t_map_2)

        # Degrade maps if needed
        if ps.new_shape is not None:
            t_map_jnp = jnp.stack([maps.degrade_map(t_map_jnp[:,:,i], ps.new_shape)
                                    for i in range(n_freq)], axis=2)
            t_map_2_jnp = jnp.stack([maps.degrade_map(t_map_2_jnp[:,:,i], ps.new_shape)
                                      for i in range(n_freq)], axis=2)

        # Apply mask to all frequency maps
        t_map_masked = t_map_jnp * mask_jnp[:,:,None]
        t_map_2_masked = t_map_2_jnp * mask_jnp[:,:,None]

        # FFT all frequency channels at once
        tmap_fft_1 = maps.get_fft_f(t_map_masked, pix)
        tmap_fft_2 = maps.get_fft_f(t_map_2_masked, pix)

        # Compute all cross-spectra at once: conj(fft1_i) * fft2_j
        cross_maps = jnp.real(jnp.conj(tmap_fft_1[:,:,:,None]) * tmap_fft_2[:,:,None,:])

        # Get ell map
        ell_map = maps.rmap(pix).get_ell()

        # Bin all cross-spectra
        bins_low = jnp.asarray(ps.l0_bins)
        bins_high = jnp.asarray(ps.lf_bins)
        n_bins = len(ps.l0_bins)

        spec_binned, n_modes = _bin_cross_spec_jit(cross_maps, ell_map,
                                                     bins_low, bins_high, n_bins)

        # Effective ell (bin centres)
        ell_eff = np.asarray(0.5 * (ps.l0_bins + ps.lf_bins))

        # Apply fsky correction for decouple_type == "fsky" or "none"
        # For "none": no correction (coupled spectrum)
        # For "fsky": divide by effective fsky
        if decouple_type == "fsky":
            mask_np = np.asarray(mask)
            fsky = np.sum(mask_np) / (pix.nx * pix.ny)
            m1 = np.sum(mask_np**2) / (pix.nx * pix.ny)
            m2 = np.sum(mask_np**4) / (pix.nx * pix.ny)
            fsky = fsky * m1**2 / m2
            spec_binned = spec_binned / fsky

        # Apply inpainting correction if needed
        if inpaint_flag:
            mask_point_np = np.asarray(mask_point)
            fsky_inpaint = np.sum(mask_point_np) / (mask_point_np.shape[0] * mask_point_np.shape[1])
            # Correction: divide high-ell bins by fsky
            ell_eff_jnp = jnp.asarray(ell_eff)
            correction = jnp.where(ell_eff_jnp > lsep, 1.0 / fsky_inpaint, 1.0)
            spec_binned = spec_binned * correction[:, None, None]

        # Drop first bin (matching original behaviour: ell = ps.ell_eff[1:])
        spec_tensor = np.asarray(spec_binned[1:])
        ell_vec = ell_eff[1:]

        self.spec_tensor = spec_tensor
        self.ell_vec = ell_vec

        return ell_vec, spec_tensor


    def get_cov(self,pix,t_map=None,mask=None,bin_fac=4,new_shape=None,ps=None,
    decouple_type="master",interp_type="nearest",cov_type="isotropic",cov_kernel_shape=None):

        def get_cov_isotropic():

            if self.spec_tensor is None:

                ell_vec,spec_tensor = self.get_cross_spec(pix,t_map=t_map,
                                                          mask=mask,
                                                          bin_fac=bin_fac,
                                                          new_shape=new_shape,
                                                          ps=ps,
                                                          decouple_type=decouple_type)

            else:

                ell_vec = self.ell_vec
                spec_tensor = self.spec_tensor

            pix_d = maps.degrade_pix(pix,new_shape)
            ell_map = maps.rmap(pix_d).get_ell()
            cov_tensor = interp_cov(ell_map,ell_vec,spec_tensor,interp_type=interp_type)

            return cov_tensor

        if cov_type == "isotropic":

            cov_tensor = get_cov_isotropic()

        elif cov_type == "anisotropic_gaussian" or cov_type == "anisotropic_boxcar":

            cov_tensor = get_cov_anisotropic(t_map,pix,mask=mask,cov_type=cov_type,cov_kernel_shape=cov_kernel_shape)

        elif cov_type == "mixed_gaussian" or cov_type == "mixed_boxcar":

            cov_tensor_isotropic = get_cov_isotropic()
            cov_tensor = get_cov_anisotropic(t_map,pix,mask=mask,cov_type=cov_type,cov_kernel_shape=cov_kernel_shape,cov_tensor_isotropic=cov_tensor_isotropic)

        return cov_tensor # nx x ny x n_freq x n_freq covariance tensor

    def get_inv_cov(self,pix,t_map=None,mask=None,bin_fac=4,new_shape=None,ps=None,
                    decouple_type="master",interp_type="nearest",cov_type="isotropic",
                    cov_kernel_shape=None,inv_cov_backend="numpy"):

        """Inverse of the per-mode noise covariance.

        `inv_cov_backend` selects the linear-algebra backend:

          "numpy" (default) — np.linalg.inv, i.e. exactly what szifi calls.
          "jax"             — jnp.linalg.inv, keeping the tensor on the device.

        These are not interchangeable. The noise covariance is numerically singular at some
        modes (|inv_cov| reaches ~3e14 on Planck field 0), and LAPACK and cuSOLVER diverge
        there: measured agreement is ~2e-15 for condition numbers below 1e4 but degrades to
        ~1e-3 beyond 1e12, which propagates to ~1e-4 in the filter normalisations and ~3e-3
        in the recovered SNR. The CPU inversion of a 1024x1024 grid of 6x6 matrices takes
        ~0.4 s against a ~300 s cluster-finding run, so exactness is the better default."""

        if cov_type == "anisotropic_gaussian":

            cov = _get_cov_anisotropic_gaussian_fast(t_map, pix, mask, cov_kernel_shape)

        else:

            cov = self.get_cov(pix,t_map=t_map,mask=mask,bin_fac=bin_fac,new_shape=new_shape,
            ps=ps,decouple_type=decouple_type,interp_type=interp_type,
            cov_type=cov_type,cov_kernel_shape=cov_kernel_shape)

        if inv_cov_backend == "numpy":

            return jnp.asarray(np.linalg.inv(np.asarray(cov)))

        elif inv_cov_backend == "jax":

            return jnp.linalg.inv(jnp.asarray(cov))

        raise ValueError(f"inv_cov_backend \"{inv_cov_backend}\" not supported. "
        "Allowed values are 'numpy' and 'jax'.")

def get_cov_anisotropic(t_map,pix,mask=None,cov_type=None,cov_kernel_shape=None,cov_tensor_isotropic=None):

    # Fast vectorized path for anisotropic_gaussian (no ps_isotropic)
    if cov_type == "anisotropic_gaussian" and cov_tensor_isotropic is None:
        return _get_cov_anisotropic_gaussian_fast(t_map, pix, mask, cov_kernel_shape)

    n_freq = t_map.shape[2]

    if cov_tensor_isotropic is None:

        cov_tensor = np.zeros((pix.nx,pix.ny,n_freq,n_freq))

    else:

        cov_tensor = np.zeros(np.asarray(cov_tensor_isotropic).shape)

    for i in range(0,n_freq):

        for j in range(0,n_freq):

            if j >= i:

                if cov_tensor_isotropic is None:

                    ps_isotropic = None

                else:

                    ps_isotropic = np.asarray(cov_tensor_isotropic)[:,:,i,j]

                cov_tensor[:,:,i,j] = get_anisotropic_ps(t_map[:,:,i],t_map[:,:,j],pix,type=cov_type,mask=mask,cov_kernel_shape=cov_kernel_shape,ps_isotropic=ps_isotropic)

            else:

                cov_tensor[:,:,i,j] = cov_tensor[:,:,j,i]

    return jnp.array(cov_tensor)


def _make_gaussian_kernel_2d(sigma_x, sigma_y, truncate=4.0):
    """Build the same discrete 2D Gaussian kernel that scipy.ndimage.gaussian_filter uses."""
    radius_x = int(truncate * sigma_x + 0.5)
    radius_y = int(truncate * sigma_y + 0.5)
    x = np.arange(-radius_x, radius_x + 1)
    y = np.arange(-radius_y, radius_y + 1)
    kx = np.exp(-0.5 * (x / sigma_x)**2)
    ky = np.exp(-0.5 * (y / sigma_y)**2)
    kx /= kx.sum()
    ky /= ky.sum()
    return jnp.array(np.outer(kx, ky))



def _get_cov_anisotropic_gaussian_fast(t_map, pix, mask, cov_kernel_shape):
    """Vectorized anisotropic Gaussian covariance: all freq pairs at once.
    FFTs + Gaussian smoothing fully on GPU, matching scipy.ndimage.gaussian_filter exactly."""
    t_map_jnp = jnp.asarray(t_map)
    mask_jnp = jnp.asarray(mask) if mask is not None else jnp.ones(t_map_jnp.shape[:2])

    sigma_x, sigma_y = float(cov_kernel_shape[0]), float(cov_kernel_shape[1])
    kernel = _make_gaussian_kernel_2d(sigma_x, sigma_y)
    pad_x = kernel.shape[0] // 2
    pad_y = kernel.shape[1] // 2

    fft_norm = jnp.float64(jnp.sqrt((pix.dx * pix.dy) / (pix.nx * pix.ny)))

    return _cov_aniso_gaussian_jit(t_map_jnp, mask_jnp, fft_norm, kernel, pad_x, pad_y)


@functools.partial(jax.jit, static_argnums=(4, 5))
def _cov_aniso_gaussian_jit(t_map, mask, fft_norm, kernel, pad_x, pad_y):
    """JIT-compiled anisotropic Gaussian covariance computation."""
    n_freq = t_map.shape[2]

    # Apply mask and FFT all channels at once
    t_masked = t_map * mask[:,:,None]
    t_fft = jax.vmap(lambda m: jnp.fft.fft2(m) * fft_norm, in_axes=2, out_axes=2)(t_masked)

    # All cross-spectra at once
    cross = jnp.real(jnp.conj(t_fft[:,:,:,None]) * t_fft[:,:,None,:])

    # Extract upper triangle, fftshift, smooth, ifftshift
    idx = jnp.triu_indices(n_freq)
    cross_pairs = cross[:, :, idx[0], idx[1]].transpose(2, 0, 1)
    cross_shifted = jax.vmap(jnp.fft.fftshift)(cross_pairs)

    smoothed_shifted = _gaussian_smooth_reflect_batch_inner(cross_shifted, kernel, pad_x, pad_y)

    smoothed = jax.vmap(jnp.fft.ifftshift)(smoothed_shifted)

    # Build full symmetric covariance tensor
    cov_tensor = jnp.zeros((*t_map.shape[:2], n_freq, n_freq))
    cov_tensor = cov_tensor.at[:, :, idx[0], idx[1]].set(smoothed.transpose(1, 2, 0))
    cov_tensor = cov_tensor.at[:, :, idx[1], idx[0]].set(smoothed.transpose(1, 2, 0))

    return cov_tensor


def _gaussian_smooth_reflect_batch_inner(arr_batch, kernel, pad_x, pad_y):
    """Inner smooth function for use inside JIT (no static_argnums needed)."""
    def smooth_one(arr):
        padded = jnp.pad(arr, ((pad_x, pad_x), (pad_y, pad_y)), mode='symmetric')
        conv = jax.scipy.signal.fftconvolve(padded, kernel, mode='same')
        return conv[pad_x:-pad_x, pad_y:-pad_y]
    return jax.vmap(smooth_one)(arr_batch)


def _get_cov_and_inv_anisotropic_gaussian_fast(t_map, pix, mask, cov_kernel_shape):
    """Fused cov + inv for anisotropic_gaussian: avoids separate get_cov call."""
    t_map_jnp = jnp.asarray(t_map)
    mask_jnp = jnp.asarray(mask) if mask is not None else jnp.ones(t_map_jnp.shape[:2])

    sigma_x, sigma_y = float(cov_kernel_shape[0]), float(cov_kernel_shape[1])
    kernel = _make_gaussian_kernel_2d(sigma_x, sigma_y)
    pad_x = kernel.shape[0] // 2
    pad_y = kernel.shape[1] // 2

    fft_norm = jnp.float64(jnp.sqrt((pix.dx * pix.dy) / (pix.nx * pix.ny)))

    return _cov_and_inv_aniso_gaussian_jit(t_map_jnp, mask_jnp, fft_norm, kernel, pad_x, pad_y)


@functools.partial(jax.jit, static_argnums=(4, 5))
def _cov_and_inv_aniso_gaussian_jit(t_map, mask, fft_norm, kernel, pad_x, pad_y):
    """JIT-compiled fused cov + inv."""
    cov = _cov_aniso_gaussian_jit(t_map, mask, fft_norm, kernel, pad_x, pad_y)
    return jnp.linalg.inv(cov)


def get_anisotropic_ps(map1,map2,pix,type="boxcar",mask=None,cov_kernel_shape=None,ps_isotropic=None):

        map1_np = np.asarray(map1)
        map2_np = np.asarray(map2)
        mask_np = np.asarray(mask) if mask is not None else np.ones_like(map1_np)

        power_spectrum_val = np.real(np.conjugate(maps.get_fft(jnp.array(map1_np*mask_np),pix))*maps.get_fft(jnp.array(map2_np*mask_np),pix))
        power_spectrum_val = np.asarray(power_spectrum_val)

        if ps_isotropic is not None:

            power_spectrum_val = maps.reshape_ell_matrix(power_spectrum_val, ps_isotropic.shape)
            power_spectrum_val = np.asarray(power_spectrum_val)
            power_spectrum_val = power_spectrum_val/ps_isotropic

        if type == "anisotropic_boxcar" or type == "mixed_boxcar":

            kernel = np.ones(cov_kernel_shape)/(cov_kernel_shape[0]*cov_kernel_shape[1])
            power_spectrum_val = np.fft.ifftshift(ndimage.convolve(np.fft.fftshift(power_spectrum_val),kernel,mode='reflect')).real

        elif type == "anisotropic_gaussian" or type == "mixed_gaussian":

            gaussian_sigma = np.array(cov_kernel_shape)
            power_spectrum_val = np.fft.ifftshift(ndimage.gaussian_filter(np.fft.fftshift(power_spectrum_val),sigma=gaussian_sigma,mode="reflect")).real

        else:

            print("Power spectrum estimation type not supported")
            power_spectrum_val = None

        if ps_isotropic is not None:

            power_spectrum_val = power_spectrum_val*ps_isotropic

        return power_spectrum_val


def interp_cov(ell_interp,ell,cov_tensor,interp_type="nearest"):

    """Interpolate binned cross-spectra onto the 2D ell map.

    Mirrors szifi's scipy.interpolate.interp1d(kind=interp_type, bounds_error=False,
    fill_value="extrapolate"). "nearest" and "linear" have JAX implementations that reproduce
    scipy's tie-breaking and extrapolation; any other kind falls back to scipy so that the
    requested interpolation is never silently replaced by a different one."""

    ell_jnp = jnp.asarray(ell)
    cov_jnp = jnp.asarray(cov_tensor)
    ell_interp_jnp = jnp.asarray(ell_interp)

    if interp_type == "nearest":

        return _interp_cov_jit(ell_interp_jnp, ell_jnp, cov_jnp)

    elif interp_type == "linear":

        return _interp_cov_linear_jit(ell_interp_jnp, ell_jnp, cov_jnp)

    else:

        ell_np = np.asarray(ell)
        cov_np = np.asarray(cov_tensor)
        ell_interp_np = np.asarray(ell_interp)

        (d1,d2,d3) = cov_np.shape
        (nx,ny) = ell_interp_np.shape
        interp_tensor = np.zeros((nx,ny,d2,d3))

        for i in range(0,d2):

            for j in range(0,d3):

                interp_tensor[:,:,i,j] = interpolate.interp1d(ell_np,cov_np[:,i,j],
                kind=interp_type,bounds_error=False,fill_value="extrapolate")(ell_interp_np)

        return jnp.asarray(interp_tensor)


def get_camb_cltt(exp=None,freq=0,freq2=None,beams="gaussian"):

    ell,cl_tt = np.load("camb_cl_tt.npy",allow_pickle=True)
    cl_tt = cl_tt*2.*np.pi/(ell*(ell+1.))
    cl_tt[0] = 0.

    if exp is not None:

        if freq2 is None:

            freq2 = freq

        FWHM = exp.FWHM

        if beams == "gaussian":

            cl_tt *= np.asarray(maps.get_bl(FWHM[freq],ell))*np.asarray(maps.get_bl(FWHM[freq2],ell))

        elif beams == "real":

            beam_i = exp.get_beam(freq)[1]
            beam_j = exp.get_beam(freq2)[1]
            ell_beam = exp.get_beam(freq)[0]
            cl_tt *= np.interp(ell,ell_beam,beam_i)*np.interp(ell,ell_beam,beam_j)

    return ell,cl_tt

def bin_spec(ell,cl,bins_low,bins_high):

    cl_binned = np.zeros(len(bins_low))

    for i in range(0,len(bins_low)):

        indices = np.where((ell < bins_high[i]) & (ell >= bins_low[i]))[0]
        cl_binned[i] = np.mean(cl[indices])

    ell_eff = 0.5*(bins_low+bins_high)

    return ell_eff,cl_binned


def correct_for_inpainting(input_spectrum,mask_point,lsep=3000):

    [ell,cl] = input_spectrum
    fsky = np.sum(mask_point)/(mask_point.shape[0]*mask_point.shape[1])
    cl_corrected = np.zeros(len(ell))
    indices_low = np.where(ell>lsep)[0]
    indices_high = np.where(ell<=lsep)[0]
    cl_corrected[indices_low] = cl[indices_low]/fsky
    cl_corrected[indices_high] = cl[indices_high]

    return (ell,cl_corrected)

class ps_flat_sky:

    def __init__(self,pix,mask,ell_bins_edges=None,fac=4):

        self.pix = pix
        self.mask = np.asarray(mask)
        Lx = pix.nx*pix.dx
        Ly = pix.ny*pix.dy

        if ell_bins_edges == None:

            l0_bins = np.arange(pix.nx/fac)*fac*np.pi/(pix.nx*pix.dx)
            lf_bins = (np.arange(pix.nx/fac)+1)*fac*np.pi/(pix.nx*pix.dx)
            bins = nmt.NmtBinFlat(l0_bins, lf_bins)
            ells_uncoupled = bins.get_effective_ells()
            self.bins = bins

        f0 = nmt.NmtFieldFlat(Lx,Ly,self.mask,[np.zeros((self.pix.nx,self.pix.ny))])
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f0,f0,self.bins)
        self.w00 = w00
        self.ells_uncoupled = ells_uncoupled

    def get_ps(self,map,decouple=True):

        map_np = np.asarray(map)
        f1 = nmt.NmtFieldFlat(self.pix.nx*self.pix.dx,self.pix.ny*self.pix.dy,self.mask,[map_np])
        cl00_coupled = nmt.compute_coupled_cell_flat(f1,f1,self.bins)

        if decouple == True:

            cl_uncoupled = self.w00.decouple_cell(cl00_coupled)[0]

        return self.ells_uncoupled,cl_uncoupled
