import jax
import jax.numpy as jnp
import numpy as np
import functools
from numba import njit as numba_njit
from scipy import interpolate
import scipy.signal as sp
import pymaster as nmt
import scipy.signal as sg
import healpy as hp
import scipy
import scipy.stats as st
from pixell import enmap, utils
from scipy.ndimage import label, distance_transform_edt

#Functions for handling maps

class pixel:

    def __init__(self,nx,dx,ny=None,dy=None):

        if ny == None:

            ny = nx

        if dy == None:

            dy = dx

        self.nx = nx
        self.dx = dx
        self.ny = ny
        self.dy = dy

    def get_dx_phys(self,cosmology,z):

        return cosmology.angular_diameter_distance(z).value*self.dx  #in Mpc

    def get_dy_phys(self,cosmology,z):

        return cosmology.angular_diameter_distance(z).value*self.dy  #in Mpc

# Module-level cache for ell maps, keyed by pixel dimensions. Bounded so that a run over
# many distinct tile geometries cannot accumulate device arrays without limit.
_ell_cache = {}
_ELL_CACHE_MAX = 8


def _ell_cache_store(key, ell):

    if len(_ell_cache) >= _ELL_CACHE_MAX:

        _ell_cache.pop(next(iter(_ell_cache)))

    _ell_cache[key] = ell

    return ell

class rmap:

    def __init__(self,pix,map_value=None,):

        self.nx = pix.nx
        self.dx = pix.dx
        self.ny = pix.ny
        self.dy = pix.dy
        self.pix = pix

        if map_value is None:

            map_value = jnp.zeros((self.nx,self.ny))

        self.map_value = map_value

    def get_x_coord_map_wrt_centre(self,theta_x=0.):#actually y coordinate, theta_x in radians

        x_coord = (jnp.arange(0,self.nx)-self.nx*0.5+0.5)*self.dx + theta_x
        x_coord_map = jnp.transpose(jnp.tile(x_coord,(self.ny,1)))

        return x_coord_map

    def get_y_coord_map_wrt_centre(self,theta_y=0.):

        y_coord = (jnp.arange(0,self.ny)-self.ny*0.5+0.5)*self.dy + theta_y
        y_coord_map = jnp.tile(y_coord,(self.nx,1))

        return y_coord_map

    def get_x_coord_map_wrt_origin(self):#actually y coordinate

        x_coord = (jnp.arange(0,self.nx)+0.5)*self.dx
        x_coord_map = jnp.transpose(jnp.tile(x_coord,(self.ny,1)))

        return x_coord_map

    def get_y_coord_map_wrt_origin(self):

        y_coord = (jnp.arange(0,self.ny)+0.5)*self.dy
        y_coord_map = jnp.tile(y_coord,(self.nx,1))

        return y_coord_map

    def get_distance_map_wrt_centre(self,theta_misc=[0.,0.]):

        (theta_x,theta_y) = theta_misc

        return jnp.sqrt(self.get_x_coord_map_wrt_centre(theta_x)**2+self.get_y_coord_map_wrt_centre(theta_y)**2)

    def get_fft(self):

        return get_fft(self.map_value,self.pix)

    def get_ifft(self):

        return get_ifft(self.map_value,self.pix)

    #This works for CAR maps too

    def get_lxly(self):

        #Built in NumPy, then moved to the device. See get_ell for why this must not be a
        #device computation.
        lx,ly = np.meshgrid(np.fft.fftfreq(self.nx,self.dx)*2.*np.pi,np.fft.fftfreq(self.ny,self.dy)*2.*np.pi)
        lx = lx.transpose()
        ly = ly.transpose()

        return jnp.asarray(lx),jnp.asarray(ly)

    def get_ell(self):
        """|ell| for every Fourier mode, computed in NumPy and then moved to the device.

        This must not be a device computation. The power-spectrum bin centres sit at odd
        multiples of the fundamental mode and their midpoints at even multiples, so a large
        set of ell values -- every mode on the lx or ly axis, among others -- falls exactly
        on a bin midpoint. interp_cov snaps each mode to its nearest bin, so at those modes
        the assignment is a tie, and a ~1e-12 difference between the CPU and GPU evaluations
        of sqrt(lx^2 + ly^2) breaks the tie the other way. That shifts the covariance at
        those modes by a whole bin (~2e4 out of ~1e6 here), which propagates to ~1e-4 in the
        filter normalisations and ~3e-3 in the recovered SNR. Building ell on the host makes
        it bit-identical to szifi's, so the ties break the same way."""
        key = (self.nx, self.ny, self.dx, self.dy)
        if key in _ell_cache:
            return _ell_cache[key]
        lx, ly = self.get_lxly()
        ell = jnp.asarray(np.sqrt(np.asarray(lx)**2 + np.asarray(ly)**2))
        return _ell_cache_store(key, ell)

    def convolve_gaussian(self,fwhm_arcmin):

        return get_gaussian_convolution(self.map_value,fwhm_arcmin,self.pix)

    def filter_lrange(self,lrange):

        map_filtered = self.map_value.reshape(self.pix.nx*self.pix.ny)
        map_filtered = map_filtered.reshape((self.pix.nx,self.pix.ny))

        return map_filtered


class tmap:

    def __init__(self,pix,n_freq,map_value=None):

        self.nx = pix.nx
        self.dx = pix.dx
        self.ny = pix.ny
        self.dy = pix.dy
        self.n_freq = n_freq
        self.pix = pix

        if map_value is None:

            map_value = jnp.zeros((self.nx,self.ny,self.n_freq))

        self.map_value = map_value

    def get_x_coord_map_wrt_centre(self,theta_x=0.):

        x_coord = (jnp.arange(0,self.nx)-self.nx*0.5+0.5)*self.dx + theta_x
        x_coord_map = jnp.tile(x_coord,(self.nx,1))

        return x_coord_map

    def get_y_coord_map_wrt_centre(self,theta_y=0.):

        y_coord = (jnp.arange(0,self.ny)-self.ny*0.5+0.5)*self.dy + theta_y
        y_coord_map = jnp.tile(y_coord,(self.ny,1))

        return jnp.transpose(y_coord_map)

    def get_distance_map_wrt_centre(self,theta_misc=[0.,0.]):

        (theta_x,theta_y) = theta_misc

        return jnp.sqrt(self.get_x_coord_map_wrt_centre(theta_x)**2+self.get_y_coord_map_wrt_centre(theta_y)**2)

    def get_fft(self):

        return get_fft_f(self.map_value,self.pix)

    def get_ifft(self):

        return get_ifft_f(self.map_value,self.pix)

    def get_lxly(self):

        lx,ly = np.meshgrid(np.fft.fftfreq(self.ny,self.dy)*2.*np.pi,np.fft.fftfreq(self.nx,self.dx)*2.*np.pi)

        return jnp.asarray(lx),jnp.asarray(ly)

    def get_ell(self):
        """See rmap.get_ell: built on the host so that modes sitting exactly on a bin
        midpoint snap to the same bin as in szifi."""
        key = (self.nx, self.ny, self.dx, self.dy)
        if key in _ell_cache:
            return _ell_cache[key]
        lx, ly = self.get_lxly()
        ell = jnp.asarray(np.sqrt(np.asarray(lx)**2 + np.asarray(ly)**2))
        return _ell_cache_store(key, ell)

    def convolve_gaussian(self,fwhm_arcmin_vec):

        fmap_convolved = jnp.zeros((self.nx,self.ny,self.n_freq))

        for i in range(0,self.n_freq):

            fmap_convolved = fmap_convolved.at[:,:,i].set(get_gaussian_convolution(self.map_value[:,:,i],fwhm_arcmin_vec[i],self.pix))

        return fmap_convolved

    def filter_lrange(self,lrange):

        tmap_filtered = jnp.zeros(self.map_value.shape)

        for i in range(0,self.map_value.shape[2]):

            tmap_filtered = tmap_filtered.at[:,:,i].set(rmap(self.pix,map_value=self.map_value[:,:,i]).filter_lrange(lrange))

        return tmap_filtered


def get_fft(map_value,pix):

    return jnp.fft.fft2(map_value)*jnp.sqrt((pix.dx*pix.dy)/(pix.nx*pix.ny))#same convention as quicklens

def get_ifft(map_value,pix):

    return jnp.fft.ifft2(map_value)*jnp.sqrt((pix.nx*pix.ny)/(pix.dx*pix.dy)) #same convention as quicklens

def fftconvolve(map1,map2,pix):

    ret = get_ifft(get_fft(map1,pix)*get_fft(map2,pix),pix).real
    # Shift by nx//2, ny//2 — functional replacement for in-place loop
    ret = jnp.roll(ret, (pix.nx//2, pix.ny//2), axis=(0, 1))

    return ret


def get_fft_f(tmap_val,pix):
    norm = jnp.sqrt((pix.dx*pix.dy)/(pix.nx*pix.ny))
    # vmap over freq axis (axis 2)
    return jax.vmap(lambda m: jnp.fft.fft2(m) * norm, in_axes=2, out_axes=2)(tmap_val)

def get_ifft_f(tmap_val,pix):
    norm = jnp.sqrt((pix.nx*pix.ny)/(pix.dx*pix.dy))
    return jax.vmap(lambda m: jnp.fft.ifft2(m) * norm, in_axes=2, out_axes=2)(tmap_val)

def filter_fft(map_fft,pix,ell_filter, indices_filter=None):

    if indices_filter is None:

        [lmin,lmax] = ell_filter
        ell_map = rmap(pix).get_ell()
        mask = (ell_map >= lmin) & (ell_map <= lmax)
        return map_fft * mask

    else:
        # Convert indices to boolean mask and multiply — much faster than .at[].set()
        mask = jnp.ones(map_fft.shape, dtype=map_fft.real.dtype)
        mask = mask.at[indices_filter].set(0.)
        return map_fft * mask

def filter_fft_f(map_fft,pix,ell_filter,indices_filter=None,filter_mask=None):

    if filter_mask is not None:
        # Fast path: use precomputed boolean mask
        return map_fft * filter_mask[:,:,None]
    elif indices_filter is None:
        [lmin,lmax] = ell_filter
        ell_map = rmap(pix).get_ell()
        mask = (ell_map >= lmin) & (ell_map <= lmax)
        return map_fft * mask[:,:,None]
    else:
        # Convert indices to boolean mask and multiply — much faster than .at[].set()
        mask = jnp.ones(map_fft.shape[:2], dtype=map_fft.real.dtype)
        mask = mask.at[indices_filter[0], indices_filter[1]].set(0.)
        return map_fft * mask[:,:,None]

def filter_tmap(tmap_val,pix,ell_filter,indices_filter=None,filter_mask=None):
    dtype = tmap_val.dtype
    result = get_ifft_f(filter_fft_f(get_fft_f(tmap_val,pix),pix,ell_filter,indices_filter=indices_filter,filter_mask=filter_mask),pix).real
    return jnp.asarray(result, dtype=dtype)


@jax.jit
def _filter_tmaps_jit(tmaps, filter_mask, fft_norm, ifft_norm):
    """tmaps: (n_tile, nx, ny, n_freq), filter_mask: (nx, ny)."""
    t_freq = jnp.moveaxis(tmaps, -1, 1)
    fft = jnp.fft.fft2(t_freq) * fft_norm
    fft = fft * filter_mask[None, None, :, :]
    out = jnp.fft.ifft2(fft).real * ifft_norm
    return jnp.moveaxis(out, 1, -1).astype(tmaps.dtype)


def filter_tmaps(tmaps, pix, filter_mask):
    fft_norm = jnp.sqrt((pix.dx * pix.dy) / (pix.nx * pix.ny))
    ifft_norm = jnp.sqrt((pix.nx * pix.ny) / (pix.dx * pix.dy))
    return _filter_tmaps_jit(jnp.asarray(tmaps), jnp.asarray(filter_mask),
                              fft_norm, ifft_norm)


def filter_map(map_val,pix,ell_filter):

    return get_ifft(filter_fft(get_fft(map_val,pix),pix,ell_filter),pix).real

def filter_cov(cov,pix,ell_filter):

    [lmin,lmax] = ell_filter
    ell_map = rmap(pix).get_ell()
    mask = (ell_map >= lmin) & (ell_map <= lmax)
    return cov * mask[:,:,None,None]

def get_gaussian_convolution(map_value,fwhm_arcmin,pix):

    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = get_bl(fwhm_arcmin,ell)
    map_fft_convolved = get_ifft(map_fft*beam_fft,pix).real

    return map_fft_convolved

def get_gaussian_convolution_fft(map_fft,fwhm_arcmin,pix):

    ell = rmap(pix).get_ell()
    beam_fft = get_bl(fwhm_arcmin,ell)
    map_fft_convolved = map_fft*beam_fft

    return map_fft_convolved

def get_gaussian_deconvolution(map_value,fwhm_arcmin,pix):

    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = get_bl(fwhm_arcmin,ell)
    map_fft_convolved = get_ifft(map_fft/beam_fft,pix).real

    return map_fft_convolved

def get_convolution_isotropic(map_value,kernel,pix):

    ell_kernel,fft_kernel = kernel
    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = jnp.interp(ell,jnp.array(ell_kernel),jnp.array(fft_kernel))
    map_fft_convolved = get_ifft(map_fft*beam_fft,pix).real

    return map_fft_convolved

def get_convolution_isotropic_fft(map_fft,kernel,pix):

    ell_kernel,fft_kernel = kernel
    ell = rmap(pix).get_ell()
    beam_fft = jnp.interp(ell,jnp.array(ell_kernel),jnp.array(fft_kernel))
    map_fft_convolved = map_fft*beam_fft

    return map_fft_convolved


def get_deconvolution_isotropic(map_value,kernel,pix):

    ell_kernel,fft_kernel = kernel
    map_fft = get_fft(map_value,pix)
    ell = rmap(pix).get_ell()
    beam_fft = jnp.interp(ell,jnp.array(ell_kernel),jnp.array(fft_kernel))
    map_fft_convolved = get_ifft(map_fft/beam_fft,pix).real

    return map_fft_convolved

def convolve(map1,map2,pix):

    return get_ifft(get_fft(map1,pix)*get_fft(map2,pix),pix).real

def get_bl(fwhm_arcmin,ell):

    return jnp.exp(-(fwhm_arcmin*jnp.pi/180./60.)**2/(16.*jnp.log(2.))*ell*(ell+1.))

#theta in rad

def get_beam_real(fwhm_arcmin,theta):

    sigma_arcmin = fwhm_arcmin/(2.*jnp.sqrt(2.*jnp.log(2)))

    return eval_gaussian(theta,sigma_arcmin/60./180.*jnp.pi)

def eval_gaussian(x,sigma):

    return jnp.exp(-0.5*(x/sigma)**2) / (sigma * jnp.sqrt(2.*jnp.pi))

def eval_beam_real_map(pix,fwhm_arcmin):

    theta_map = rmap(pix).get_distance_map_wrt_centre()

    return get_beam_real(fwhm_arcmin,theta_map)

def get_bl_vec(fwhm_arcmin,lmax):

    ls = jnp.arange(0,lmax+1)

    return (ls,get_bl(fwhm_arcmin,ls))

#Beam-deconvolved noise power spectrum

def get_nl(n_lev,fwhm_arcmin,ell):

    return (n_lev*jnp.pi/180./60.)**2/get_bl(fwhm_arcmin,ell)**2

def get_nl_vec(n_lev,fwhm_arcmin,lmax):

    ls = jnp.arange(0,lmax+1)

    return (ls,get_nl(n_lev,fwhm_arcmin,ls))


def rfft2_to_fft2(pix,rfft):

    ny = pix.ny
    fft = jnp.zeros((pix.nx,pix.ny),dtype=complex)
    fft = fft.at[:,0:(ny//2+1)].set(rfft[:,:])
    fft = fft.at[0,(ny//2+1):].set(jnp.conj(rfft[0,1:ny//2][::-1]))
    fft = fft.at[1:,(ny//2+1):].set(jnp.conj(rfft[1:,1:ny//2][::-1,::-1]))

    return fft

def resample_fft(d,n,axes=None):
        """Resample numpy array d via fourier-reshaping. Requires periodic data.
        This function borrowed from Sigurd Naess' pixell,
        Copyright (c) 2018-2021, Members of the Simons Observatory Collaboration"""

        d = np.asanyarray(d)
        n = np.atleast_1d(n)

        if axes is None: axes = np.arange(-len(n),0)
        else: axes = np.atleast_1d(axes)

        if len(n) == 1: n = np.repeat(n, len(axes))
        else: assert len(n) == len(axes)

        assert len(n) <= d.ndim

        if np.all(d.shape[-len(n):] == n): return d

        if d.ndim == 2 and len(n) == 1 and (axes[0] == 1 or axes[0] == -1):
                return resample_fft_simple(d, n[0])

        fd = np.fft.fftn(d, axes=axes)
        norm = 1.0

        for ax, nnew in zip(axes, n):
                ax %= d.ndim
                nold = d.shape[ax]
                dn   = nnew-nold

                if dn > 0:
                        padvals = np.zeros(fd.shape[:ax]+(dn,)+fd.shape[ax+1:],fd.dtype)
                        spre  = tuple([slice(None)]*ax+[slice(0,nold//2)]+[slice(None)]*(fd.ndim-ax-1))
                        spost = tuple([slice(None)]*ax+[slice(nold//2,None)]+[slice(None)]*(fd.ndim-ax-1))
                        fd = np.concatenate([fd[spre],padvals,fd[spost]],axis=ax)

                elif dn < 0:
                        spre  = tuple([slice(None)]*ax+[slice(0,nnew//2)]+[slice(None)]*(fd.ndim-ax-1))
                        spost = tuple([slice(None)]*ax+[slice(nnew//2-dn,None)]+[slice(None)]*(fd.ndim-ax-1))
                        fd = np.concatenate([fd[spre],fd[spost]],axis=ax)

                norm *= float(nnew)/nold

        res  = np.fft.ifftn(fd, axes=axes, norm='backward')
        del fd
        res *= norm

        return res if np.issubdtype(d.dtype, np.complexfloating) else res.real

def resample_fft_simple(d,n,ngroup=100):
        """Resample 2d numpy array d via fourier-reshaping along last axis.
        This function borrowed from Sigurd Naess' pixell,
        Copyright (c) 2018-2021, Members of the Simons Observatory Collaboration"""

        nold = d.shape[1]
        if n == nold: return d

        res  = np.zeros([d.shape[0],n],dtype=d.dtype)
        dn   = n-nold

        for di in range(0, d.shape[0], ngroup):
                fd = np.fft.fftn(d[di:di+ngroup])
                if n < nold:
                        fd = np.concatenate([fd[:,:n//2],fd[:,n//2-dn:]],1)
                else:
                        fd = np.concatenate([fd[:,:nold//2],np.zeros([len(fd),n-nold],fd.dtype),fd[:,nold//2:]],-1)
                res[di:di+ngroup] = np.fft.ifftn(fd, norm='backward').real

        del fd
        res *= float(n)/nold

        return res

def get_target_wcs(dx_deg, field_shape, center_radec_deg):

    from pixell import wcsutils
    wcs = wcsutils.explicit(ctype=["RA---CAR", "DEC--CAR"], cdelt=[-dx_deg, dx_deg],
                            crpix=np.array(field_shape)[::-1]/2, crval=center_radec_deg)
    return wcs

def get_newshape_lmax1d(shape, lmax1d, dx_rad, powerOfTwo=False):

    if len(shape) > 2:
        raise ValueError("Expected 2-tuple shape")

    newdx = np.pi / lmax1d

    if newdx <= dx_rad:
        return shape

    extent = np.array(shape) * dx_rad
    new_shape = np.ceil(extent / newdx).astype(int)

    if powerOfTwo:
        new_shape = (2**np.ceil(np.log2(new_shape))).astype(int)

    return tuple(new_shape)

def degrade_map(arr, new_shape, deg_axes=[0,1]):

    return resample_fft(arr, new_shape, deg_axes)

def degrade_pix(pix, new_shape):

    if new_shape is None:
        return pix

    nx, ny = new_shape
    new_dx = pix.dx * (pix.nx / nx)
    new_dy = pix.dy * (pix.ny / ny)
    deg_pix = pixel(nx, new_dx, ny, new_dy)

    return deg_pix

def reshape_ell_matrix(arr, new_shape, axes=[0,1]):
    """
    Zero fill or cut arr in the given axes to reach new_shape.
    Splits into quadrants and adds zeros (or removes vals) in the central cross.
    """
    # Fast path: if shapes already match, return as-is (avoid GPU->CPU->GPU copy)
    if all(arr.shape[ax] == ns for ax, ns in zip(axes, new_shape)):
        return jnp.asarray(arr) if not isinstance(arr, jnp.ndarray) else arr

    # Convert JAX array to numpy for concatenation with dynamic shapes
    arr_np = np.asarray(arr)

    for ax, nnew in zip(axes, new_shape):

        ax %= arr_np.ndim
        nold = arr_np.shape[ax]
        dn = nnew - nold

        if dn < 0:
            spre  = tuple([slice(None)]*ax+[slice(0,nnew//2)]+[slice(None)]*(arr_np.ndim-ax-1))
            spost = tuple([slice(None)]*ax+[slice(nnew//2-dn,None)]+[slice(None)]*(arr_np.ndim-ax-1))
            arr_np = np.concatenate([arr_np[spre], arr_np[spost]], axis=ax)

        elif dn == 0:
            continue

        else:
            padvals = np.zeros(arr_np.shape[:ax]+(dn,)+arr_np.shape[ax+1:],arr_np.dtype)
            spre  = tuple([slice(None)]*ax+[slice(0,nold//2)]+[slice(None)]*(arr_np.ndim-ax-1))
            spost = tuple([slice(None)]*ax+[slice(nold//2,None)]+[slice(None)]*(arr_np.ndim-ax-1))
            arr_np = np.concatenate([arr_np[spre],padvals,arr_np[spost]],axis=ax)

    return jnp.asarray(arr_np)

def get_noise_map(pix,n_lev): #white noise

    sigma_pixel = n_lev/(180.*60./np.pi*pix.dx)

    return get_noise_map_sigma_pix(pix,sigma_pixel)


def get_noise_tmap(pix,n_lev_vec): #generates white noise frequency maps

    tmap_val = np.zeros((pix.nx,pix.ny,len(n_lev_vec)))

    for i in range(0,len(n_lev_vec)):

        tmap_val[:,:,i] = get_noise_map(pix,n_lev_vec[i])

    return jnp.array(tmap_val)

def get_noise_tmap_exp(pix,exp,ptf_flag=False):

    noise = get_noise_tmap(pix,exp.noise_levels)

    if ptf_flag == True:

        for i in range(0,exp.n_freqs):

            noise = noise.at[:,:,i].set(get_convolution_isotropic(noise[:,:,i],exp.get_ptf(),pix))

    return noise

def get_noise_map_sigma_pix(pix,sigma_pixel):

    return np.random.standard_normal((pix.nx,pix.ny))*sigma_pixel

def get_nlev_from_sigma_pixel(pix,sigma_pixel):

    return sigma_pixel*(180.*60./np.pi*np.sqrt(pix.dx*pix.dy))


def get_cl_noise(ell,my_sz_data):

    return jnp.interp(ell,my_sz_data.l_eff,my_sz_data.cl_all)

def cl_to_map(ell,cl,pix,lmax=None):

    ell_map = rmap(pix).get_ell()
    ret = jnp.interp(ell_map,jnp.array(ell),jnp.array(cl))

    if lmax != None:

        ret = jnp.where(ell_map > lmax, 0., ret)

    return ret

def cl_binned_to_map(ell_bin,cl_bin,pix):

    ell = jnp.arange(ell_bin[0],ell_bin[-1])
    cl = jnp.interp(ell,jnp.array(ell_bin),jnp.array(cl_bin))
    ell_map = rmap(pix).get_ell()

    return jnp.interp(ell_map,ell,cl)


def get_noise_cl_map_gaussian(pix,n_lev,fwhm_arcmin):

    ell_map = rmap(pix).get_ell()
    noise = get_nl(n_lev,fwhm_arcmin,ell_map)

    return noise

def get_cross_spectrum(pix,map1,map2=None,ell=None): #deprecated

    if ell is None:
        ell = rmap(pix).get_ell()
    if map2 is None:
        map2 = map1

    map1 = get_fft(map1,pix).flatten()
    map2 = get_fft(map2,pix).flatten()
    ell = ell.flatten()
    idx_sort = jnp.argsort(ell)
    map1_sorted = map1[idx_sort]
    map2_sorted = map2[idx_sort]
    ell_sorted = ell[idx_sort]
    ell_new, idx_start, count = jnp.unique(ell_sorted, return_counts=True, return_index=True, size=len(ell_sorted))
    cross_spec = jnp.zeros(len(ell_new))

    for i in range(0,len(ell_new)):
        mapps1 = map1_sorted[idx_start[i]:idx_start[i]+count[i]]
        mapps2 = map2_sorted[idx_start[i]:idx_start[i]+count[i]]
        cross_spec = cross_spec.at[i].set(jnp.mean(mapps1*jnp.conjugate(mapps2)).real)

    return (ell_new,cross_spec)

def clone_map_freq(map1,n_freq):

    return jnp.broadcast_to(map1[:,:,None], (map1.shape[0], map1.shape[1], n_freq)).copy()

def multiply_t(map1, map2):
    """Add an axis to 'map1' and multiply it by 'map2'"""
    if map1.ndim > map2.ndim:
        map1, map2 = map2, map1
    imap = jnp.expand_dims(map1, map1.ndim) # Add a len-1 axis for compatible shapes
    return imap * map2

def get_mask_apod(pix,alpha=0.1):

    line1 = sp.tukey(pix.nx,alpha)
    line2 = sp.tukey(pix.ny,alpha)

    return jnp.array(np.outer(line1,line2))

def get_mask_apod_freq(pix,n_freq=6,alpha=0.1):

    mask = jnp.zeros((pix.nx,pix.ny,n_freq))

    for i in range(0,n_freq):

        mask = mask.at[:,:,i].set(get_mask_apod(pix,alpha=alpha))

    return mask

def get_mask_square(nx,buffer,ny=None):

    if ny is None:
        ny = nx

    mask = np.zeros((nx,ny))
    mask[buffer:nx-buffer,buffer:ny-buffer] = 1.

    return jnp.array(mask)

def get_theta_cart(theta_misc,pix):

    theta_x_misc,theta_y_misc = theta_misc
    theta_x = pix.nx/2*pix.dx - theta_y_misc
    theta_y = pix.nx/2*pix.dx + theta_x_misc
    theta_cart = [theta_x,theta_y]

    return theta_cart

def get_theta_misc(theta_cart,pix):

    [theta_x,theta_y] = theta_cart
    theta_y_misc = pix.nx/2*pix.dx - theta_x
    theta_x_misc = -pix.nx/2*pix.dx + theta_y
    theta_misc = [theta_x_misc,theta_y_misc]

    return theta_misc

def get_ij_from_theta(theta_x,theta_y,pix):

    j = np.floor(theta_x/pix.dx).astype("int")
    i = (pix.ny-np.ceil(theta_y/pix.dy)).astype("int")

    return i,j

def get_theta_from_ij(i,j,pix):

    theta_x = j*pix.dx
    theta_y = (pix.ny-i)*pix.dy

    return theta_x,theta_y

def get_buffered_mask(pix,mask_input,buffer_arcmin,type="fft",tile_type="healpix",wcs=None):

    buffer_rad = buffer_arcmin/60./180.*np.pi
    buffer_pix_x = int(round(buffer_arcmin/60./180.*np.pi/pix.dx))+1
    buffer_pix_y = int(round(buffer_arcmin/60./180.*np.pi/pix.dy))+1

    nx_padded = pix.nx+4*buffer_pix_x
    ny_padded = pix.ny+4*buffer_pix_y

    mask_input_np = np.asarray(mask_input)
    mask_input_padded = np.ones((nx_padded,ny_padded))
    mask_input_padded[2*buffer_pix_x:nx_padded-2*buffer_pix_x,2*buffer_pix_y:ny_padded-2*buffer_pix_y] = mask_input_np
    pix_padded = pixel(nx_padded,pix.dx,ny=ny_padded,dy=pix.dy)

    kernel = np.zeros((pix_padded.nx,pix_padded.ny))

    if tile_type == "healpix":

        distances = np.asarray(rmap(pix_padded).get_distance_map_wrt_centre())

    elif tile_type == "car":

        from pixell import enmap, utils

        mask_input_padded = enmap.enmap(mask_input_padded,wcs)
        position_map = enmap.empty(mask_input_padded.shape,wcs)
        pos = position_map.posmap()
        dec_map, ra_map = pos
        pix_center = [(mask_input_padded.shape[-2] - 1) / 2, (mask_input_padded.shape[-1] - 1) / 2]
        dec0, ra0 = enmap.pix2sky(mask_input_padded.shape,wcs,pix_center)

        distances = np.asarray(utils.angdist([ra0,dec0], [ra_map,dec_map]))

    indices = np.where(distances <= buffer_rad)
    kernel[indices] = 1.
    kernel_area = np.sum(kernel)

    mask_convolved = np.around(sg.fftconvolve(np.asarray(mask_input_padded),kernel,mode='same')/kernel_area,decimals=4)[2*buffer_pix_x:nx_padded-2*buffer_pix_x,2*buffer_pix_y:ny_padded-2*buffer_pix_y]

    mask_output = np.ones(mask_input_np.shape)
    mask_output[np.where(mask_convolved != 1.)] = 0.

    return jnp.array(mask_output)

def get_buffer_region(pix, mask, buffer_arcmin):

    buffered_mask = get_buffered_mask(pix, mask, buffer_arcmin, type='fft')
    ibuffered_mask = get_buffered_mask(pix, 1-mask, buffer_arcmin, type='fft')
    buffer_region = 1 - buffered_mask - ibuffered_mask

    return buffer_region

#apotype: C1, C2 and Smooth

def get_apodised_mask(pix,mask_input2,apotype="C1",aposcale=0.2):

    mask_input = np.copy(np.asarray(mask_input2))

    mask_input[0,:] = 0.
    mask_input[:,0] = 0.
    mask_input[-1,:] = 0.
    mask_input[:,-1] = 0.

    return jnp.array(nmt.mask_apodization_flat(mask_input,pix.nx*pix.dx,pix.ny*pix.dy,aposcale,apotype=apotype))

def get_fsky_criterion_mask(pix,mask_select,nside_tile,criterion=0.3,tile_type="healpix"):

    if tile_type == "healpix":

        area_tile = hp.pixelfunc.nside2pixarea(nside_tile)

    elif tile_type == "car":

        area_tile = pix.nx*pix.ny*pix.dx*pix.dy

    area_select = jnp.sum(mask_select)*pix.dx*pix.dy
    frac = area_select/area_tile

    if frac < criterion:

        mask_select = jnp.zeros((pix.nx,pix.ny))

    return mask_select

class ps_mask:

    def __init__(self,pix,n_source,r_source_arcmin,tile_type="healpix",wcs=None):

        self.pix = pix
        self.n_source = n_source
        self.r_source_arcmin = r_source_arcmin
        self.tile_type = tile_type
        self.wcs = wcs
        self.r_source = self.r_source_arcmin/60./180.*np.pi

    def get_mask_map(self,source_coords=None):

        mask = np.ones((self.pix.nx,self.pix.ny))

        if self.tile_type == "car":
            from pixell import enmap, utils

        for i in range(0,self.n_source):

            if self.tile_type == "healpix":

                distances = np.asarray(rmap(self.pix).get_distance_map_wrt_centre(theta_misc=source_coords[i,:]))

            elif self.tile_type == "car":

                position_map = enmap.empty(mask.shape,self.wcs)
                pos = position_map.posmap()
                dec_map, ra_map = pos
                distances = np.asarray(utils.angdist([source_coords[i,0],source_coords[i,1]], [ra_map,dec_map]))

            indices = np.where(distances <= self.r_source)
            mask[indices] = 0.

        return jnp.array(mask)

    def get_mask_map_t(self,n_freq,source_coords=None):

        mask_map = self.get_mask_map(source_coords=source_coords)
        return jnp.broadcast_to(mask_map[:,:,None], (self.pix.nx, self.pix.ny, n_freq)).copy()

def mask_sigma(map_val,sigma,inpaint_flag=False,n_inpaint=100):

    map_np = np.asarray(map_val)
    std = np.std(map_np)
    mean = np.mean(map_np)
    indices = np.where(np.abs(map_np)>sigma*std+mean)
    map_np[indices] = 0.

    if inpaint_flag == True:

        mask = np.ones(map_np.shape)
        mask[indices] = 0.
        map_np = np.asarray(diffusive_inpaint(map_np,mask,n_inpaint))

    return jnp.array(map_np)

def mask_sigma_freq(tmap_val,sigma,inpaint_flag=False):

    ret = jnp.zeros(tmap_val.shape)

    for i in range(0,tmap_val.shape[2]):

        ret = ret.at[:,:,i].set(mask_sigma(tmap_val[:,:,i],sigma=sigma,inpaint_flag=inpaint_flag))

    return ret

def extract_value_at_catalogue(tsz,catalogue_tile,pix):

    i,j = get_ij_from_theta(catalogue_tile.theta_x,catalogue_tile.theta_y,pix)

    return tsz[i,j]

def get_binned_mean_map(map_val,pix,bins,fft_flag=False):

    map_real = rmap(pix,map_val)

    if fft_flag == False:
        map_fft = map_real.get_fft()
    else:
        map_fft = map_val

    ell = map_real.get_ell()

    binned_map = np.zeros(len(bins)-1)
    bins_centres = np.zeros(len(binned_map))

    ell_np = np.asarray(ell)
    map_fft_np = np.asarray(map_fft)

    for i in range(0,len(bins_centres)):

        indices = np.where((ell_np > bins[i]) & (ell_np < bins[i+1]))
        selected_map = np.mean(map_fft_np[indices]).real
        binned_map[i] = selected_map
        bins_centres[i] = 0.5*(bins[i] + bins[i+1])

    return bins_centres,binned_map

def convolve_tmap_experiment(pix,tmap_val,exp,freqs=None,beam_type="gaussian",tile_type="healpix",wcs=None):

    n_freqs = tmap_val.shape[2]
    freqs = np.arange(n_freqs)
    tmap_convolved = jnp.zeros(tmap_val.shape)

    for i in range(0,n_freqs):

        if tile_type == "healpix":

            if beam_type == "gaussian":

                tmap_convolved = tmap_convolved.at[:,:,i].set(get_gaussian_convolution(tmap_val[:,:,i],exp.FWHM[freqs[i]],pix))

            elif beam_type == "real":

                tmap_convolved = tmap_convolved.at[:,:,i].set(get_convolution_isotropic(tmap_val[:,:,i],exp.get_beam(freqs[i]),pix))

        elif tile_type == "car":

            from pixell import enmap, enplot

            if beam_type == "gaussian":

                map_freq = enmap.enmap(np.asarray(tmap_val[:,:,i]),wcs)
                sigma = exp.FWHM[freqs[i]]/(2.*np.sqrt(2.*np.log(2)))/60./180.*np.pi
                tmap_convolved = tmap_convolved.at[:,:,i].set(jnp.array(enmap.smooth_gauss(map_freq, sigma)))

            elif beam_type == "real":

                tmap_convolved = None #todo

    return tmap_convolved

def convolve_tmap_fft_experiment(pix,tmap_fft,exp,freqs=None,beam_type="gaussian"):

    n_freqs = tmap_fft.shape[2]

    if freqs is None:

        freqs = np.arange(n_freqs)

    tmap_convolved_fft = jnp.zeros(tmap_fft.shape, dtype=tmap_fft.dtype)

    for i in range(0,n_freqs):

        if beam_type == "gaussian":

            tmap_convolved_fft = tmap_convolved_fft.at[:,:,i].set(get_gaussian_convolution_fft(tmap_fft[:,:,i],exp.FWHM[freqs[i]],pix))

        elif beam_type == "real":

            tmap_convolved_fft = tmap_convolved_fft.at[:,:,i].set(get_convolution_isotropic_fft(tmap_fft[:,:,i],exp.get_beam(freqs[i]),pix))

    return tmap_convolved_fft


def get_tmap_times_fvec(tmap_val,fvec):

    fvec = jnp.asarray(fvec)
    return tmap_val * fvec[None, None, :]

def get_tmap_from_map(map_val,n_freq):

    return jnp.broadcast_to(map_val[:,:,None], (map_val.shape[0], map_val.shape[1], n_freq)).copy()


def select_freqs(tmap_val,freqs):

    return tmap_val[:,:,freqs]

def get_tmap_with_sed(pix,tmap_val,base_freq,exp,sed,beam_type="gaussian"):

    sed = sed/sed[base_freq]
    ret = jnp.zeros(tmap_val.shape)

    deconv_freq = base_freq

    if beam_type == "gaussian":

        spatial_map = get_gaussian_deconvolution(tmap_val[:,:,deconv_freq],exp.FWHM[deconv_freq],pix)

    elif beam_type == "real":

        spatial_map = get_deconvolution_isotropic(tmap_val[:,:,deconv_freq],exp.get_beam(deconv_freq),pix)

    for i in range(0,tmap_val.shape[2]):

        ret = ret.at[:,:,i].set(spatial_map*sed[i])

    ret = convolve_tmap_experiment(pix,ret,exp,freqs=np.arange(len(sed)),beam_type=beam_type)

    return ret

def get_hankel_transform(theta_range,function,n=512,pad=128):

    lrange = [1/theta_range[1], 1/theta_range[0]]

    logl1, logl2 = np.log(lrange)
    logl0        = (logl2+logl1)/2
    dlog    = (logl2-logl1)/n
    i0           = (n+1)/2+pad
    l       = np.exp(logl0 + (np.arange(1,n+2*pad+1)-i0)*dlog)
    r       = 1/l[::-1]
    rprof = function(r)
    lprof = 2*np.pi*scipy.fft.fht(rprof*r,dlog,0)/l

    res = [l,lprof]
    l,lprof = tuple([arr[...,pad:-pad] for arr in res])

    return l,r,lprof,dlog

def bin_radially(pix,map_val,bins_edges=None,nbins=None,theta_misc=[0.,0.],return_n_pixels=False):

    theta_map = np.asarray(rmap(pix).get_distance_map_wrt_centre(theta_misc))

    if bins_edges is None:
        bins_edges = np.linspace(0,np.max(theta_map),nbins)

    bins = np.zeros(len(bins_edges)-1)
    binned_map = np.zeros(len(bins_edges)-1)
    n_pixels = np.zeros(len(bins_edges)-1)

    theta_flattened = theta_map.flatten()
    map_flattened = np.asarray(map_val).flatten()

    for i in range(0,len(bins_edges)-1):

        bins[i] = 0.5*(bins_edges[i+1]+bins_edges[i])
        indices = np.where((theta_flattened < bins_edges[i+1]) & (theta_flattened > bins_edges[i]))
        binned_map[i] = np.mean(map_flattened[indices])
        n_pixels[i] = len(indices[0])

    if return_n_pixels == True:
        ret = bins,binned_map,n_pixels
    else:
        ret = bins,binned_map

    return ret

def get_integrated_map_r(pix,map_val,theta_misc=[0.,0.],equality_type="less_or_equal"):

    # This function is inherently non-JIT-able due to variable indexing
    map_np = np.asarray(map_val)
    integrated_map = np.zeros(map_np.shape)
    theta_map = np.asarray(rmap(pix).get_distance_map_wrt_centre(theta_misc))

    thetas = np.unique(theta_map.flatten())

    for i in range(0,len(thetas)):

        indices = np.where(theta_map == thetas[i])

        if equality_type == "less_or_equal":
            indices_integrated = np.where(theta_map <= thetas[i])
        elif equality_type == "less":
            indices_integrated = np.where(theta_map < thetas[i])

        integrated_map[indices] = np.mean(map_np[indices_integrated])

    return jnp.array(integrated_map)

def repixelise(image,repixelise_factor,type="average"):

    repixelise_factor = int(repixelise_factor)
    image_np = np.asarray(image)
    nx = image_np.shape[0]
    nx_rep = nx//repixelise_factor
    image_repixelised = np.zeros((nx_rep,nx_rep))

    for i in range(0,image_repixelised.shape[0]):
        for j in range(0,image_repixelised.shape[1]):
            if type == "average":
                image_repixelised[i,j] = np.mean(image_np[repixelise_factor*i:(repixelise_factor*i)+repixelise_factor,repixelise_factor*j:(repixelise_factor*j)+repixelise_factor])
            elif type == "central":
                shift = (repixelise_factor-1)//2
                image_repixelised[i,j] = image_np[repixelise_factor*i+shift,repixelise_factor*j+shift]

    return jnp.array(image_repixelised)


def bin_harmonic_map(map_val,pix,ell_bins_edges):

    ell_map = np.asarray(rmap(pix).get_ell())
    ell_bins_centres = (ell_bins_edges[0:-1]+ell_bins_edges[1:])*0.5
    binned_map = np.zeros(len(ell_bins_centres))
    map_np = np.asarray(map_val)

    for i in range(0,len(binned_map)):
        indices = np.where((ell_map.flatten() > ell_bins_edges[i]) & (ell_map.flatten() < ell_bins_edges[i+1]))
        binned_map[i] = np.mean(map_np.flatten()[indices])

    return ell_bins_centres,binned_map

def paint_profile(pix,theta_rad_vec,profile_vec):

    distance_map = rmap(pix).get_distance_map_wrt_centre()

    return jnp.interp(distance_map,jnp.array(theta_rad_vec),jnp.array(profile_vec))

def repixelise_map(pix_in,map_in,pix_out,method="cubic"):

    # Keep as NumPy — uses scipy RegularGridInterpolator
    rmap_in = rmap(pix_in)
    theta_x = np.asarray(rmap_in.get_x_coord_map_wrt_centre()[:,0])
    theta_y = np.asarray(rmap_in.get_y_coord_map_wrt_centre()[0,:])
    interp_fn = interpolate.RegularGridInterpolator((theta_x,theta_y),np.asarray(map_in),bounds_error=False,fill_value=0.,method=method)

    rmap_out = rmap(pix_out)
    theta_x = np.asarray(rmap_out.get_x_coord_map_wrt_centre())
    theta_y = np.asarray(rmap_out.get_y_coord_map_wrt_centre())
    map_out = interp_fn((theta_x,theta_y))

    return jnp.array(map_out)

def get_map_convolved_fft(map_fft_original,pix,freqs,beam_type,mask,lrange,exp):

    a_map_fft = convolve_tmap_fft_experiment(pix,map_fft_original,exp,freqs=freqs,beam_type=beam_type)
    a_map_fft = get_fft_f(get_ifft_f(a_map_fft,pix)*mask,pix)
    a_map_fft = filter_fft_f(a_map_fft,pix,lrange)

    return a_map_fft

def mask_has_holes(mask):
    """True if any pixel is a hole (mask <= 0.5). All-ones masks are a no-op for inpaint."""
    return bool(np.any(np.asarray(mask) <= 0.5))


def inpaint_freq(tmap_val,mask,n_inpaint=None, pix=None, noise=None,inpaint_type="diffusive"):

    if inpaint_type == "diffusive_jax":
        # Vectorised: vmap the Jacobi inpainting over all frequencies at once, on the GPU.
        # Faster than the exact Gauss-Seidel sweep, but not equal to it at finite n_inpaint.
        tmap_jnp = jnp.asarray(tmap_val)
        mask_jnp = jnp.asarray(mask)
        ret = _inpaint_freq_vmap(tmap_jnp, mask_jnp, n_inpaint)
    else:
        ret = jnp.zeros(tmap_val.shape, dtype=tmap_val.dtype)
        noise = [None]*tmap_val.shape[2] if noise is None else noise
        for i in range(0,tmap_val.shape[2]):
            ret = ret.at[:,:,i].set(inpaint(tmap_val[:,:,i],mask,n_inpaint, pix, noise[i],inpaint_type=inpaint_type))

    if isinstance(tmap_val, enmap.ndmap):
        ret = enmap.ndmap(np.asarray(ret), wcs=tmap_val.wcs)

    return ret


@jax.jit
def _inpaint_freq_vmap(tmap_3d, mask, n_inpaint):
    """Vectorized diffusive inpainting over all frequency channels at once."""
    return jax.vmap(lambda img: diffusive_inpaint_jax(img, mask, n_inpaint),
                    in_axes=2, out_axes=2)(tmap_3d)


@functools.partial(jax.jit, static_argnums=(2,))
def _inpaint_freq_tiles_jit(tmaps, masks, n_inpaint):
    """Jacobi inpaint over tiles and frequencies. tmaps (T, nx, ny, nf)."""
    def one_tile(tmap, mask):
        return jax.vmap(lambda img: diffusive_inpaint_jax(img, mask, n_inpaint),
                        in_axes=2, out_axes=2)(tmap)
    return jax.vmap(one_tile)(tmaps, masks)


def inpaint_freq_tiles(tmaps, masks, n_inpaint=100):
    """GPU Jacobi inpaint for a stack of tiles (it1 cluster holes)."""
    return _inpaint_freq_tiles_jit(jnp.asarray(tmaps), jnp.asarray(masks), int(n_inpaint))


def inpaint_freq_tiles_gs(tmaps, masks, n_inpaint=100):
    """Exact Gauss-Seidel inpaint, one tile per thread (tiles are independent).

    Same algorithm as ``inpaint_type="diffusive"``. Numba GS releases the GIL, so
    a thread pool over tiles is faster than a Python loop and does not change
    the pixels. Do not use ``inpaint_freq_tiles`` (Jacobi) when the catalogue
    must match szifi / CNC.
    """
    from concurrent.futures import ThreadPoolExecutor
    import os

    t_list = [np.asarray(t) for t in tmaps]
    m_list = [np.asarray(m) for m in masks]
    n = len(t_list)
    n_inpaint = int(n_inpaint)

    def _one(i):
        if not mask_has_holes(m_list[i]):
            return t_list[i]
        return np.asarray(inpaint_freq(
            t_list[i], m_list[i], n_inpaint=n_inpaint, inpaint_type="diffusive"))

    workers = min(n, max(1, os.cpu_count() or 8))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_one, range(n)))
    return jnp.stack(parts)


def inpaint(image, mask, n_inpaint=None, pix=None, noise=None, inpaint_type="diffusive"):

    if inpaint_type == "diffusive":

        inpainted_image = diffusive_inpaint(image, mask, n_inpaint)

    elif inpaint_type == "diffusive_jax":

        inpainted_image = diffusive_inpaint_jax(jnp.asarray(image), jnp.asarray(mask), n_inpaint)

    elif inpaint_type == "orphics":

        from orphics import maps as omaps

        image_np = np.asarray(image)
        mask_np = np.asarray(mask)

        if not isinstance(image_np, enmap.ndmap):

            shape = (pix.ny, pix.nx)
            wcs = get_target_wcs(np.rad2deg(pix.dx), shape, (0,0))
            image_np = enmap.ndmap(image_np, wcs=wcs)
            mask_np = enmap.ndmap(mask_np, wcs=wcs)

        ivar = omaps.ivar(image_np.shape, image_np.wcs, noise_muK_arcmin=noise) if noise is not None else None
        inpainted_image = jnp.array(omaps.gapfill_edge_conv_flat(image_np, ~mask_np.astype(bool), ivar=ivar, rmin=0.5*utils.arcmin, alpha=-4))

    return inpainted_image

def circular_cosine_blend_adaptive(
    original_map, inpainted_map, mask,
    inner_zero_frac=0., min_radius=0., pad=2
):
    """
    Adaptive region blending with cosine taper.
    Vectorised: one global EDT + per-region radius lookup, no per-region loop.

    szifi computes, for each labelled hole, the EDT of that hole alone inside a padded
    bounding box, i.e. the distance to the nearest pixel not in that hole. The single global
    EDT used here is equal to it: hole regions are 8-connected components, so any raster path
    from a pixel of one hole to a pixel of another must cross a non-hole pixel, which is
    therefore at least as close as the other hole. Hence distance-to-nearest-non-hole equals
    distance-to-nearest-pixel-not-in-this-hole, pixel by pixel.

    `pad` is accepted for signature compatibility and is not needed by the global transform.
    """
    assert 0 <= inner_zero_frac < 1, "inner_zero_frac must be in [0, 1)"

    original_np = np.asarray(original_map)
    inpainted_np = np.asarray(inpainted_map)
    mask_np = np.asarray(mask)

    mask_bin = mask_np > 0.5
    hole_mask = ~mask_bin

    if not np.any(hole_mask):
        return jnp.array(original_np)

    # Label connected hole regions
    structure = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=bool)
    labeled_mask, n_regions = label(hole_mask, structure=structure)

    # Global distance transform: distance from each hole pixel to nearest edge
    dist = distance_transform_edt(hole_mask)

    # Compute per-region radius from area
    # region labels are 1..n_regions
    region_areas = np.bincount(labeled_mask.ravel(), minlength=n_regions+1)
    region_radii = np.sqrt(region_areas / np.pi)
    # region_radii[0] is background (mask=1), ignore

    # szifi skips regions smaller than min_radius, leaving them purely inpainted with no
    # cosine taper. Zeroing their radius reproduces that: the annulus condition
    # 0 < dist <= radius is then never satisfied, so w stays 0 across the region.
    region_radii = np.where(region_radii < min_radius, 0., region_radii)

    # Map each pixel to its region's radius
    radius_map = region_radii[labeled_mask]

    # Compute blend weight globally (inner_zero_frac=0 common case)
    blend_core_radius = inner_zero_frac * radius_map
    annulus_width = radius_map - blend_core_radius
    annulus_width = np.maximum(annulus_width, 1e-30)  # avoid div by zero

    # Cosine blend weight: 1 at edge (dist=0), 0 at center (dist=radius)
    w = np.where(
        hole_mask & (dist > blend_core_radius) & (dist <= radius_map),
        0.5 * (1 + np.cos(np.pi * (dist - blend_core_radius) / annulus_width)),
        0.
    )

    # Blend: w=1 → original, w=0 → inpainted
    blended = np.where(hole_mask, w * original_np + (1 - w) * inpainted_np, original_np)

    return jnp.array(blended)


@jax.jit
def _edt_jfa_jax(hole_mask):
    """Euclidean distance transform by jump flooding, on the GPU.

    hole_mask: (ny, nx) bool array, True inside holes.
    Returns: (ny, nx) distance from each hole pixel to the nearest non-hole pixel.

    NOT exact. Jump flooding is an approximate algorithm: it propagates a single candidate
    seed per pixel, so a pixel whose true nearest seed never reaches it through the halving
    step sequence keeps a farther one. On scattered discs of radius <= 8 px it agrees with
    scipy.ndimage.distance_transform_edt to ~4e-7 px, but on the Planck field 0 point-source
    mask (111 holes, radii up to 20 px) it is wrong for 17680 of 29658 hole pixels, by up to
    0.36 px. That propagates into the cosine blend weights and shifts the recovered SNR at
    the 1e-3 level, so the blending routines use scipy's exact transform instead. Kept as a
    GPU primitive for callers that can accept the approximation; measure before relying on it.
    """
    ny, nx = hole_mask.shape

    # Pad with True (hole) so JFA only finds seeds within original data
    max_dim = max(nx, ny)
    pad = 1
    while pad < max_dim:
        pad *= 2
    pad_size = pad // 2

    padded_mask = jnp.pad(hole_mask, pad_size, mode='constant', constant_values=True)
    pny, pnx = padded_mask.shape

    yy = jnp.arange(pny, dtype=jnp.float32)
    xx = jnp.arange(pnx, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(yy, xx, indexing='ij')

    INF = jnp.float32(pnx + pny)

    # Non-hole pixels are their own seeds
    seed_y = jnp.where(~padded_mask, grid_y, INF)
    seed_x = jnp.where(~padded_mask, grid_x, INF)

    # JFA iterations: largest power of 2 down to 1
    step = 1
    while step < max(pnx, pny):
        step *= 2
    step //= 2

    while step >= 1:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                n_sy = jnp.roll(jnp.roll(seed_y, -dy * step, axis=0), -dx * step, axis=1)
                n_sx = jnp.roll(jnp.roll(seed_x, -dy * step, axis=0), -dx * step, axis=1)
                d2_new = (grid_y - n_sy)**2 + (grid_x - n_sx)**2
                d2_cur = (grid_y - seed_y)**2 + (grid_x - seed_x)**2
                closer = d2_new < d2_cur
                seed_y = jnp.where(closer, n_sy, seed_y)
                seed_x = jnp.where(closer, n_sx, seed_x)
        step //= 2

    dist = jnp.sqrt((grid_y - seed_y)**2 + (grid_x - seed_x)**2)
    return dist[pad_size:pad_size+ny, pad_size:pad_size+nx]


def circular_cosine_blend_adaptive_multi(
    original_3d, inpainted_3d, mask,
    inner_zero_frac=0., min_radius=0.
):
    """
    Vectorized blend for multi-frequency maps (nx, ny, n_freq).
    Computes blend weights once from mask, applies to all frequencies.
    Uses scipy label (fast, ~2ms) + JFA EDT on GPU (~1ms), JAX for blending.
    """
    mask_np = np.asarray(mask)
    mask_bin = mask_np > 0.5
    hole_mask = ~mask_bin

    if not np.any(hole_mask):
        return original_3d

    # Label on CPU (fast C implementation, ~2 ms)
    structure = np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=bool)
    labeled_mask, n_regions = label(hole_mask, structure=structure)

    # Exact EDT on CPU. The GPU jump-flooding version (_edt_jfa_jax) is ~60x faster but only
    # approximate, and its error reaches 0.36 px on the real Planck point-source mask, which
    # moves the blend weights and the recovered SNR at the 1e-3 level. ~63 ms per tile against
    # a ~300 s cluster-finding run is not worth that.
    hole_mask_jnp = jnp.array(hole_mask)
    dist_jnp = jnp.array(distance_transform_edt(hole_mask))

    # Compute region radii from areas
    region_areas = np.bincount(labeled_mask.ravel(), minlength=n_regions + 1)
    region_radii = np.sqrt(region_areas / np.pi)

    # szifi skips regions smaller than min_radius, leaving them purely inpainted with no
    # cosine taper. Setting their radius to 0 reproduces that: the annulus condition
    # 0 < dist <= radius is then never satisfied, so w stays 0 across the region.
    region_radii = np.where(region_radii < min_radius, 0., region_radii)

    radius_map = jnp.array(region_radii[labeled_mask])

    # Blend weights on GPU
    blend_core_radius = inner_zero_frac * radius_map
    annulus_width = jnp.maximum(radius_map - blend_core_radius, 1e-30)

    w = jnp.where(
        hole_mask_jnp & (dist_jnp > blend_core_radius) & (dist_jnp <= radius_map),
        0.5 * (1 + jnp.cos(jnp.pi * (dist_jnp - blend_core_radius) / annulus_width)),
        0.
    )

    # Apply to all frequencies at once on GPU
    original_jnp = jnp.asarray(original_3d)
    inpainted_jnp = jnp.asarray(inpainted_3d)
    w_3d = w[:,:,None]
    hole_3d = hole_mask_jnp[:,:,None]
    blended = jnp.where(hole_3d, w_3d * original_jnp + (1. - w_3d) * inpainted_jnp, original_jnp)

    return blended


@numba_njit(cache=True, nogil=True)
def _diffusive_inpaint_gs(masked_image, x_0, y_0, n_inpaint):
    """Raster-order Gauss-Seidel sweeps, writing in place. szifi's algorithm, exactly.

    Each masked pixel is set to the mean of its in-bounds 8-neighbours, using values already
    updated earlier in the same sweep. Out-of-bounds neighbours are excluded from both the
    sum and the count. (szifi bounds the second index by nx rather than ny, so it is only
    correct for square tiles; the per-axis bound used here is correct for any shape and
    identical for square ones.)"""

    nx, ny = masked_image.shape

    for _ in range(n_inpaint):

        for i in range(len(x_0)):

            x = x_0[i]
            y = y_0[i]
            c = 0
            value = masked_image.dtype.type(0.)

            for dx in (-1, 0, 1):

                for dy in (-1, 0, 1):

                    if dx == 0 and dy == 0:
                        continue

                    xn = x + dx
                    yn = y + dy

                    if 0 <= xn < nx and 0 <= yn < ny:

                        value += masked_image[xn, yn]
                        c += 1

            masked_image[x, y] = value / c

    return masked_image


def diffusive_inpaint(image, mask, n_inpaint):
    """Diffusive inpainting, bit-compatible with szifi.

    szifi sweeps the masked pixels sequentially in raster order and writes in place, i.e.
    Gauss-Seidel. That is not GPU-parallelisable at this problem size, and it does not need
    to be: it costs ~1 s per tile against a ~300 s cluster-finding run. The GPU Jacobi form
    lives in diffusive_inpaint_jax and is selected with inpaint_type="diffusive_jax".

    The distinction matters. With n_inpaint = 100 neither iteration is converged for the
    larger point-source holes (Planck field 0 has holes up to ~20 px in radius), and Jacobi
    converges about half as fast as Gauss-Seidel, so the two give visibly different maps.
    See PORT_STATUS.md."""

    image_np = np.asarray(image)
    mask_np = np.asarray(mask, dtype=image_np.dtype)

    if np.array_equal(np.ones(mask_np.shape, dtype=mask_np.dtype), mask_np):

        return jnp.asarray(image_np)

    masked_image = np.ascontiguousarray(image_np * mask_np)
    x_0, y_0 = np.where(mask_np == 0.)

    out = _diffusive_inpaint_gs(masked_image, x_0.astype(np.int64), y_0.astype(np.int64),
    int(n_inpaint))

    return jnp.asarray(out)


def _sum_8_neighbours(arr):
    """Sum of the 8 neighbours of every pixel, with out-of-bounds neighbours contributing 0."""
    padded = jnp.pad(arr, 1, mode='constant', constant_values=0.)
    return (padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
            padded[1:-1, :-2] +                    padded[1:-1, 2:] +
            padded[2:, :-2]   + padded[2:, 1:-1]  + padded[2:, 2:])


@jax.jit
def diffusive_inpaint_jax(image, mask, n_inpaint):
    """Diffusive inpainting of the masked pixels, via lax.fori_loop.

    Each masked pixel is replaced by the mean of its in-bounds 8-neighbours, n_inpaint times.
    Out-of-bounds neighbours are excluded from both the sum and the count, so edge and corner
    pixels average over 5 and 3 neighbours respectively — this reproduces szifi's explicit
    bounds checks. (szifi tests the second index against nx rather than ny, so it is only
    correct for square tiles; the per-dimension bound used here is correct for any shape.)

    szifi sweeps the masked pixels sequentially in raster order and writes in place, i.e.
    Gauss-Seidel, which is inherently sequential. This is the Jacobi form of the same
    iteration: both converge to the same discrete harmonic extension, but they are not
    identical at finite n_inpaint. See PORT_STATUS.md for the measured difference."""
    zeros_mask = (mask == 0.)
    masked_image = image * mask

    counts = _sum_8_neighbours(jnp.ones_like(masked_image))

    def body_fn(_, img):
        avg = _sum_8_neighbours(img) / counts
        return jnp.where(zeros_mask, avg, img)

    return jax.lax.fori_loop(0, n_inpaint, body_fn, masked_image)


def smooth_inpainting_border(map_val, mask, n_buffer):
    # Keep as NumPy — uses scipy distance_transform_edt
    map_np = np.asarray(map_val)
    mask_np = np.asarray(mask).astype(int)

    distance, (i_ind, j_ind) = distance_transform_edt(mask_np, return_indices=True)
    blend_region = (mask_np > 0) & (distance <= n_buffer)
    blend_weight = np.clip(1 - distance / n_buffer, 0, 1)
    real_neighbor = map_np[i_ind, j_ind]

    blended = np.copy(map_np)
    blended[blend_region] = (
        blend_weight[blend_region] * real_neighbor[blend_region] +
        (1 - blend_weight[blend_region]) * map_np[blend_region]
    )
    return jnp.array(blended)

def feather_mask_edge(map_val, mask, n_buffer):
    from scipy.ndimage import distance_transform_edt as edt, convolve

    map_np = np.asarray(map_val)
    mask_np = np.asarray(mask).astype(bool)

    distance = edt(mask_np)
    blend_region = (mask_np) & (distance <= n_buffer)
    blend_weight = np.clip(distance / n_buffer, 0, 1)
    feathered = np.copy(map_np)

    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
    edge_val = convolve(np.where(~mask_np, map_np, 0), kernel, mode='constant', cval=0)
    norm = convolve((~mask_np).astype(float), kernel, mode='constant', cval=0)
    edge_pix = ((mask_np) & (distance==1))
    edge_val[edge_pix & (norm>0)] /= norm[edge_pix & (norm>0)]

    feathered[blend_region] = (blend_weight[blend_region] * map_np[blend_region] +
                                (1 - blend_weight[blend_region]) * edge_val[blend_region])

    return jnp.array(feathered)


def get_expanded_map_car(full_map, radec, expansion_deg=1.0):
    # Keep as NumPy — uses healpy
    from pixell import utils as pixell_utils

    [ra_min,ra_max,dec_min,dec_max] = radec

    def move_point_constant_dec(dec,ra,expansion_degree):
        theta = np.radians(90 - dec)
        phi = np.radians(ra)
        vec = hp.ang2vec(theta, phi)
        rot = hp.Rotator(rot=[expansion_degree/np.sin(theta), 0, 0], deg=True)
        vec_rot = rot(vec)
        theta_new, phi_new = hp.vec2ang(vec_rot)
        ra_new = np.degrees(phi_new) % 360
        return ra_new

    def ra_rightmost(ra1, ra2):
        diff = (ra2 - ra1) % 360
        return ra2 if diff < 180 else ra1

    def ra_leftmost(ra1, ra2):
        diff = (ra2 - ra1) % 360
        return ra1 if diff < 180 else ra2

    ra_topleft_expanded = move_point_constant_dec(dec_max,ra_min,expansion_degree=expansion_deg)
    ra_topright_expanded = move_point_constant_dec(dec_max,ra_max,expansion_degree=-expansion_deg)
    ra_bottomleft_expanded = move_point_constant_dec(dec_min,ra_min,expansion_degree=expansion_deg)
    ra_bottomright_expanded = move_point_constant_dec(dec_min,ra_max,expansion_degree=-expansion_deg)

    ra_min_expanded = ra_leftmost(ra_bottomleft_expanded,ra_topleft_expanded)[0]
    ra_max_expanded = ra_rightmost(ra_topright_expanded,ra_bottomright_expanded)[0]

    dec_min_expanded = dec_min - expansion_deg
    dec_max_expanded = dec_max + expansion_deg

    submap_expanded = full_map.submap([[dec_min_expanded*pixell_utils.degree,ra_min_expanded*pixell_utils.degree], [dec_max_expanded*pixell_utils.degree,ra_max_expanded*pixell_utils.degree]])

    dec_map, ra_map = submap_expanded.posmap()
    dec_map = np.degrees(dec_map)
    ra_map = np.degrees(ra_map)

    mask = ((dec_map >= dec_min) & (dec_map <= dec_max) &
            ((ra_map - ra_min) % (360.) <= (ra_max - ra_min) % (360.))).astype(float)

    return submap_expanded, mask

def smooth_mask_transition(map_val, mask, n_buffer):
    from scipy.ndimage import gaussian_filter

    map_np = np.asarray(map_val)
    mask_np = np.asarray(mask)

    sigma = n_buffer / 2.0
    smooth_mask = gaussian_filter(mask_np.astype(float), sigma=sigma)

    masked_values = np.where(mask_np.astype(int), map_np, 0)
    unmasked_values = np.where(~mask_np.astype(int), map_np, 0)

    smoothed_map = (1 - smooth_mask) * unmasked_values + smooth_mask * masked_values

    return jnp.array(smoothed_map)
