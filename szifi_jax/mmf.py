import numpy as np
import jax.numpy as jnp
import jax
from szifi_jax import params, maps, model, cat, spec, utils, sed
import warnings, os
warnings.filterwarnings("ignore")
import gzip
import functools

_TEMPLATE_CACHE = {}

#Core class for cluster finding
class cluster_finder:

    def __init__(self,params_szifi=params.params_szifi_default,
    params_model=params.params_model_default,
    data_file=None,
    rank=0):

        self.params_szifi = params.normalise_params_szifi(params_szifi)
        self.params_model = params_model
        self.data_file = data_file.data #instance of input_data
        self.theta_500_vec = self.params_szifi["theta_500_vec_arcmin"]
        self.rank = rank
        self.exp = self.data_file["experiment"]
        self.cosmology = model.cosmological_model(self.params_szifi).cosmology
        self.indices_filter=None
        self.wcs = None

        if self.params_szifi["detection_method"] == "DBSCAN":

            from sklearn.cluster import DBSCAN

        if self.params_szifi["mmf_type"] == "standard" and self.params_szifi["a_matrix"] is None:

            a_matrix = np.zeros((len(self.exp.tsz_sed),1))
            a_matrix[:,0] = self.exp.tsz_sed
            self.params_szifi["a_matrix"] = a_matrix

        if self.params_szifi["deproject_cib"] is not None:

            get_a_matrix_cib(self.params_szifi,self.params_model,self.data_file)

    def find_clusters(self):

        #Print some information

        if self.params_szifi["mmf_type"] == "standard":

            type_name = "standard"

        elif self.params_szifi["mmf_type"] == "spectrally_constrained":

            n_dep = self.params_szifi["a_matrix"].shape[1]-1
            type_name = "spectrally constrained, " + str(n_dep) + " component(s) deprojected"

        print("SZiFi")
        print("")
        print("")
        print("MMF type:",type_name)
        print("Iterative:",self.params_szifi["iterative"])
        print("Extraction mode:",self.params_szifi["extraction_mode"])
        print("Experiment:",self.exp.experiment_name)
        print("Frequency channels:",self.params_szifi["freqs"])

        #Iterate over fields

        self.results_dict = {}

        field_ids = list(self.data_file["params_data"]["field_ids"])
        gpu_n = int(self.params_szifi.get("gpu_tile_batch") or 1)
        if gpu_n > 1 and self._gpu_tile_batch_ok():
            print("GPU tile batch:", gpu_n)
            for start in range(0, len(field_ids), gpu_n):
                self._find_clusters_gpu_chunk(field_ids[start:start + gpu_n])
            return

        for field_id in field_ids:

            print("")
            print("")
            print("Field",field_id)
            print("")

            #Gather input data

            map_dtype = self.params_szifi["map_dtype"]

            self.t_obs = self.data_file["t_obs"][field_id] #in muK
            self.t_noi = self.data_file["t_noi"][field_id] #in muK

            self.mask_point = utils.extract(self.data_file,"mask_point",field_id,map_dtype)
            self.mask_ps = utils.extract(self.data_file,"mask_ps", field_id,map_dtype)
            self.mask_select = utils.extract(self.data_file,"mask_select",field_id,map_dtype)
            self.mask_select_no_tile = utils.extract(self.data_file,"mask_select_no_tile",field_id,map_dtype)
            self.mask_select_buffer = utils.extract(self.data_file, "mask_select_buffer", field_id, map_dtype)
            self.mask_map = utils.extract(self.data_file,"mask_map",field_id,map_dtype)
            self.mask_peak_finding_no_tile = utils.extract(self.data_file,"mask_peak_finding_no_tile",field_id,map_dtype)
            self.mask_peak_finding = utils.extract(self.data_file,"mask_peak_finding",field_id,map_dtype)

            self.weights = None

            if self.params_szifi["snr_weighting"] == True:

                self.weights = utils.extract(self.data_file,"snr_weights",field_id,map_dtype)

            self.dx = self.data_file["dx_arcmin"][field_id]/60./180.*np.pi
            self.nx = self.data_file["nx"][field_id]
            self.dy = self.dx
            self.ny = self.nx

            if "dy_arcmin" in self.data_file:

                self.dy = self.data_file["dy_arcmin"][field_id]/60./180.*np.pi

            if "ny" in self.data_file:

                self.ny = self.data_file["ny"][field_id]

            self.pix = maps.pixel(self.nx,self.dx,ny=self.ny,dy=self.dy)

            if self.params_szifi["decouple_type"] == "master" and self.params_szifi["compute_coupling_matrix"] == False:

                self.coupling_matrix_name = self.data_file["coupling_matrix_name"][field_id]

            else:

                self.coupling_matrix_name = None

            if self.params_szifi["extraction_mode"] == "fixed":

                self.catalogue_fixed = self.data_file["catalogue_input"][field_id]

            if self.params_szifi["tile_type"] == "car":

                self.wcs = self.t_obs.wcs

            #Select frequency channels to use

            self.t_obs = maps.select_freqs(self.t_obs,self.params_szifi["freqs"])
            self.t_noi = maps.select_freqs(self.t_noi,self.params_szifi["freqs"])

            if self.params_szifi["get_q_true"] == True:

                self.t_true = utils.extract(self.data_file,"t_true",field_id,map_dtype)
                self.t_true = maps.select_freqs(self.t_true,self.params_szifi["freqs"])

            #Inpaint point sources. All-ones masks (e.g. unmasked homog tiles) are a
            #no-op; skip rather than sweeping 6 channels and re-filtering.

            if self.params_szifi["inpaint"] == True and maps.mask_has_holes(self.mask_point):

                self.t_obs = maps.inpaint_freq(self.t_obs,
                                               self.mask_point,
                                               n_inpaint=self.params_szifi["n_inpaint"],
                                               pix=self.pix,
                                               noise=self.exp.noise_levels,
                                               inpaint_type=self.params_szifi["inpaint_type"],
                                               )

            mask_point_0 = self.mask_point

            #Initialise results class

            self.results = cat.results_detection()
            self.results.info = self.params_szifi
            self.results.theta_500_vec = self.theta_500_vec
            self.results.fsky = np.sum(np.asarray(self.mask_select))*self.pix.dx*self.pix.dy/(4.*np.pi)

            #Initialise cluster masking (if iterative noise covariance estimation is required)

            mask_cluster = np.ones((self.pix.nx,self.pix.ny),dtype=np.asarray(self.t_obs).dtype)
            clusters_masked_old = cat.cluster_catalogue()

            #Apodise (except t_noise, which is apodised in the power spectra estimation functions)

            if self.params_szifi["apod_type"] == "new":

                self.t_obs = maps.multiply_t(self.mask_map,self.t_obs)

                if self.params_szifi["get_q_true"] == True:

                    self.t_true = maps.multiply_t(self.mask_map,self.t_true)

            #Filter input maps in harmonic space

            [lmin,lmax] = self.params_szifi["lrange"]

            ell_map = np.asarray(maps.rmap(self.pix).get_ell())
            self.indices_filter = tuple(np.asarray(np.where((ell_map < lmin) | (ell_map > lmax)),dtype=np.int32))
            # Precompute boolean mask for fast filtering (1 where to keep, 0 where to zero)
            self.filter_mask = jnp.array((ell_map >= lmin) & (ell_map <= lmax), dtype=jnp.float64)

            t_noi_original_before_filter = jnp.array(self.t_noi)

            self.t_obs = maps.filter_tmap(self.t_obs,self.pix,self.params_szifi["lrange"],filter_mask=self.filter_mask)
            self.t_noi = maps.filter_tmap(self.t_noi,self.pix,self.params_szifi["lrange"],filter_mask=self.filter_mask)

            t_noi_original = self.t_noi

            if self.params_szifi["get_q_true"] == True:

                self.t_true = maps.filter_tmap(self.t_true,self.pix,self.params_szifi["lrange"],filter_mask=self.filter_mask)

            #Power-spectrum setup. The coupling matrix depends only on mask_ps, the
            #pixelisation and the binning, none of which change with the noise iteration, so
            #build it once per field rather than once per iteration. On Planck field 0 that
            #single NaMaster call is ~139 s, i.e. 97% of the runtime, so recomputing it for
            #every iteration doubled the cost of the whole cluster finder for nothing.

            lmax1d = self.params_szifi["powspec_lmax1d"]
            new_shape = self.params_szifi["powspec_new_shape"]

            if lmax1d is not None:

                if new_shape is not None:

                    raise ValueError("Only one of powspec_lmax1d or powspec_new_shape can be specified")

                new_shape = maps.get_newshape_lmax1d((self.pix.nx,self.pix.ny),lmax1d,self.pix.dx)

            bin_fac = self.params_szifi["powspec_bin_fac"]

            self.ps = spec.power_spectrum(self.pix,
            mask=np.asarray(self.mask_ps),
            cm_compute=self.params_szifi["coupling_matrix_needed"],
            cm_compute_scratch=self.params_szifi["compute_coupling_matrix"],
            cm_save=self.params_szifi["save_coupling_matrix"],
            cm_name=self.coupling_matrix_name,
            bin_fac=bin_fac,
            new_shape=new_shape,
            cm_backend=self.params_szifi["coupling_matrix_backend"])

            #Initiate loop over iterative noise covariance estimation

            i = 0

            while i >= 0:

                if self.rank == 0:

                    print("Noise it",i)

                if not np.any(np.asarray(self.mask_select)):

                    print(f"{self.rank}: mask_select all zeroes for field {field_id}, iteration {i}; breaking")

                    break

                self.mask_point = mask_point_0*mask_cluster

                #Inpaint noise map (as it may change with iteration)

                if self.params_szifi["inpaint"] == True and maps.mask_has_holes(self.mask_point):

                    self.t_noi = maps.inpaint_freq(t_noi_original,
                                                   self.mask_point,
                                                   n_inpaint=self.params_szifi["n_inpaint"],
                                                   pix=self.pix,
                                                   noise=self.exp.noise_levels,
                                                   inpaint_type=self.params_szifi["inpaint_type"])

                    # Vectorized blend: compute blend weights once, apply to all freqs
                    # blend_multi handles jnp/np conversions internally
                    self.t_noi = maps.circular_cosine_blend_adaptive_multi(
                        t_noi_original, self.t_noi, self.mask_point, inner_zero_frac=0.)
                    self.t_noi = maps.filter_tmap(self.t_noi,self.pix,self.params_szifi["lrange"],filter_mask=self.filter_mask)

                #Estimate channel cross-spectra

                cov_type = self.params_szifi["cov_type"]

                self.cspec = spec.cross_spec(np.arange(len(self.params_szifi["freqs"])))

                self.cspec.get_cross_spec(self.pix,
                t_map=self.t_noi,
                ps=self.ps,
                decouple_type=self.params_szifi["decouple_type"],
                inpaint_flag=self.params_szifi["inpaint"],
                mask_point=self.mask_point,
                lsep=self.params_szifi["lsep"],
                bin_fac=bin_fac,
                new_shape=new_shape,
                implementation=self.params_szifi["powspec_implementation"])

                self.inv_cov = self.cspec.get_inv_cov(self.pix,
                                                      t_map=self.t_noi,
                                                      interp_type=self.params_szifi["interp_type"],
                                                      bin_fac=bin_fac,
                                                      new_shape=new_shape,
                                                      mask=np.asarray(self.mask_ps),
                                                      cov_type=cov_type,
                                                      cov_kernel_shape=self.params_szifi["cov_kernel_shape"],
                                                      inv_cov_backend=self.params_szifi["inv_cov_backend"]
                                                      )

                del self.mask_point

                #Compute weights for constrained MMF

                if self.params_szifi["mmf_type"] == "standard":

                    self.params_szifi["cmmf_type"] = "standard_mmf"

                self.cmmf = scmmf_precomputation(pix=self.pix,
                freqs=self.params_szifi["freqs"],
                inv_cov=self.inv_cov,
                lrange=self.params_szifi["lrange"],
                beam_type=self.params_szifi["beam"],
                exp=self.exp,
                cmmf_type=self.params_szifi["cmmf_type"],
                a_matrix=self.params_szifi["a_matrix"],
                comp_to_calculate=self.params_szifi["comp_to_calculate"],
                mmf_type=self.params_szifi["mmf_type"])

                tilemask_mode = self.params_szifi["tilemask_mode"]

                if tilemask_mode == "catalogue":
                    mask_peak_finding_names_list = ["field"]
                    mask_peak_finding_dict = {"field": self.mask_peak_finding_no_tile, "names": mask_peak_finding_names_list}
                    if i == 0:
                        del self.mask_peak_finding
                    mask_select_names_dict = {"field":["field", "tile"]}
                    mask_select_dict = {"field": self.mask_select_no_tile, "tile": self.mask_select,
                                        "names": mask_select_names_dict}
                    catalogue_find_key = ("field", "tile")
                    catalogue_buffer_key = "empty"
                    for_masking_key = ("field", "field")

                elif tilemask_mode == "catalogue_buffer":

                    mask_peak_finding_names_list = ["field"]
                    mask_peak_finding_dict = {"field": self.mask_peak_finding_no_tile, "names": mask_peak_finding_names_list}
                    if i == 0:
                        del self.mask_peak_finding
                    mask_select_names_dict = {"field":["field", "tile", "buffer"]}
                    mask_select_dict = {"field": self.mask_select_no_tile, "tile": self.mask_select,
                                        "buffer": self.mask_select_buffer, "names": mask_select_names_dict}
                    catalogue_find_key = ("field", "tile")
                    catalogue_buffer_key = ("field", "buffer")
                    for_masking_key = ("field", "field")

                elif tilemask_mode == "field":

                    mask_peak_finding_names_list = ["field", "tile"]
                    mask_peak_finding_dict = {"field": self.mask_peak_finding_no_tile, "tile": self.mask_peak_finding,
                                              "names": mask_peak_finding_names_list}
                    mask_select_names_dict= {"field":["field"], "tile":["tile"]}
                    mask_select_dict = {"field": self.mask_select_no_tile, "tile": self.mask_select, "names": mask_select_names_dict}
                    catalogue_find_key = ("tile", "tile")
                    catalogue_buffer_key = "empty"
                    for_masking_key = ("field", "field")

                else:

                    raise ValueError(f"SziFi parameter tilemask_mode=\"{tilemask_mode}\" invalid. Allowed values are 'catalogue' and 'field'.")

                #Matched filter construction

                #Cache templates across noise iterations (they depend on theta_500 and on the
                #tile geometry, but not on the noise). The cache is keyed on the tile geometry
                #and, for CAR, on the tile WCS, because CAR templates are built from the tile's
                #own posmap and therefore differ from tile to tile.
                self._template_cache_key = get_template_cache_key(self.pix,self.wcs,
                self.params_szifi["tile_type"])

                if getattr(self,"_template_cache_key_current",None) != self._template_cache_key:

                    self._template_cache = {}
                    self._template_cache_key_current = self._template_cache_key

                self.filtered_maps = filter_maps(t_obs=self.t_obs,
                inv_cov=self.inv_cov,
                pix=self.pix,
                cosmology=self.cosmology,
                theta_500_vec=self.theta_500_vec,
                field_id=field_id,
                i_it=i,
                params=self.params_szifi,
                params_model=self.params_model,
                mask_map=self.mask_map,
                mask_select_dict=mask_select_dict,
                mask_peak_finding_dict=mask_peak_finding_dict,
                rank=self.rank,
                exp=self.exp,
                cmmf=self.cmmf,
                wcs=self.wcs,
                weights=self.weights,
                filter_mask=self.filter_mask)
                self.filtered_maps._template_cache = self._template_cache

                #SZiFi in cluster finding mode: blind cluster detection

                if self.params_szifi["extraction_mode"] == "find" or (self.params_szifi["extraction_mode"] == "fixed" and self.params_szifi["iterative"] == True):

                    if self.rank == 0:

                        print("Cluster finding")

                    self.filtered_maps.find_clusters()

                    self.results.sigma_vec["find_" + str(i)] = self.filtered_maps.sigma_vec

                    self.results.catalogues["catalogue_find_" + str(i)] = self.filtered_maps.results[catalogue_find_key]
                    self.results.catalogues["catalogue_find_buffer_" + str(i)] = self.filtered_maps.results[catalogue_buffer_key]
                    results_for_masking = self.filtered_maps.results[for_masking_key]

                    if self.rank == 0:

                        print("Detections SNR",np.flip(np.sort(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])))

                #SZiFi in fixed mode: extraction for an input catalogue

                if (self.params_szifi["extraction_mode"] == "fixed"):

                    con1 = (self.params_szifi["iterative"] == True and i > 0)
                    con2 = (self.params_szifi["iterative"] == False)
                    con3 = (self.params_szifi["iterative"] == True and len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.)
                    con4 = (self.params_szifi["iterative"] == True and all(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"] < self.params_szifi["q_th_noise"]))

                    if (con1 or con2 or con3 or con4):

                        if self.rank == 0:

                            print("Extraction for input catalogue")

                        self.filtered_maps.extract_at_true_values(self.t_obs,
                        self.catalogue_fixed)
                        catalogue_obs = self.filtered_maps.catalogue_true_values

                        self.results.catalogues["catalogue_fixed_" + str(i)] = catalogue_obs

                        print("Fixed SNR",np.flip(np.sort(self.results.catalogues["catalogue_fixed_" + str(i)].catalogue["q_opt"])))

                #Extract true SNR

                if self.params_szifi["get_q_true"] == True:

                    con1 = (self.params_szifi["iterative"] == True and i > 0)
                    con2 = (self.params_szifi["iterative"] == False)
                    con3 = (self.params_szifi["iterative"] == True and len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) == 0.)
                    con4 = (self.params_szifi["iterative"] == True and all(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"] < self.params_szifi["q_th_noise"]))

                    if (con1 or con2 or con3 or con4):

                        self.filtered_maps.extract_at_true_values(self.t_true,
                        self.catalogue_fixed,
                        comp_to_calculate=self.params_szifi["comp_to_calculate"],
                        profile_type=self.params_model["profile_type"])
                        catalogue_true_values = self.filtered_maps.catalogue_true_values

                        self.catalogue_true_values = catalogue_true_values
                        self.results.catalogues["catalogue_true_" + str(i)] = catalogue_true_values

                        print("q true",np.flip(np.sort(self.results.catalogues["catalogue_true_" + str(i)].catalogue["q_opt"])))

                #Several breaking conditions

                self.n_clusters = len(self.results.catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]) + \
                                  len(self.results.catalogues["catalogue_find_buffer_" + str(i)].catalogue["q_opt"])

                if self.n_clusters == 0.:

                    break

                if self.params_szifi["iterative"] == False:

                    break

                if i == self.params_szifi["max_it"]:

                    break

                #Cluster masking for iterative covariance estimation

                self.results_for_masking = results_for_masking

                clusters_masked_new = cat.apply_q_cut(self.results_for_masking,self.params_szifi["q_th_noise"])
                clusters_masked_old = cat.apply_q_cut(clusters_masked_old,self.params_szifi["q_th_noise"])

                if len(clusters_masked_new.catalogue["q_opt"]) == 0:

                    break

                if self.params_szifi["tile_type"] == "healpix":

                    lonlat = False

                elif self.params_szifi["tile_type"] == "car":

                    lonlat = "equatorial"

                clusters_masked_old_id,clusters_masked_new_id = cat.identify_clusters(clusters_masked_old,clusters_masked_new,lonlat=lonlat)

                if np.all(clusters_masked_old_id.catalogue["q_opt"]) != -1 and i > 0:

                    break

                mask_cluster = get_cluster_mask(self.pix,
                self.results_for_masking,
                self.params_szifi["q_th_noise"],
                self.params_szifi["mask_radius"],
                tile_type=self.params_szifi["tile_type"],
                wcs=self.wcs,
                max_masking_radius_arcmin=self.params_szifi["max_radius_mask_arcmin"],
                min_masking_radius_arcmin=self.params_szifi["min_radius_mask_arcmin"])

                clusters_masked_old = clusters_masked_new

                i += 1

            # Clean up the saved templates

            if self.params_szifi["save_and_load_template"] and hasattr(self,"filtered_maps"):

                for j_theta in range(len(self.theta_500_vec)):

                    os.remove(self.filtered_maps.template_name % j_theta)
                    os.remove(self.filtered_maps.template_norm_name % j_theta)

            if self.params_szifi["tile_type"] == "healpix" and self.params_szifi["get_lonlat"] == True:

                self.results.get_lonlat(field_id,self.pix,nside=self.data_file["nside_tile"])

            self.results_dict[field_id] = self.results

    def _gpu_tile_batch_ok(self):
        p = self.params_szifi
        mmf_ok = p["mmf_type"] == "standard" or (
            p["mmf_type"] == "spectrally_constrained"
            and p.get("cmmf_type", "one_dep") == "one_dep"
            and p["a_matrix"] is not None
            and np.asarray(p["a_matrix"]).shape[1] >= 2
        )
        return (
            mmf_ok
            and p["decouple_type"] == "fsky"
            and p["cov_type"] == "isotropic"
            and p["extraction_mode"] == "find"
            and p["tile_type"] == "healpix"
            and not p["get_q_true"]
            and not p["snr_weighting"]
            and p.get("powspec_new_shape") is None
            and p.get("powspec_lmax1d") is None
        )

    def _ensure_standard_templates(self, pix, filter_mask):
        n_theta = len(self.theta_500_vec)
        key = (pix.nx, pix.ny, float(pix.dx), float(pix.dy), n_theta,
               self.params_szifi["beam"], self.params_model["los_integral"],
               self.params_szifi["norm_type"], tuple(self.params_szifi["lrange"]),
               tuple(self.theta_500_vec))
        global _TEMPLATE_CACHE
        if key in _TEMPLATE_CACHE:
            self._template_cache = _TEMPLATE_CACHE[key]
            return self._template_cache
        cache = getattr(self, "_template_cache", {})
        if all(j in cache for j in range(n_theta)):
            _TEMPLATE_CACHE[key] = cache
            self._template_cache = cache
            return cache
        for j in range(n_theta):
            if j in cache:
                continue
            if self.rank == 0:
                print("Theta", j, self.theta_500_vec[j])
            z = 0.2
            M_500 = model.get_m_500(self.theta_500_vec[j], z, self.cosmology)
            nfw = model.gnfw(M_500, z, self.cosmology,
                             type=self.params_model["profile_type"],
                             los_integral=self.params_model["los_integral"])
            theta_cart = [(0.5 * pix.nx) * pix.dx, (0.5 * pix.nx) * pix.dx]
            t_tem = nfw.get_t_map_convolved(
                pix, self.exp, beam=self.params_szifi["beam"],
                theta_cart=theta_cart, get_nc=False, sed=False,
                tile_type=self.params_szifi["tile_type"], wcs=None)
            t_tem = t_tem / nfw.get_y_norm(self.params_szifi["norm_type"])
            tem = maps.filter_tmap(t_tem, pix, self.params_szifi["lrange"],
                                   filter_mask=filter_mask)
            cache[j] = (tem, None)
        _TEMPLATE_CACHE[key] = cache
        self._template_cache = cache
        return cache

    def _setup_one_tile(self, field_id):
        map_dtype = self.params_szifi["map_dtype"]
        t_obs = maps.select_freqs(self.data_file["t_obs"][field_id],
                                  self.params_szifi["freqs"])
        t_noi = maps.select_freqs(self.data_file["t_noi"][field_id],
                                  self.params_szifi["freqs"])
        mask_point = utils.extract(self.data_file, "mask_point", field_id, map_dtype)
        mask_select = utils.extract(self.data_file, "mask_select", field_id, map_dtype)
        mask_select_no_tile = utils.extract(self.data_file, "mask_select_no_tile",
                                             field_id, map_dtype)
        mask_select_buffer = utils.extract(self.data_file, "mask_select_buffer",
                                            field_id, map_dtype)
        mask_map = utils.extract(self.data_file, "mask_map", field_id, map_dtype)
        mask_ps = utils.extract(self.data_file, "mask_ps", field_id, map_dtype)
        mask_peak_finding_no_tile = utils.extract(
            self.data_file, "mask_peak_finding_no_tile", field_id, map_dtype)
        mask_peak_finding = utils.extract(
            self.data_file, "mask_peak_finding", field_id, map_dtype)
        dx = self.data_file["dx_arcmin"][field_id] / 60. / 180. * np.pi
        nx = self.data_file["nx"][field_id]
        dy = dx
        ny = nx
        if "dy_arcmin" in self.data_file:
            dy = self.data_file["dy_arcmin"][field_id] / 60. / 180. * np.pi
        if "ny" in self.data_file:
            ny = self.data_file["ny"][field_id]
        pix = maps.pixel(nx, dx, ny=ny, dy=dy)

        if self.params_szifi["inpaint"] and maps.mask_has_holes(mask_point):
            t_obs = maps.inpaint_freq(
                t_obs, mask_point, n_inpaint=self.params_szifi["n_inpaint"],
                pix=pix, noise=self.exp.noise_levels,
                inpaint_type=self.params_szifi["inpaint_type"])

        results = cat.results_detection()
        results.info = self.params_szifi
        results.theta_500_vec = self.theta_500_vec
        results.fsky = np.sum(np.asarray(mask_select)) * pix.dx * pix.dy / (4. * np.pi)

        if self.params_szifi["apod_type"] == "new":
            t_obs = maps.multiply_t(mask_map, t_obs)

        # Harmonic-space cut is applied once on the stacked batch in
        # _find_clusters_gpu_chunk (16 sequential 1024² FFTs was the old setup cost).
        ell_map = np.asarray(maps.rmap(pix).get_ell())
        lmin, lmax = self.params_szifi["lrange"]
        filter_mask = jnp.array((ell_map >= lmin) & (ell_map <= lmax), dtype=jnp.float64)
        return {
            "field_id": field_id, "pix": pix, "t_obs": t_obs, "t_noi": t_noi,
            "t_noi_original": None, "mask_point0": mask_point,
            "mask_cluster": np.ones((pix.nx, pix.ny), dtype=np.asarray(t_obs).dtype),
            "mask_select": mask_select, "mask_select_no_tile": mask_select_no_tile,
            "mask_select_buffer": mask_select_buffer, "mask_map": mask_map,
            "mask_ps": mask_ps, "mask_peak_finding": mask_peak_finding,
            "mask_peak_finding_no_tile": mask_peak_finding_no_tile,
            "filter_mask": filter_mask, "ell_map": ell_map, "results": results,
            "clusters_masked_old": cat.cluster_catalogue(),
        }

    def _find_clusters_gpu_chunk(self, field_ids):
        """fsky iMMF for many tiles: one GPU launch per noise iteration."""
        tiles = [self._setup_one_tile(fid) for fid in field_ids]
        pix = tiles[0]["pix"]
        filter_mask = tiles[0]["filter_mask"]
        t_obs = maps.filter_tmaps(
            jnp.stack([jnp.asarray(t["t_obs"]) for t in tiles]), pix, filter_mask)
        t_noi = maps.filter_tmaps(
            jnp.stack([jnp.asarray(t["t_noi"]) for t in tiles]), pix, filter_mask)
        for k, tile in enumerate(tiles):
            tile["t_obs"] = t_obs[k]
            tile["t_noi"] = t_noi[k]
            tile["t_noi_original"] = t_noi[k]
        cache = self._ensure_standard_templates(pix, filter_mask)
        tem_list = [cache[j][0] for j in range(len(self.theta_500_vec))]
        a_vec = jnp.asarray(self.params_szifi["a_matrix"][:, 0])
        b_vec = None
        sed_a_fft = sed_b_fft = None
        if self.params_szifi["mmf_type"] == "spectrally_constrained":
            b_vec = jnp.asarray(self.params_szifi["a_matrix"][:, 1])
            dummy_inv = jnp.zeros((pix.nx, pix.ny, len(self.params_szifi["freqs"]),
                                   len(self.params_szifi["freqs"])))
            sed_a_fft, sed_b_fft = scmmf_one_dep_sed_ffts(
                pix=pix,
                freqs=self.params_szifi["freqs"],
                inv_cov=dummy_inv,
                lrange=self.params_szifi["lrange"],
                beam_type=self.params_szifi["beam"],
                exp=self.exp,
                a_matrix=self.params_szifi["a_matrix"],
            )
        bin_fac = self.params_szifi["powspec_bin_fac"]
        l0_bins = np.arange(pix.nx / bin_fac) * bin_fac * np.pi / (pix.nx * pix.dx)
        lf_bins = (np.arange(pix.nx / bin_fac) + 1) * bin_fac * np.pi / (pix.nx * pix.dx)
        ell_eff = 0.5 * (l0_bins + lf_bins)
        ell_map = tiles[0]["ell_map"]
        max_it = self.params_szifi["max_it"] if self.params_szifi["iterative"] else 0
        tilemask_mode = self.params_szifi["tilemask_mode"]

        dummy_inv = jnp.zeros((pix.nx, pix.ny, len(self.params_szifi["freqs"]),
                               len(self.params_szifi["freqs"])))
        cmmf = scmmf_precomputation(
            pix=pix, freqs=self.params_szifi["freqs"], inv_cov=dummy_inv,
            lrange=self.params_szifi["lrange"], beam_type=self.params_szifi["beam"],
            exp=self.exp, cmmf_type="standard_mmf",
            a_matrix=self.params_szifi["a_matrix"], mmf_type="standard")

        i = 0
        active = tiles
        while i >= 0:
            if self.rank == 0:
                print("Noise it", i, "tiles", active[0]["field_id"], "..",
                      active[-1]["field_id"], "n=", len(active))
            t_noi = jnp.stack([jnp.asarray(t["t_noi"]) for t in active])
            if self.params_szifi["apod_type"] == "old":
                t_obs = jnp.stack([
                    maps.multiply_t(t["mask_map"], t["t_obs"]) for t in active])
                t_obs = maps.filter_tmaps(t_obs, pix, filter_mask)
            else:
                t_obs = jnp.stack([jnp.asarray(t["t_obs"]) for t in active])
            masks_ps = jnp.stack([jnp.asarray(t["mask_ps"]) for t in active])
            mask_point = jnp.stack([
                jnp.asarray(t["mask_point0"]) * jnp.asarray(t["mask_cluster"])
                for t in active])

            spec_full = spec.coupled_cells_nmt(t_noi, pix, l0_bins, lf_bins,
                                               masks=masks_ps)
            ell_vec, spec_tensor = spec.apply_fsky_inpaint_drop_first(
                spec_full, ell_eff, np.asarray(masks_ps), np.asarray(mask_point),
                "fsky", self.params_szifi["inpaint"], self.params_szifi["lsep"])
            del spec_full
            inv = spec.invert_spec_interp_tiles(
                ell_map, ell_vec, spec_tensor,
                backend=self.params_szifi.get("inv_cov_backend", "numpy"))
            if b_vec is None:
                q_all, y_all, stds_all = get_mmf_q_maps_tiles(
                    t_obs, tem_list, inv, pix, a_vec)
            else:
                a_dot_b, b_dot_b = scmmf_one_dep_dots_tiles(sed_a_fft, sed_b_fft, inv)
                q_all, y_all, stds_all = get_mmf_q_maps_tiles(
                    t_obs, tem_list, inv, pix, a_vec,
                    b_vec=b_vec, a_dot_b=a_dot_b, b_dot_b=b_dot_b,
                )

            next_active = []
            for k, tile in enumerate(active):
                field_id = tile["field_id"]
                if tilemask_mode == "catalogue":
                    mask_peak_finding_dict = {
                        "field": tile["mask_peak_finding_no_tile"],
                        "names": ["field"]}
                    mask_select_dict = {
                        "field": tile["mask_select_no_tile"],
                        "tile": tile["mask_select"],
                        "names": {"field": ["field", "tile"]}}
                    catalogue_find_key = ("field", "tile")
                    catalogue_buffer_key = "empty"
                    for_masking_key = ("field", "field")
                elif tilemask_mode == "catalogue_buffer":
                    mask_peak_finding_dict = {
                        "field": tile["mask_peak_finding_no_tile"],
                        "names": ["field"]}
                    mask_select_dict = {
                        "field": tile["mask_select_no_tile"],
                        "tile": tile["mask_select"],
                        "buffer": tile["mask_select_buffer"],
                        "names": {"field": ["field", "tile", "buffer"]}}
                    catalogue_find_key = ("field", "tile")
                    catalogue_buffer_key = ("field", "buffer")
                    for_masking_key = ("field", "field")
                else:
                    mask_peak_finding_dict = {
                        "field": tile["mask_peak_finding_no_tile"],
                        "tile": tile["mask_peak_finding"],
                        "names": ["field", "tile"]}
                    mask_select_dict = {
                        "field": tile["mask_select_no_tile"],
                        "tile": tile["mask_select"],
                        "names": {"field": ["field"], "tile": ["tile"]}}
                    catalogue_find_key = ("tile", "tile")
                    catalogue_buffer_key = "empty"
                    for_masking_key = ("field", "field")

                fm = filter_maps(
                    t_obs=tile["t_obs"], inv_cov=inv[k], pix=pix,
                    cosmology=self.cosmology, theta_500_vec=self.theta_500_vec,
                    params=self.params_szifi, params_model=self.params_model,
                    field_id=field_id, i_it=i, mask_map=tile["mask_map"],
                    mask_select_dict=mask_select_dict,
                    mask_peak_finding_dict=mask_peak_finding_dict,
                    rank=self.rank, exp=self.exp, cmmf=cmmf,
                    filter_mask=filter_mask)
                fm._template_cache = cache
                fm._precomputed_q = (q_all[k], y_all[k], stds_all[k])
                fm.find_clusters()
                tile["results"].sigma_vec["find_" + str(i)] = fm.sigma_vec
                tile["results"].catalogues["catalogue_find_" + str(i)] = (
                    fm.results[catalogue_find_key])
                tile["results"].catalogues["catalogue_find_buffer_" + str(i)] = (
                    fm.results[catalogue_buffer_key])
                results_for_masking = fm.results[for_masking_key]
                n_clusters = (
                    len(tile["results"].catalogues["catalogue_find_" + str(i)].catalogue["q_opt"])
                    + len(tile["results"].catalogues["catalogue_find_buffer_" + str(i)].catalogue["q_opt"]))
                if self.rank == 0:
                    qdet = tile["results"].catalogues["catalogue_find_" + str(i)].catalogue["q_opt"]
                    n_q = len(qdet)
                    qmax = float(np.max(qdet)) if n_q else 0.0
                    print(f"  field {field_id}  n={n_q} qmax={qmax:.2f}")

                if (not self.params_szifi["iterative"]) or i == max_it or n_clusters == 0:
                    continue

                clusters_masked_new = cat.apply_q_cut(
                    results_for_masking, self.params_szifi["q_th_noise"])
                if len(clusters_masked_new.catalogue["q_opt"]) == 0:
                    continue
                tile["mask_cluster"] = get_cluster_mask(
                    pix, results_for_masking, self.params_szifi["q_th_noise"],
                    self.params_szifi["mask_radius"],
                    tile_type=self.params_szifi["tile_type"],
                    max_masking_radius_arcmin=self.params_szifi["max_radius_mask_arcmin"],
                    min_masking_radius_arcmin=self.params_szifi["min_radius_mask_arcmin"])
                mp = np.asarray(tile["mask_point0"]) * tile["mask_cluster"]
                tile["_mp_it1"] = mp
                next_active.append(tile)

            if next_active and self.params_szifi["inpaint"]:
                # Exact Gauss-Seidel (szifi), not Jacobi. Tiles are independent, so
                # thread the GS sweeps and cosine blends, then one stacked FFT.
                from concurrent.futures import ThreadPoolExecutor
                hole = []
                for tile in next_active:
                    if maps.mask_has_holes(tile["_mp_it1"]):
                        hole.append(tile)
                    else:
                        tile["t_noi"] = tile["t_noi_original"]
                if hole:
                    t_gs = maps.inpaint_freq_tiles_gs(
                        [t["t_noi_original"] for t in hole],
                        [t["_mp_it1"] for t in hole],
                        n_inpaint=self.params_szifi["n_inpaint"])

                    def _blend(k):
                        return np.asarray(maps.circular_cosine_blend_adaptive_multi(
                            hole[k]["t_noi_original"], t_gs[k],
                            hole[k]["_mp_it1"], inner_zero_frac=0.))

                    with ThreadPoolExecutor(max_workers=min(len(hole), 16)) as ex:
                        blended = list(ex.map(_blend, range(len(hole))))
                    t_filt = maps.filter_tmaps(jnp.stack(blended), pix, filter_mask)
                    for k, tile in enumerate(hole):
                        tile["t_noi"] = t_filt[k]
            elif next_active:
                for tile in next_active:
                    tile["t_noi"] = tile["t_noi_original"]

            if (not self.params_szifi["iterative"]) or i == max_it or not next_active:
                break
            active = next_active
            i += 1


        for tile in tiles:
            field_id = tile["field_id"]
            if self.params_szifi["get_lonlat"]:
                tile["results"].get_lonlat(
                    field_id, pix, nside=self.data_file["nside_tile"])
            self.results_dict[field_id] = tile["results"]


#Get mask at the location of detected clusters (for iterative noise covariance estimation)

def get_cluster_mask(pix,catalogue,q_th_noise,mask_radius,max_masking_radius_arcmin=np.inf,tile_type="healpix",wcs=None,min_masking_radius_arcmin=0.):

    q_opt = np.asarray(catalogue.catalogue["q_opt"])
    above_th = q_opt > q_th_noise
    n_above = np.sum(above_th)

    if n_above == 0:
        return np.ones((pix.nx, pix.ny))

    if tile_type == "healpix":
        # Vectorized: compute all cluster positions at once
        theta_x_all = np.asarray(catalogue.catalogue["theta_x"])[above_th]
        theta_y_all = np.asarray(catalogue.catalogue["theta_y"])[above_th]
        theta_500_all = np.asarray(catalogue.catalogue["theta_500"])[above_th]

        # Compute pixel positions for all clusters
        coords = np.array([maps.get_theta_misc([tx, ty], pix) for tx, ty in zip(theta_x_all, theta_y_all)])

        # Compute masking radii in radians
        masking_radii_arcmin = np.clip(theta_500_all * mask_radius, min_masking_radius_arcmin, max_masking_radius_arcmin)
        masking_radii_rad = masking_radii_arcmin / 60. / 180. * np.pi

        # Coordinate grids
        x_coord = (jnp.arange(pix.nx) - pix.nx * 0.5 + 0.5) * pix.dx
        y_coord = (jnp.arange(pix.ny) - pix.ny * 0.5 + 0.5) * pix.dy

        # Vectorized distance computation for all clusters at once
        mask_cluster = _compute_cluster_mask_jax(
            x_coord, y_coord,
            jnp.array(coords[:, 0]), jnp.array(coords[:, 1]),
            jnp.array(masking_radii_rad),
            pix.dx, pix.dy, pix.nx, pix.ny
        )
        return np.asarray(mask_cluster)

    else:
        # Fallback for CAR
        mask_cluster = np.ones((pix.nx,pix.ny))
        for j in range(0,len(catalogue.catalogue["q_opt"])):
            if catalogue.catalogue["q_opt"][j] > q_th_noise:
                source_coords = np.zeros((1,2))
                source_coords[0,0] = catalogue.catalogue["ra"][j]
                source_coords[0,1] = catalogue.catalogue["dec"][j]
                masking_radius_arcmin = max(min_masking_radius_arcmin,min(max_masking_radius_arcmin,catalogue.catalogue["theta_500"][j]*mask_radius))
                mask_cluster *= np.asarray(maps.ps_mask(pix,1,masking_radius_arcmin,tile_type=tile_type,wcs=wcs).get_mask_map(source_coords=source_coords))
        return mask_cluster


@jax.jit
def _compute_cluster_mask_jax(x_coord, y_coord, cx_arr, cy_arr, r_arr, dx, dy, nx, ny):
    """Vectorized cluster mask computation: all clusters at once on GPU.

    x_coord, y_coord: base coordinate grids (i - nx/2 + 0.5) * dx
    cx_arr, cy_arr: theta_misc offsets for each cluster
    r_arr: masking radii in radians
    """
    # 2D coordinate grids
    x_grid = x_coord[:, None]  # (nx, 1) — broadcasts
    y_grid = y_coord[None, :]  # (1, ny)

    # Distance = sqrt((x_base + theta_x)^2 + (y_base + theta_y)^2) for each cluster
    # This matches rmap.get_distance_map_wrt_centre(theta_misc)
    def mask_one_cluster(cx, cy, r):
        dist2 = (x_grid + cx)**2 + (y_grid + cy)**2
        return dist2 > r**2  # True = keep, False = mask

    # vmap over clusters, then reduce with all (logical AND via min)
    all_masks = jax.vmap(mask_one_cluster)(cx_arr, cy_arr, r_arr)  # (n_clusters, nx, ny)
    return jnp.all(all_masks, axis=0).astype(jnp.float64)

def get_template_cache_key(pix,wcs,tile_type):

    """Key identifying the tile geometry a set of MMF templates was built for.

    For HEALPix tiles the template depends only on the pixelisation, which is shared by every
    tile, so templates can be reused across fields. For CAR tiles the template is built from
    the tile's own posmap and so depends on the tile WCS as well; reusing a cached template
    across CAR tiles would silently apply the first tile's template everywhere."""

    key = (tile_type,pix.nx,pix.ny,pix.dx,pix.dy)

    if tile_type == "car" and wcs is not None:

        key = key + (str(wcs.to_header()),)

    return key

#Construct and apply MMF

class filter_maps:

    def __init__(self,t_obs=None,inv_cov=None,pix=None,cosmology=None,theta_500_vec=None,
    params=None,field_id=0,i_it=0,mask_map=None,mask_select_dict=None,
    mask_peak_finding_dict=None,rank=0,exp=None,cmmf=None,params_model=None,indices_filter=None,wcs=None,weights=None,filter_mask=None):

        self.t_obs = t_obs
        self.inv_cov = inv_cov
        self.pix = pix
        self.cosmology = cosmology
        self.theta_500_vec = theta_500_vec
        self.params = params
        self.params_model = params_model
        self.field_id = field_id
        self.i_it = i_it
        self.mask_map = mask_map
        self.mask_select_dict = mask_select_dict
        self.mask_peak_finding_dict = mask_peak_finding_dict
        self.mask_names_finding = mask_peak_finding_dict["names"]
        self.mask_names_select = mask_select_dict["names"]
        self.rank = rank
        self.theta_500_vec = theta_500_vec
        self.exp = exp
        self.cmmf = cmmf
        self.indices_filter=indices_filter
        self.filter_mask = filter_mask
        self.wcs = wcs
        self.weights = weights

        self.theta_range = [0.,pix.nx*pix.dx,0.,pix.ny*pix.dy]

        if self.params["save_and_load_template"]:

            self.template_name = self.params["path_template"] + f"tem_{self.field_id}_%s.npy"
            self.template_norm_name = self.params["path_template"] + f"tem_norm_{self.field_id}_%s.npy"

    #Find detections blindly

    def find_clusters(self):

        t_obs = self.t_obs

        if self.params["apod_type"] == "old":

            t_obs = maps.multiply_t(self.mask_map,t_obs)
            t_obs = maps.filter_tmap(t_obs,self.pix,self.params["lrange"],filter_mask=self.filter_mask)

        n_theta = len(self.theta_500_vec)

        self.sigma_vec = np.zeros(n_theta)
        detect_peaks_maxima_lomem = False  # overridden in sequential fallback

        #Check whether the batched GPU path can be used. It reproduces the "maxima" /
        #"maxima_lomem" peak finders and the standard MMF only, and it contracts the template
        #FFT directly against inv_cov, so it requires inv_cov to be on the full map grid
        #(i.e. no degraded power spectrum via powspec_lmax1d / powspec_new_shape).
        _cache = getattr(self, '_template_cache', {})
        all_cached = all(j in _cache for j in range(n_theta))
        inv_cov_full_shape = tuple(self.inv_cov.shape[:2]) == (self.pix.nx,self.pix.ny)
        can_batch = (self.params.get("mmf_batched",True)
                     and self.params["mmf_type"] == "standard"
                     and self.params["detection_method"] in ("maxima","maxima_lomem")
                     and inv_cov_full_shape
                     and not self.params["snr_weighting"]
                     and not self.params["save_snr_maps"])
        precomputed_q = getattr(self, "_precomputed_q", None)

        # If templates not yet cached but batch is possible, compute all templates first
        if precomputed_q is None and not all_cached and can_batch:
            for j in range(n_theta):
                if j in _cache:
                    continue

                if self.rank == 0:
                    print("Theta",j,self.theta_500_vec[j])

                if self.params["save_and_load_template"] == True:
                    template_name = self.template_name % j
                    template_norm_name = self.template_norm_name % j
                    if self.i_it == 0 and os.path.isfile(template_name):
                        raise RuntimeError(f"template file {template_name} exists already; overwriting not allowed to prevent collisions during parallel runs")

                if self.params["save_and_load_template"] == False or (self.params["save_and_load_template"] == True and self.i_it == 0):

                    if self.params_model["profile_type"] == "point":
                        ps = model.point_source(self.exp,beam_type=self.params["beam"])
                        t_tem = ps.get_t_map_convolved(self.pix,tile_type=self.params["tile_type"],wcs=self.wcs)
                        t_tem_norm = None

                    elif self.params_model["profile_type"] == "arnaud":
                        z = 0.2
                        M_500 = model.get_m_500(self.theta_500_vec[j],z,self.cosmology)
                        nfw = model.gnfw(M_500,z,self.cosmology,
                        type=self.params_model["profile_type"],
                        los_integral=self.params_model["los_integral"])
                        theta_cart = [(0.5*self.pix.nx)*self.pix.dx,
                        (0.5*self.pix.nx)*self.pix.dx]

                        t_tem = nfw.get_t_map_convolved(self.pix,
                        self.exp,
                        beam=self.params["beam"],
                        theta_cart=theta_cart,
                        get_nc=False,
                        sed=False,
                        tile_type=self.params["tile_type"],
                        wcs=self.wcs
                        )
                        t_tem_norm = None
                        t_tem = t_tem/nfw.get_y_norm(self.params["norm_type"])

                    tem = maps.filter_tmap(t_tem,
                    self.pix,
                    self.params["lrange"],
                    filter_mask=self.filter_mask)
                    _cache[j] = (tem, t_tem_norm)

                    if self.params["save_and_load_template"] == True:
                        np.save(template_name,np.asarray(tem))
                        np.save(template_norm_name,np.asarray(t_tem_norm) if t_tem_norm is not None else t_tem_norm)

                elif self.params["save_and_load_template"] == True and self.i_it > 0:
                    tem = jnp.array(np.load(template_name,allow_pickle=True)[()])
                    t_tem_norm_loaded = np.load(template_norm_name,allow_pickle=True)[()]
                    t_tem_norm = jnp.array(t_tem_norm_loaded) if t_tem_norm_loaded is not None else None
                    _cache[j] = (tem, t_tem_norm)

            all_cached = True

        use_batch = (precomputed_q is not None) or (
            all_cached and can_batch
            and all(_cache[j][1] is None for j in range(n_theta)))

        if use_batch:
            if precomputed_q is not None:
                q_maps_all, y_maps_all, stds_all = precomputed_q
            else:
                # BATCHED PATH: compute all q_maps in a single GPU call
                tem_list = [_cache[j][0] for j in range(n_theta)]
                a_vec = jnp.asarray(self.cmmf.a_matrix[:, 0])

                q_maps_all, y_maps_all, stds_all = get_mmf_q_maps_batch(
                    t_obs, tem_list, self.inv_cov, self.pix, a_vec)

            self.sigma_vec = np.asarray(stds_all)

            # Keep on GPU: (n_theta, nx, ny) -> (nx, ny, n_theta)
            q_tensor_jnp = q_maps_all.transpose(1, 2, 0)
            y_tensor_jnp = y_maps_all.transpose(1, 2, 0)

            # Do masked 3D peak detection on GPU, one call per mask
            q_th = self.params["q_th"]
            self._batch_detections = {}  # store per-mask detection results
            for mask_name in self.mask_names_finding:
                mask_jnp = jnp.asarray(self.mask_peak_finding_dict[mask_name]).astype(bool)
                peak_mask = _get_maxima_mask_3d_masked_jit(q_tensor_jnp, mask_jnp, q_th)
                # Transfer only the small boolean peak mask
                maxs_idx = np.argwhere(np.asarray(peak_mask))
                if maxs_idx.shape[0] == 0:
                    self._batch_detections[mask_name] = (np.array([]), np.array([]), np.array([]),
                                                         np.array([]), np.array([]))
                else:
                    ii = maxs_idx[:,0].astype(int)
                    jj = maxs_idx[:,1].astype(int)
                    tt = maxs_idx[:,2].astype(int)
                    # Extract values at peak locations on GPU, then transfer small arrays
                    q_opt = np.asarray(q_tensor_jnp[ii, jj, tt])
                    y0_est = np.asarray(y_tensor_jnp[ii, jj, tt])
                    theta_est = self.theta_500_vec[tt]
                    self._batch_detections[mask_name] = (q_opt, y0_est, theta_est, ii, jj)

            # Keep JAX tensors for potential downstream use (lazy transfer)
            self.q_tensor_jnp = q_tensor_jnp
            self.y_tensor_jnp = y_tensor_jnp

        else:
            # SEQUENTIAL FALLBACK: non-standard MMF types, snr_weighting, etc.
            detect_peaks_maxima_lomem = (self.params["detection_method"] == "maxima_lomem") and (n_theta > 3)

            if detect_peaks_maxima_lomem:
                self.peak_info = {}
                for mask_name in self.mask_names_finding:
                    self.peak_info[mask_name] = np.empty((5,0),dtype=np.float32)
                self.q_tensor = np.zeros((self.pix.nx,self.pix.ny,3),dtype=self.params["map_dtype"])
                self.y_map_last = None
            else:
                self.q_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta),dtype=self.params["map_dtype"])
                self.y_tensor = np.zeros((self.pix.nx,self.pix.ny,n_theta),dtype=self.params["map_dtype"])

            for j in range(0,n_theta):

                if self.rank == 0:

                    print("Theta",j,self.theta_500_vec[j])

                if self.params["save_and_load_template"] == True:

                    template_name = self.template_name % j
                    template_norm_name = self.template_norm_name % j

                    if self.i_it == 0 and os.path.isfile(template_name):

                        raise RuntimeError(f"template file {template_name} exists already; overwriting not allowed to prevent collisions during parallel runs")

                if j in _cache:
                    # Reuse cached template from previous noise iteration
                    tem, t_tem_norm = _cache[j]

                elif self.params["save_and_load_template"] == False or (self.params["save_and_load_template"] == True and self.i_it == 0):

                    if self.params_model["profile_type"] == "point":

                        ps = model.point_source(self.exp,beam_type=self.params["beam"])
                        t_tem = ps.get_t_map_convolved(self.pix,tile_type=self.params["tile_type"],wcs=self.wcs)
                        t_tem_norm = None

                    elif self.params_model["profile_type"] == "arnaud":

                        z = 0.2
                        M_500 = model.get_m_500(self.theta_500_vec[j],z,self.cosmology)
                        nfw = model.gnfw(M_500,z,self.cosmology,
                        type=self.params_model["profile_type"],
                        los_integral=self.params_model["los_integral"])

                        theta_cart = [(0.5*self.pix.nx)*self.pix.dx,
                        (0.5*self.pix.nx)*self.pix.dx]

                        if self.params["mmf_type"] == "standard" or (self.params["mmf_type"] == "spectrally_constrained" and self.params["cmmf_type"] == "one_dep"):

                            t_tem = nfw.get_t_map_convolved(self.pix,
                            self.exp,
                            beam=self.params["beam"],
                            theta_cart=theta_cart,
                            get_nc=False,
                            sed=False,
                            tile_type=self.params["tile_type"],
                            wcs=self.wcs
                            )
                            t_tem_norm = None

                        elif self.params["mmf_type"] == "spectrally_constrained" and self.params["cmmf_type"] == "general":

                            t_tem_norm,t_tem = nfw.get_t_map_convolved(self.pix,
                            self.exp,
                            beam=self.params["beam"],
                            theta_cart=theta_cart,
                            get_nc=True,
                            sed=False,
                            tile_type=self.params["tile_type"],
                            wcs=self.wcs
                            )

                            t_tem_norm = t_tem_norm/nfw.get_y_norm(self.params["norm_type"])
                            t_tem_norm = maps.filter_tmap(t_tem_norm,
                            self.pix,
                            self.params["lrange"],
                            filter_mask=self.filter_mask)

                        t_tem = t_tem/nfw.get_y_norm(self.params["norm_type"])

                    tem = maps.filter_tmap(t_tem,
                    self.pix,
                    self.params["lrange"],
                    filter_mask=self.filter_mask)

                    # Cache template in memory for reuse across noise iterations
                    _cache[j] = (tem, t_tem_norm)

                    if self.params["save_and_load_template"] == True:

                        np.save(template_name,np.asarray(tem))
                        np.save(template_norm_name,np.asarray(t_tem_norm) if t_tem_norm is not None else t_tem_norm)

                elif self.params["save_and_load_template"] == True and self.i_it > 0:

                    tem = jnp.array(np.load(template_name,allow_pickle=True)[()])
                    t_tem_norm_loaded = np.load(template_norm_name,allow_pickle=True)[()]
                    t_tem_norm = jnp.array(t_tem_norm_loaded) if t_tem_norm_loaded is not None else None
                    _cache[j] = (tem, t_tem_norm)

                q_map,y_map,std = get_mmf_q_map(t_obs,
                tem,
                self.inv_cov,
                self.pix,
                mmf_type=self.params["mmf_type"],
                cmmf_prec=self.cmmf,
                tem_norm=t_tem_norm)

                if self.params["snr_weighting"] == True:

                    q_map = q_map*self.weights

                del tem, t_tem_norm

                if self.params["save_snr_maps"] == True:

                    q_map_np = np.asarray(q_map)
                    fil = gzip.GzipFile(self.params["snr_maps_path"] + "/" + self.params["snr_maps_name"] + "_q_" + str(self.i_it) + "_" + str(j) + ".npy.gz", "w")
                    np.save(file=fil,arr=q_map_np*np.asarray(self.mask_select_dict["tile"]))
                    fil.close()

                if detect_peaks_maxima_lomem:

                    q_map_np = np.asarray(q_map)
                    y_map_np = np.asarray(y_map)

                    q_th = self.params["q_th"]

                    if j == 0:

                        self.q_tensor[:,:,1] = q_map_np
                        self.y_map_last = y_map_np

                        #szifi `continue`s here before recording sigma_vec[0], so its
                        #maxima_lomem path always reports sigma_vec[0] == 0. Record it, so
                        #that this path agrees with the batched one.
                        self.sigma_vec[j] = float(std)

                        continue

                    else:

                        self.q_tensor[:,:,2] = q_map_np

                        qm = self.q_tensor[:,:,1]
                        zmax = np.argmax(self.q_tensor, axis=2) == 1

                        for mask_name in self.mask_names_finding:

                            q_opt,y0_est,maxs_idx = make_detections2(qm,self.y_map_last,np.asarray(self.mask_peak_finding_dict[mask_name]),q_th,zmax)

                            self.save_detections(q_opt,y0_est,maxs_idx,mask_name,j-1)

                        if j == n_theta-1:

                            qm = self.q_tensor[:,:,2]
                            zmax = np.argmax(self.q_tensor[:,:,1:], axis=2) == 1

                            for mask_name in self.mask_names_finding:

                                q_opt,y0_est,maxs_idx = make_detections2(qm,y_map_np,np.asarray(self.mask_peak_finding_dict[mask_name]),q_th,zmax)
                                self.save_detections(q_opt,y0_est,maxs_idx,mask_name,j)

                        self.q_tensor = np.roll(self.q_tensor,-1,axis=2)
                        self.y_map_last = y_map_np

                        del qm, zmax, q_map, y_map

                else:

                    self.q_tensor[:,:,j] = np.asarray(q_map)
                    self.y_tensor[:,:,j] = np.asarray(y_map)

                    del q_map, y_map

                self.sigma_vec[j] = float(std)

        self.results = {}

        if self.params["tile_type"] == "healpix":

            x_coord = np.asarray(maps.rmap(self.pix).get_x_coord_map_wrt_origin())
            y_coord = np.asarray(maps.rmap(self.pix).get_y_coord_map_wrt_origin())

        elif self.params["tile_type"] == "car":

            from pixell import enmap
            position_map = enmap.empty(np.asarray(t_obs[:,:,0]).shape,self.wcs)
            dec_map, ra_map = position_map.posmap()
            x_coord = ra_map
            y_coord = dec_map

        for mask_name in self.mask_names_finding:

            if use_batch:
                # Results already computed on GPU during batch detection
                q_opt, y0_est, theta_est, inds0, inds1 = self._batch_detections[mask_name]
                x_est = x_coord[(inds0, inds1)] if len(inds0) > 0 else np.array([])
                y_est = y_coord[(inds0, inds1)] if len(inds0) > 0 else np.array([])

            elif not detect_peaks_maxima_lomem:

                q_tensor = apply_mask_peak_finding(self.q_tensor,np.asarray(self.mask_peak_finding_dict[mask_name]))
                indices = make_detections(q_tensor,self.params["q_th"],self.pix,detection_method=self.params["detection_method"])
                q_opt = q_tensor[indices]
                y_tensor = apply_mask_peak_finding(self.y_tensor,np.asarray(self.mask_peak_finding_dict[mask_name]))
                y0_est = y_tensor[indices]
                x_est = x_coord[(indices[0],indices[1])]
                y_est = y_coord[(indices[0],indices[1])]
                theta_est = self.theta_500_vec[indices[2]]

            else:
                q_opt, y0_est, theta_est, inds0, inds1 = self.peak_info[mask_name]
                inds0, inds1 = inds0.astype(np.int64), inds1.astype(np.int64)
                x_est = x_coord[(inds0, inds1)]
                y_est = y_coord[(inds0, inds1)]

            cat_new = cat.cluster_catalogue()

            cat_new.catalogue["q_opt"] = q_opt
            cat_new.catalogue["y0"] = y0_est
            cat_new.catalogue["theta_500"] = theta_est
            cat_new.catalogue["pixel_ids"] = [self.field_id]*len(q_opt)

            if self.params["tile_type"] == "healpix":

                theta_x = y_est + self.theta_range[0]
                theta_y = self.pix.nx*self.pix.dx-x_est + self.theta_range[2]
                cat_new.catalogue["theta_x"] = theta_x
                cat_new.catalogue["theta_y"] = theta_y

            elif self.params["tile_type"] == "car":

                cat_new.catalogue["ra"] = x_est
                cat_new.catalogue["dec"] = y_est

            for mask_name_select in self.mask_names_select[mask_name]:
                cat_new = cat.apply_mask_select(cat_new,self.mask_select_dict[mask_name_select],self.pix,tile_type=self.params["tile_type"],wcs=self.wcs)
                self.results[(mask_name, mask_name_select)] = cat_new

        self.results["empty"] = cat.cluster_catalogue()

        return 0

    #Apply MMF for input catalogue

    def extract_at_true_values(self,t_true,true_catalogue):

        if self.params["apod_type"] == "old":

            t_true = maps.multiply_t(self.mask_map,t_true)
            t_true = maps.filter_tmap(t_true,self.pix,self.params["lrange"],filter_mask=self.filter_mask)

        if self.params["tile_type"] == "healpix":

            n_clus = len(true_catalogue.catalogue["theta_x"])
            theta_0 = true_catalogue.catalogue["theta_x"]
            theta_1 = true_catalogue.catalogue["theta_y"]

        elif self.params["tile_type"] == "car":

            n_clus = len(true_catalogue.catalogue["ra"])
            theta_0 = true_catalogue.catalogue["ra"]
            theta_1 = true_catalogue.catalogue["dec"]

        q_opt = np.zeros((n_clus,len(self.params["comp_to_calculate"])))
        y0_est = np.zeros((n_clus,len(self.params["comp_to_calculate"])))

        if self.rank == 0:

            print("Extracting at true parameters")

        for i in range(0,n_clus):

            z = 0.2
            M_500 = model.get_m_500(true_catalogue.catalogue["theta_500"][i],z,self.cosmology)

            if self.params["rSZ"] == True:

                tsz_sed_model = sed.tsz_model(T_e=true_catalogue.catalogue["T"][i])

                if self.params["integrate_bandpass"] == True:

                    tsz_sed = tsz_sed_model.get_sed_exp_bandpass(self.exp)

                elif self.params["integrate_bandpass"] == False:

                    tsz_sed = tsz_sed_model.get_sed(self.exp.nu_eff)

                tsz_sed = tsz_sed[self.params["freqs"]]

                a_matrix = self.params["a_matrix"][self.params["freqs"],:]
                a_matrix[:,0] = tsz_sed

            q_extracted,y0_extracted,q_map = extract_at_input_value(t_true,
            self.inv_cov,
            self.pix,
            self.params["beam"],
            M_500,
            z,
            self.cosmology,
            self.params["norm_type"],
            theta_0[i],
            theta_1[i],
            self.params["lrange"],
            apod_type=self.params["apod_type"],
            mmf_type=self.params["mmf_type"],
            cmmf_type=self.params["cmmf_type"],
            cmmf_prec=self.cmmf,
            freqs=self.params["freqs"],
            exp=self.exp,
            comp_to_calculate=self.params["comp_to_calculate"],
            profile_type=self.params_model["profile_type"],
            los_integral=self.params_model["los_integral"],
            indices_filter=self.indices_filter,
            tile_type=self.params["tile_type"],
            wcs=self.wcs,
            weights=self.weights,
            filter_mask=self.filter_mask
            )

            q_opt[i,:] = np.asarray(q_extracted)
            y0_est[i,:] = np.asarray(y0_extracted)

            if self.rank == 0:

                print("theta_500",true_catalogue.catalogue["theta_500"][i],"SNR",q_extracted)

        catalogue_at_true_values = cat.cluster_catalogue()

        catalogue_at_true_values.catalogue["q_opt"] = q_opt[:,0]
        catalogue_at_true_values.catalogue["y0"] = y0_est[:,0]
        catalogue_at_true_values.catalogue["theta_500"] = true_catalogue.catalogue["theta_500"]
        catalogue_at_true_values.catalogue["pixel_ids"] =  [self.field_id]*n_clus
        catalogue_at_true_values.catalogue["m_500"] = true_catalogue.catalogue["m_500"]
        catalogue_at_true_values.catalogue["z"] = true_catalogue.catalogue["z"]

        if self.params["tile_type"] == "healpix":

            catalogue_at_true_values.catalogue["theta_x"] = true_catalogue.catalogue["theta_x"]
            catalogue_at_true_values.catalogue["theta_y"] = true_catalogue.catalogue["theta_y"]

        elif self.params["tile_type"] == "car":

            catalogue_at_true_values.catalogue["dec"] = true_catalogue.catalogue["dec"]
            catalogue_at_true_values.catalogue["ra"] = true_catalogue.catalogue["ra"]

        #Extraction of other components

        for i in range(1,len(self.params["comp_to_calculate"])):

            catalogue_at_true_values.catalogue["q_c" + str(i)] = q_opt[:,i]
            catalogue_at_true_values.catalogue["c" + str(i)] = y0_est[:,i]

        self.catalogue_true_values = catalogue_at_true_values

    def save_detections(self,q_opt,y0_est,maxs_idx,mask_name,j_theta):

        thetas = np.ones_like(q_opt)*self.theta_500_vec[j_theta]
        ans = np.vstack([q_opt,y0_est,thetas,maxs_idx.T])
        self.peak_info[mask_name] = np.hstack([self.peak_info[mask_name], ans])

#Apply MMF for input catalogue

def extract_at_input_value(t_true,inv_cov,pix,beam,M_500,z,cosmology,norm_type,
theta_0,theta_1,lrange,apod_type=None,mmf_type=None,cmmf_prec=None,cmmf_type=None,
freqs=None,exp=None,comp_to_calculate=None,profile_type=None,indices_filter=None,
tile_type="healpix",wcs=None,weights=None,filter_mask=None,los_integral="gauss_legendre"):

    nfw = model.gnfw(M_500,z,cosmology,type=profile_type,los_integral=los_integral)

    q_opt = np.zeros(len(comp_to_calculate))
    y0_est = np.zeros(len(comp_to_calculate))

    if mmf_type == "standard" or (mmf_type == "spectrally_constrained" and cmmf_type == "one_dep"):

        tem = nfw.get_t_map_convolved(pix,
        exp,
        beam=beam,
        get_nc=False,
        sed=False,
        tile_type=tile_type,
        wcs=wcs
        )
        t_tem_norm = None

    elif mmf_type == "spectrally_constrained" and cmmf_type == "general":

        t_tem_norm,tem = nfw.get_t_map_convolved(pix,
        exp,
        beam=beam,
        get_nc=True,
        sed=False,
        tile_type=tile_type,
        wcs=wcs
        )

        t_tem_norm = t_tem_norm/nfw.get_y_norm(norm_type)
        t_tem_norm = maps.filter_tmap(t_tem_norm,pix,lrange)

    tem = tem/nfw.get_y_norm(norm_type)
    tem = maps.select_freqs(tem,freqs)
    tem = maps.filter_tmap(tem,pix,lrange,filter_mask=filter_mask)

    for k in range(0,len(comp_to_calculate)):

        q_map_tem,y_map,std = get_mmf_q_map(t_true,tem,inv_cov,pix,mmf_type=mmf_type,
        cmmf_prec=cmmf_prec,tem_norm=t_tem_norm,comp=comp_to_calculate[k])

        if weights is not None:

            q_map_tem = q_map_tem*weights

        if k == 0:

            q_map = q_map_tem

        if tile_type == "healpix":

            j_clus = int(np.floor(theta_0/pix.dx))
            i_clus = int(pix.ny-np.ceil(theta_1/pix.dy))

        elif tile_type == "car":

            ra = theta_0
            dec = theta_1

            from pixell import enmap
            pix_coords = enmap.sky2pix((pix.nx,pix.ny),wcs,(dec,ra))
            i_clus,j_clus = np.round(pix_coords).astype(int)

        q_opt[k] = float(q_map_tem[i_clus,j_clus])
        y0_est[k] = float(y_map[i_clus,j_clus])

    return q_opt,y0_est,q_map

def apply_mask_peak_finding(tensor,mask):

    return tensor * mask[:,:,None]

#Engine for cluster finding

def make_detections(q_tensor,q_th,pix,detection_method="maxima"):

    if detection_method == "DBSCAN":

        indices = np.where(q_tensor > q_th)
        coords = np.zeros((len(indices[0]),2),dtype=int)
        coords[:,0] = indices[0]
        coords[:,1] = indices[1]
        theta_coord = np.array(indices[2])

        if coords.shape[0] == 0:
            return ([],[],[])

        from sklearn.cluster import DBSCAN

        clust = DBSCAN(eps=2)
        clust.fit(coords)
        labels = clust.labels_
        n_clusters = np.max(labels)+1

        i_opt_vec = np.zeros(n_clusters,dtype=int)
        j_opt_vec = np.zeros(n_clusters,dtype=int)
        theta_opt_vec = np.zeros(n_clusters,dtype=int)

        for i in range(0,n_clusters):

            indices = np.where(labels==i)
            q_tensor_select = q_tensor[coords[:,0][indices],coords[:,1][indices],theta_coord[indices]]
            opt_index = np.argmax(q_tensor_select)
            i_opt_vec[i]  = coords[:,0][indices][opt_index]
            j_opt_vec[i] = coords[:,1][indices][opt_index]
            theta_opt_vec[i] = theta_coord[indices][opt_index]

        return (i_opt_vec,j_opt_vec,theta_opt_vec)

    elif detection_method in ["maxima", "maxima_lomem"]:

        # Use JIT'd 3D maxima detection
        q_jnp = jnp.asarray(q_tensor)
        peak_mask = _get_maxima_mask_3d_jit(q_jnp, q_th)
        maxs_idx = np.argwhere(np.asarray(peak_mask))

        if maxs_idx.shape[0] == 0:
            return ([],[],[])

        i = maxs_idx[:,0].astype(int)
        j = maxs_idx[:,1].astype(int)
        theta = maxs_idx[:,2].astype(int)

        return (i,j,theta)

    raise ValueError(f"detection_method \"{detection_method}\" not supported. "
    "Allowed values are 'maxima', 'maxima_lomem' and 'DBSCAN'.")

def make_detections2(qmap,ymap,mask_peak_finding,q_th,zmax):

    # Build the thresholded + masked q_map for peak finding
    qmap_jnp = jnp.asarray(qmap)
    mask_jnp = jnp.asarray(mask_peak_finding).astype(bool)
    zmax_jnp = jnp.asarray(zmax)

    # Use JIT'd peak finding
    peak_mask = _get_maxima_mask_2d_jit(qmap_jnp, mask_jnp, q_th, zmax_jnp)
    maxs_idx = np.argwhere(np.asarray(peak_mask))

    indices = tuple(maxs_idx.astype(int).T)
    q_opt = qmap[indices]
    y0_est = ymap[indices]

    return q_opt,y0_est,maxs_idx

#Mask for peak finding

def get_maxima_mask(r_input):

    r = np.ones((r_input.shape[0]+2,r_input.shape[1]+2,r_input.shape[2]+2)) * -np.inf
    r[1:-1,1:-1,1:-1] = r_input

    xmax=np.argmax([r[2:,1:-1,1:-1],r[:-2,1:-1,1:-1],r[1:-1,1:-1,1:-1]],axis=0) == 2
    ymax=np.argmax([r[1:-1,2:,1:-1], r[1:-1,:-2,1:-1],r[1:-1,1:-1,1:-1]], axis=0) == 2
    zmax=np.argmax([r[1:-1,1:-1, 2:], r[1:-1,1:-1, :-2], r[1:-1,1:-1, 1:-1]], axis=0) == 2

    return xmax&ymax&zmax

@jax.jit
def _get_maxima_mask_3d_jit(q_tensor, q_th):
    """JIT-compiled 3D local maxima detection with threshold."""
    r = jnp.where(q_tensor > q_th, q_tensor, 0.)
    r_padded = jnp.pad(r, 1, mode='constant', constant_values=-jnp.inf)
    center = r_padded[1:-1, 1:-1, 1:-1]
    xmax = (center > r_padded[2:, 1:-1, 1:-1]) & (center > r_padded[:-2, 1:-1, 1:-1])
    ymax = (center > r_padded[1:-1, 2:, 1:-1]) & (center > r_padded[1:-1, :-2, 1:-1])
    zmax = (center > r_padded[1:-1, 1:-1, 2:]) & (center > r_padded[1:-1, 1:-1, :-2])
    return xmax & ymax & zmax & (center > 0)

@jax.jit
def _get_maxima_mask_3d_masked_jit(q_tensor, mask, q_th):
    """3D maxima detection with 2D mask applied, all on GPU."""
    # Apply mask: zero out pixels outside mask
    r = jnp.where((q_tensor > q_th) & mask[:,:,None], q_tensor, 0.)
    r_padded = jnp.pad(r, 1, mode='constant', constant_values=-jnp.inf)
    center = r_padded[1:-1, 1:-1, 1:-1]
    xmax = (center > r_padded[2:, 1:-1, 1:-1]) & (center > r_padded[:-2, 1:-1, 1:-1])
    ymax = (center > r_padded[1:-1, 2:, 1:-1]) & (center > r_padded[1:-1, :-2, 1:-1])
    zmax = (center > r_padded[1:-1, 1:-1, 2:]) & (center > r_padded[1:-1, 1:-1, :-2])
    return xmax & ymax & zmax & (center > 0)

@jax.jit
def _get_maxima_mask_2d_jit(qmap, mask_peak_finding, q_th, zmax):
    """JIT-compiled 2D local maxima detection with threshold and mask.

    Returns boolean mask of peaks that are:
    - Above q_th threshold
    - Within mask_peak_finding
    - Local maxima in both x and y
    - Also maxima along the theta axis (zmax)
    """
    # Build masked qmap: zero where not (mask & above threshold)
    valid = mask_peak_finding & (qmap > q_th)
    r = jnp.where(valid, qmap, 0.)

    # Pad with -inf for boundary handling
    r_padded = jnp.pad(r, 1, mode='constant', constant_values=-jnp.inf)

    # Local maxima: center > all neighbors
    center = r_padded[1:-1, 1:-1]
    xmax = (center > r_padded[2:, 1:-1]) & (center > r_padded[:-2, 1:-1])
    ymax = (center > r_padded[1:-1, 2:]) & (center > r_padded[1:-1, :-2])

    # Also require center > 0 (i.e., it was a valid pixel)
    return xmax & ymax & zmax & (center > 0)

def get_maxima_mask_2d(r_input, coords=None):

    r = np.ones((r_input.shape[0]+2, r_input.shape[1]+2)) * -np.inf
    if coords is None:
        r[1:-1, 1:-1] = r_input
    else:
        r[1:-1, 1:-1] = 0
        r[1:-1, 1:-1][coords] = r_input[coords]

    xmax = np.argmax([r[2:, 1:-1], r[:-2, 1:-1], r[1:-1, 1:-1]], axis=0) == 2
    ymax = np.argmax([r[1:-1, 2:], r[1:-1, :-2], r[1:-1, 1:-1]], axis=0) == 2

    return xmax&ymax

#Peak finding

def get_maxima(max_mask):

    maxima = np.argwhere(max_mask == True)

    return maxima

#Get SNR map — uses JAX for FFT convolution

def get_mmf_q_map(tmap,tem,inv_cov,pix,
mmf_type="standard",cmmf_prec=None,comp=0,tem_norm=None):

    n_freqs = tmap.shape[2]
    y_map = jnp.zeros((pix.nx,pix.ny), dtype=tmap.dtype)

    tem_fft = maps.get_fft_f(tem,pix)
    filter_fft = get_tem_conv_fft(pix,tem_fft,inv_cov,mmf_type=mmf_type,
    cmmf_prec=cmmf_prec,comp=comp)

    del tem_fft

    filter_val = jnp.asarray(maps.get_ifft_f(filter_fft,pix).real, dtype=tmap.dtype)

    del filter_fft

    tem = maps.get_tmap_times_fvec(tem,cmmf_prec.a_matrix[:,comp]).real

    if tem_norm is None:

        tem_norm = tem

    else:

        tem_norm = maps.get_tmap_times_fvec(tem_norm,cmmf_prec.a_matrix[:,comp]).real

        del tem

    y_map = utils._fftconvolve_same_sum(tmap, filter_val) * pix.dx * pix.dy
    y_map = jnp.asarray(y_map, dtype=tmap.dtype)

    norm = jnp.sum(tem_norm*filter_val)*pix.dx*pix.dy

    del tem_norm, filter_val

    y_map = y_map/norm

    std = 1./jnp.sqrt(norm)
    q_map = y_map/std

    return q_map,y_map,std


@functools.partial(jax.jit, static_argnums=(6, 7))
def _get_mmf_q_maps_batch_jit(tmap, tem_batch, inv_cov, a_vec, dx, dy, pnx, pny):
    """Batched matched filter: compute q_map for all templates at once.

    tmap: (nx, ny, n_freq) — observed map (float64)
    tem_batch: (n_theta, nx, ny, n_freq) — all templates stacked
    inv_cov: (nx, ny, n_freq, n_freq) — inverse covariance
    a_vec: (n_freq,) — SED vector
    dx, dy: pixel sizes
    pnx, pny: padded FFT sizes (static)
    Returns: q_maps (n_theta, nx, ny), y_maps (n_theta, nx, ny), stds (n_theta,)
    """
    nx, ny_dim, n_freq = tmap.shape
    fft_norm = jnp.sqrt((dx * dy) / (nx * ny_dim))
    ifft_norm = jnp.sqrt((nx * ny_dim) / (dx * dy))
    dxdy = dx * dy

    # Pre-compute tmap FFTs for the convolution (shared across all templates). szifi runs this
    # convolution at the map dtype (params["map_dtype"]), so follow tmap's dtype rather than
    # forcing float32 — otherwise map_dtype=np.float64 would be silently downgraded.
    conv_dtype = tmap.dtype
    n_theta = tem_batch.shape[0]
    n_pix = nx * ny_dim

    # FFTs batch over a leading axis. Do not vmap a 5D (n_theta,nx,ny,n_freq) x
    # (nx,ny,n_freq,n_freq) einsum: on Blackwell (sm_120, JAX 0.10) that raises
    # JaxRuntimeError INTERNAL (not OOM — n_theta=1 works, n_theta=2 fails from
    # nx>=256 or n_freq>=2). Flattening to a 3D batched gemv is the same math
    # and is supported.
    fa_tmap = jnp.fft.fft2(jnp.moveaxis(tmap, -1, 0), s=(pnx, pny))  # (n_freq, pnx, pny)

    tem_f = jnp.moveaxis(tem_batch, -1, 1)  # (n_theta, n_freq, nx, ny)
    tem_fft = jnp.fft.fft2(tem_f) * fft_norm
    tem_fft_a = jnp.moveaxis(tem_fft * a_vec[None, :, None, None], 1, -1)  # (nt, nx, ny, nf)

    a = tem_fft_a.reshape((n_theta * n_pix, n_freq))
    inv_b = jnp.broadcast_to(inv_cov, (n_theta, nx, ny_dim, n_freq, n_freq)).reshape(
        (n_theta * n_pix, n_freq, n_freq)
    )
    filter_fft = jnp.einsum("bi,bij->bj", a, inv_b).reshape((n_theta, nx, ny_dim, n_freq))
    filter_fft = jnp.where(jnp.isnan(filter_fft), 0., filter_fft)

    filter_val = jnp.fft.ifft2(jnp.moveaxis(filter_fft, -1, 1)).real * ifft_norm
    filter_val = jnp.moveaxis(filter_val, 1, -1).astype(tmap.dtype)

    tem_a = tem_batch * a_vec[None, None, None, :]
    norm = jnp.sum(tem_a * filter_val, axis=(1, 2, 3)) * dxdy  # (n_theta,)

    fb = jnp.fft.fft2(jnp.moveaxis(filter_val.astype(conv_dtype), -1, 1), s=(pnx, pny))
    conv = jnp.fft.ifft2(jnp.sum(fa_tmap[None, :, :, :] * fb, axis=1)).real
    start0 = (nx - 1) // 2
    start1 = (ny_dim - 1) // 2
    y_maps = conv[:, start0:start0 + nx, start1:start1 + ny_dim].astype(jnp.float64) * dxdy
    y_maps = y_maps / norm[:, None, None]
    stds = 1. / jnp.sqrt(norm)
    q_maps = y_maps / stds[:, None, None]
    return q_maps, y_maps, stds


def get_mmf_q_maps_batch(tmap, tem_list, inv_cov, pix, a_vec):
    """Compute q_maps for all templates in a single batched GPU call.

    tmap: (nx, ny, n_freq) observed map
    tem_list: list of (nx, ny, n_freq) templates
    inv_cov: (nx, ny, n_freq, n_freq)
    pix: pixel object
    a_vec: (n_freq,) SED vector
    Returns: q_maps (n_theta, nx, ny), y_maps (n_theta, nx, ny), stds (n_theta,)
    """
    tem_batch = jnp.stack(tem_list, axis=0)  # (n_theta, nx, ny, n_freq)

    nx, ny = pix.nx, pix.ny
    pnx = utils._next_fast_fft_len(2 * nx - 1)
    pny = utils._next_fast_fft_len(2 * ny - 1)

    return _get_mmf_q_maps_batch_jit(
        jnp.asarray(tmap), tem_batch, inv_cov, a_vec,
        jnp.float64(pix.dx), jnp.float64(pix.dy), pnx, pny)


def get_mmf_q_maps_tiles(tmaps, tem_list, inv_covs, pix, a_vec,
                        b_vec=None, a_dot_b=None, b_dot_b=None):
    """Matched filter for a stack of tiles, templates shared.

    tmaps: (n_tile, nx, ny, n_freq)
    tem_list: list of (nx, ny, n_freq)
    inv_covs: (n_tile, nx, ny, n_freq, n_freq)
    Returns q (n_tile, n_theta, nx, ny), y, stds (n_tile, n_theta)

    FFTs and the 6×6 gemv run over all tiles at once. Theta is scanned so we
    never broadcast inv_cov over theta (that 5D einsum crashes on Blackwell).
    If ``b_vec`` is set, apply one-dep sciMMF
    ``C^{-1}(a⊙T) - (a·C^{-1}b)/(b·C^{-1}b) C^{-1}(b⊙T)``.
    """
    tem_batch = jnp.stack(tem_list, axis=0)
    nx, ny = pix.nx, pix.ny
    pnx = utils._next_fast_fft_len(2 * nx - 1)
    pny = utils._next_fast_fft_len(2 * ny - 1)
    tmaps_j = jnp.asarray(tmaps)
    inv_j = jnp.asarray(inv_covs)
    dx = jnp.float64(pix.dx)
    dy = jnp.float64(pix.dy)
    if b_vec is None:
        q, y, stds = _get_mmf_q_maps_tiles_jit(
            tmaps_j, tem_batch, inv_j, a_vec, dx, dy, pnx, pny)
    else:
        q, y, stds = _get_scimmf_q_maps_tiles_jit(
            tmaps_j, tem_batch, inv_j, a_vec, jnp.asarray(b_vec),
            jnp.asarray(a_dot_b), jnp.asarray(b_dot_b), dx, dy, pnx, pny)
    return q, y, stds


@functools.partial(jax.jit, static_argnums=(6, 7))
def _get_mmf_q_maps_tiles_jit(tmaps, tem_batch, inv_covs, a_vec, dx, dy, pnx, pny):
    """Tile-parallel MMF. tmaps (T, nx, ny, nf), tem (n_theta, nx, ny, nf).

    Scan over theta so we never materialise (n_theta, T, nf, nx, ny) complex
    (that was 12 GB at T=16 and OOM'd). Each step still FFTs every tile at once.
    """
    n_tile, nx, ny_dim, n_freq = tmaps.shape
    n_pix = nx * ny_dim
    fft_norm = jnp.sqrt((dx * dy) / (nx * ny_dim))
    ifft_norm = jnp.sqrt((nx * ny_dim) / (dx * dy))
    dxdy = dx * dy
    conv_dtype = tmaps.dtype
    start0 = (nx - 1) // 2
    start1 = (ny_dim - 1) // 2

    fa_tmap = jnp.fft.fft2(jnp.moveaxis(tmaps, -1, 1), s=(pnx, pny))

    tem_f = jnp.moveaxis(tem_batch, -1, 1)
    tem_fft = jnp.fft.fft2(tem_f) * fft_norm
    tem_fft_a = jnp.moveaxis(tem_fft * a_vec[None, :, None, None], 1, -1)
    tem_a = tem_batch * a_vec[None, None, None, :]

    inv_b = inv_covs.reshape((n_tile * n_pix, n_freq, n_freq))

    def one_theta(tem_a_fft, tem_a_map):
        a = jnp.broadcast_to(tem_a_fft, (n_tile, nx, ny_dim, n_freq)).reshape(
            (n_tile * n_pix, n_freq))
        filt = jnp.einsum("bi,bij->bj", a, inv_b).reshape(
            (n_tile, nx, ny_dim, n_freq))
        filt = jnp.where(jnp.isnan(filt), 0., filt)
        filter_val = jnp.fft.ifft2(jnp.moveaxis(filt, -1, 1)).real * ifft_norm
        filter_val = jnp.moveaxis(filter_val, 1, -1).astype(tmaps.dtype)
        norm = jnp.sum(tem_a_map[None] * filter_val, axis=(1, 2, 3)) * dxdy
        fb = jnp.fft.fft2(jnp.moveaxis(filter_val.astype(conv_dtype), -1, 1), s=(pnx, pny))
        conv = jnp.fft.ifft2(jnp.sum(fa_tmap * fb, axis=1)).real
        y = conv[:, start0:start0 + nx, start1:start1 + ny_dim].astype(jnp.float64) * dxdy
        y = y / norm[:, None, None]
        std = 1. / jnp.sqrt(norm)
        q = y / std[:, None, None]
        return q, y, std

    def step(_, inputs):
        return None, one_theta(*inputs)

    q, y, stds = jax.lax.scan(step, None, (tem_fft_a, tem_a))[1]
    return jnp.swapaxes(q, 0, 1), jnp.swapaxes(y, 0, 1), jnp.swapaxes(stds, 0, 1)


@functools.partial(jax.jit, static_argnums=(9, 10))
def _get_scimmf_q_maps_tiles_jit(
    tmaps, tem_batch, inv_covs, a_vec, b_vec, a_dot_b, b_dot_b, dx, dy, pnx, pny
):
    """Tile-parallel one-dep sciMMF. Same layout as ``_get_mmf_q_maps_tiles_jit``."""
    n_tile, nx, ny_dim, n_freq = tmaps.shape
    n_pix = nx * ny_dim
    fft_norm = jnp.sqrt((dx * dy) / (nx * ny_dim))
    ifft_norm = jnp.sqrt((nx * ny_dim) / (dx * dy))
    dxdy = dx * dy
    conv_dtype = tmaps.dtype
    start0 = (nx - 1) // 2
    start1 = (ny_dim - 1) // 2

    fa_tmap = jnp.fft.fft2(jnp.moveaxis(tmaps, -1, 1), s=(pnx, pny))

    tem_f = jnp.moveaxis(tem_batch, -1, 1)
    tem_fft = jnp.fft.fft2(tem_f) * fft_norm
    tem_fft_a = jnp.moveaxis(tem_fft * a_vec[None, :, None, None], 1, -1)
    tem_fft_b = jnp.moveaxis(tem_fft * b_vec[None, :, None, None], 1, -1)
    tem_a = tem_batch * a_vec[None, None, None, :]

    inv_b = inv_covs.reshape((n_tile * n_pix, n_freq, n_freq))
    lam = a_dot_b / b_dot_b
    lam = jnp.where(jnp.isfinite(lam), lam, 0.)

    def one_theta(tem_a_fft, tem_a_map, tem_b_fft):
        a = jnp.broadcast_to(tem_a_fft, (n_tile, nx, ny_dim, n_freq)).reshape(
            (n_tile * n_pix, n_freq))
        b = jnp.broadcast_to(tem_b_fft, (n_tile, nx, ny_dim, n_freq)).reshape(
            (n_tile * n_pix, n_freq))
        filt_a = jnp.einsum("bi,bij->bj", a, inv_b).reshape(
            (n_tile, nx, ny_dim, n_freq))
        filt_b = jnp.einsum("bi,bij->bj", b, inv_b).reshape(
            (n_tile, nx, ny_dim, n_freq))
        filt = filt_a - lam[..., None] * filt_b
        filt = jnp.where(jnp.isnan(filt), 0., filt)
        filter_val = jnp.fft.ifft2(jnp.moveaxis(filt, -1, 1)).real * ifft_norm
        filter_val = jnp.moveaxis(filter_val, 1, -1).astype(tmaps.dtype)
        norm = jnp.sum(tem_a_map[None] * filter_val, axis=(1, 2, 3)) * dxdy
        fb = jnp.fft.fft2(jnp.moveaxis(filter_val.astype(conv_dtype), -1, 1), s=(pnx, pny))
        conv = jnp.fft.ifft2(jnp.sum(fa_tmap * fb, axis=1)).real
        y = conv[:, start0:start0 + nx, start1:start1 + ny_dim].astype(jnp.float64) * dxdy
        y = y / norm[:, None, None]
        std = 1. / jnp.sqrt(norm)
        q = y / std[:, None, None]
        return q, y, std

    def step(_, inputs):
        return None, one_theta(*inputs)

    q, y, stds = jax.lax.scan(step, None, (tem_fft_a, tem_a, tem_fft_b))[1]
    return jnp.swapaxes(q, 0, 1), jnp.swapaxes(y, 0, 1), jnp.swapaxes(stds, 0, 1)


def scmmf_one_dep_sed_ffts(pix, freqs, inv_cov, lrange, beam_type, exp, a_matrix, mask=None):
    """Beam-convolved tSZ/CIB SED maps used in one-dep ``a_dot_b`` / ``b_dot_b``."""
    if mask is None:
        mask = jnp.ones((pix.nx, pix.ny, len(freqs)))
    sed_a = a_matrix[:, 0][freqs]
    sed_b = a_matrix[:, 1][freqs]
    a_map_fft = jnp.ones((pix.nx, pix.nx, len(freqs)))
    a_map_fft = maps.get_tmap_times_fvec(a_map_fft, sed_a)
    a_map_fft = maps.get_map_convolved_fft(a_map_fft, pix, freqs, beam_type, mask, lrange, exp)
    a_map_fft = maps.reshape_ell_matrix(a_map_fft, inv_cov.shape[:2])
    b_map_fft = jnp.ones((pix.nx, pix.nx, len(freqs)))
    b_map_fft = maps.get_tmap_times_fvec(b_map_fft, sed_b)
    b_map_fft = maps.get_map_convolved_fft(b_map_fft, pix, freqs, beam_type, mask, lrange, exp)
    b_map_fft = maps.reshape_ell_matrix(b_map_fft, inv_cov.shape[:2])
    return a_map_fft, b_map_fft


def scmmf_one_dep_dots_tiles(a_map_fft, b_map_fft, inv_covs):
    """``a† C^{-1} b`` and ``b† C^{-1} b`` for a stack of tile inverse covariances."""
    ac = jnp.conjugate(a_map_fft)
    bc = jnp.conjugate(b_map_fft)
    filt_a = jnp.einsum("xyi,txyij->txyj", ac, inv_covs)
    a_dot_b = jnp.einsum("txyj,xyj->txy", filt_a, b_map_fft)
    filt_b = jnp.einsum("xyi,txyij->txyj", bc, inv_covs)
    b_dot_b = jnp.einsum("txyj,xyj->txy", filt_b, b_map_fft)
    return a_dot_b, b_dot_b


def get_tem_conv_fft(pix_tem,tem_fft,inv_cov,mmf_type="standard",cmmf_prec=None,comp=0):

    tem_fft = maps.reshape_ell_matrix(tem_fft, inv_cov.shape[:2])

    if mmf_type == "standard":

        tem_fft = maps.get_tmap_times_fvec(tem_fft,cmmf_prec.a_matrix[:,comp])
        filter_fft = utils.get_inv_cov_conjugate(tem_fft,inv_cov)

    elif mmf_type == "spectrally_constrained":

        if cmmf_prec.cmmf_type == "one_dep":

            n_freq = tem_fft.shape[2]
            a_dot_b = maps.get_tmap_from_map(cmmf_prec.a_dot_b,n_freq)
            b_dot_b = maps.get_tmap_from_map(cmmf_prec.b_dot_b,n_freq)
            b = cmmf_prec.a_matrix[:,1]
            a = cmmf_prec.a_matrix[:,0]

            tem_fft_b = maps.get_tmap_times_fvec(tem_fft,b)
            tem_fft = maps.get_tmap_times_fvec(tem_fft,a)

            filter_fft = utils.get_inv_cov_conjugate(tem_fft,inv_cov) - a_dot_b/b_dot_b*utils.get_inv_cov_conjugate(tem_fft_b,inv_cov)

        elif cmmf_prec.cmmf_type == "general":

            filter_fft_0 = cmmf_prec.w_norm_list[comp]
            filter_fft = filter_fft_0*tem_fft

    filter_fft = jnp.where(jnp.isnan(filter_fft), 0., filter_fft)
    filter_fft = maps.reshape_ell_matrix(filter_fft, (pix_tem.nx, pix_tem.ny))

    return filter_fft

#Class to compute weights for spectrally constrained MMF

class scmmf_precomputation:

    #sed's in muK

    def __init__(self,pix=None,freqs=None,inv_cov=None,lrange=[0,1000000],beam_type="gaussian",exp=None,
    mask=None,cmmf_type="one_dep",a_matrix=None,n_dep=1,comp_to_calculate=[0],mmf_type="standard"):

        self.cmmf_type = cmmf_type
        self.a_matrix = a_matrix

        if mask is None:

            mask = jnp.ones((pix.nx,pix.ny,len(freqs)))

        if self.cmmf_type == "one_dep" and mmf_type == "spectrally_constrained":

            self.sed_a = a_matrix[:,0][freqs]
            self.sed_b = a_matrix[:,1][freqs]

            a_map_fft = jnp.ones((pix.nx,pix.nx,len(freqs)))
            a_map_fft = maps.get_tmap_times_fvec(a_map_fft,self.sed_a)

            a_map_fft = maps.get_map_convolved_fft(a_map_fft,pix,freqs,beam_type,mask,lrange,exp)

            a_map_fft = maps.reshape_ell_matrix(a_map_fft, inv_cov.shape[:2])

            b_map_fft = jnp.ones((pix.nx,pix.nx,len(freqs)))
            b_map_fft = maps.get_tmap_times_fvec(b_map_fft,self.sed_b)

            b_map_fft = maps.get_map_convolved_fft(b_map_fft,pix,freqs,beam_type,mask,lrange,exp)
            b_map_fft = maps.reshape_ell_matrix(b_map_fft, inv_cov.shape[:2])

            self.a_dot_b = utils.get_inv_cov_dot(jnp.conjugate(a_map_fft),inv_cov,b_map_fft)
            self.b_dot_b = utils.get_inv_cov_dot(jnp.conjugate(b_map_fft),inv_cov,b_map_fft)

            del a_map_fft
            del b_map_fft

        elif self.cmmf_type == "general" and mmf_type == "spectrally_constrained":

            a_matrix_fft = jnp.zeros((pix.nx,pix.nx,len(freqs),self.a_matrix.shape[1]))

            for i in range(0,a_matrix.shape[1]):

                a_matrix_fft_i = jnp.ones((pix.nx,pix.nx,len(freqs)))
                a_matrix_fft_i = maps.get_tmap_times_fvec(a_matrix_fft_i,self.a_matrix[freqs,i])
                a_matrix_fft = a_matrix_fft.at[:,:,:,i].set(maps.get_map_convolved_fft(a_matrix_fft_i,pix,freqs,beam_type,mask,lrange,exp))

            del a_matrix_fft_i

            a_matrix_fft = maps.reshape_ell_matrix(a_matrix_fft, inv_cov.shape[:2])

            pre = jnp.einsum("abcd,abde->abce",inv_cov,a_matrix_fft)
            dot = utils.invert_cov(jnp.einsum("abdc,abde->abce",a_matrix_fft,pre))

            del a_matrix_fft

            W = jnp.einsum("abcd,abed->abce",dot,pre)
            cov = utils.invert_cov(inv_cov)

            self.w_norm_list = []

            for i in range(0,len(comp_to_calculate)):

                w = W[:,:,comp_to_calculate[i]]

                sigma_y = utils.get_inv_cov_dot(jnp.conjugate(w),cov,w)

                w_norm = w / sigma_y[:,:,None]

                self.w_norm_list.append(w_norm)

        elif mmf_type == "standard":

            r = 1.

        else:

            if mask is None:

                mask = jnp.ones((pix.nx,pix.ny,len(freqs)))

            if self.cmmf_type == "one_dep" and mmf_type == "spectrally_constrained":

                self.sed_a = a_matrix[:,0][freqs]
                self.sed_b = a_matrix[:,1][freqs]

                a_map_fft = jnp.ones((pix.nx,pix.nx,len(freqs)))
                a_map_fft = maps.get_tmap_times_fvec(a_map_fft,self.sed_a)

                a_map_fft = maps.get_map_convolved_fft(a_map_fft,pix,freqs,beam_type,mask,lrange,exp)

                b_map_fft = jnp.ones((pix.nx,pix.nx,len(freqs)))
                b_map_fft = maps.get_tmap_times_fvec(b_map_fft,self.sed_b)

                b_map_fft = maps.get_map_convolved_fft(b_map_fft,pix,freqs,beam_type,mask,lrange,exp)

                self.a_dot_b = utils.get_inv_cov_dot(jnp.conjugate(a_map_fft),inv_cov,b_map_fft)
                self.b_dot_b = utils.get_inv_cov_dot(jnp.conjugate(b_map_fft),inv_cov,b_map_fft)

            elif self.cmmf_type == "general" and mmf_type == "spectrally_constrained":

                a_matrix_fft = jnp.zeros((pix.nx,pix.nx,len(freqs),self.a_matrix.shape[1]))

                for i in range(0,a_matrix.shape[1]):

                    a_matrix_fft_i = jnp.ones((pix.nx,pix.nx,len(freqs)))
                    a_matrix_fft_i = maps.get_tmap_times_fvec(a_matrix_fft_i,self.a_matrix[freqs,i])
                    a_matrix_fft = a_matrix_fft.at[:,:,:,i].set(maps.get_map_convolved_fft(a_matrix_fft_i,pix,freqs,beam_type,mask,lrange,exp))

                self.a_matrix_fft = a_matrix_fft

                pre = jnp.einsum("abcd,abde->abce",inv_cov,a_matrix_fft)
                dot = utils.invert_cov(jnp.einsum("abdc,abde->abce",a_matrix_fft,pre))
                self.W = jnp.einsum("abcd,abed->abce",dot,pre)
                cov = utils.invert_cov(inv_cov)

                self.w_list = []
                self.sigma_y_list = []
                self.w_norm_list = []

                for i in range(0,len(comp_to_calculate)):

                    e_map = jnp.zeros((pix.nx,pix.nx,self.a_matrix.shape[1]))
                    e_map = e_map.at[:,:,comp_to_calculate[i]].set(1.)

                    w = jnp.einsum("ija,ijab->ijb",e_map,self.W)

                    sigma_y = utils.get_inv_cov_dot(jnp.conjugate(w),cov,w)

                    w_norm = w / sigma_y[:,:,None]

                    self.w_list.append(w)
                    self.sigma_y_list.append(sigma_y)
                    self.w_norm_list.append(w_norm)

def get_a_matrix_cib(params_szifi,params_model,data_file):

    cib = sed.cib_model(params_model=params_model)
    cib_sed = cib.get_sed_muK_experiment(experiment=data_file["experiment"],bandpass=params_szifi["integrate_bandpass"])
    cib.get_sed_first_moments_experiment(experiment=data_file["experiment"],
    bandpass=params_szifi["integrate_bandpass"],moment_parameters=params_szifi["deproject_cib"])

    freq_idx = np.asarray(params_szifi["freqs"])
    cib_sed = np.asarray(cib_sed)
    tsz_sed = np.asarray(data_file["experiment"].tsz_sed)

    a_matrix = np.zeros((len(freq_idx),len(params_szifi["deproject_cib"])))

    if params_szifi["deproject_cib"] == ["cib"]:

        a_matrix = np.zeros((len(freq_idx),2))
        a_matrix[:,0] = tsz_sed[freq_idx]
        a_matrix[:,1] = cib_sed[freq_idx]

    elif params_szifi["deproject_cib"] == ["cib","betaT"]:

        a_matrix = np.zeros((len(freq_idx),3))
        a_matrix[:,0] = tsz_sed[freq_idx]
        a_matrix[:,1] = cib_sed[freq_idx]
        a_matrix[:,2] = np.asarray(cib.moments["betaT"])[freq_idx]

    elif params_szifi["deproject_cib"] == ["cib","beta"]:

        a_matrix = np.zeros((len(freq_idx),3))
        a_matrix[:,0] = tsz_sed[freq_idx]
        a_matrix[:,1] = cib_sed[freq_idx]
        a_matrix[:,2] = np.asarray(cib.moments["beta"])[freq_idx]

    elif params_szifi["deproject_cib"] == ["cib","betaT","beta"]:

        a_matrix = np.zeros((len(freq_idx),4))
        a_matrix[:,0] = tsz_sed[freq_idx]
        a_matrix[:,1] = cib_sed[freq_idx]
        a_matrix[:,2] = np.asarray(cib.moments["betaT"])[freq_idx]
        a_matrix[:,3] = np.asarray(cib.moments["beta"])[freq_idx]

    params_szifi["a_matrix"] = a_matrix
