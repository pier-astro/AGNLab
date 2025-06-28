import os
import numpy as np
import warnings
from astropy.io import fits
# from astropy.wcs import WCS. GitHub issues. https://github.com/astropy/astropy/issues/887 , https://github.com/astropy/astropy/issues/1084
from mpdaf.obj.coords import WCS # Better handles keywords.
import astropy.units as u
import astropy.constants as const

import multiprocess as mp
from tqdm.auto import tqdm
import copy
import dill

from sherpa.data import Data1D
from sherpa.fit import Fit
from sherpa.optmethods import LevMar
from sherpa.stats import Chi2

from . import models
from . import tools

c = const.c.to(u.km/u.s).value # Speed of light in km/s

class CubeAnalyzer:
    def __init__(self):
        """
        Initialize the FitCube object with wavelength, data, and optional redshift.
        
        Parameters:
        wave (array): Wavelength array.
        data (array): Data array.
        z (float, optional): Redshift value. Defaults to None.
        """
        self.wave = None
        self.data = None
        self.dataerr = None
        self.z = None
        self._zcorrected = None
        self.model = None
        self.parnames = None
        self.parscube = None
        self.stat = None
        self.dof = None
        self.wcs = None
        self.snr_maps = None
        self.snr_lines = None
    
    def set_data(self, wave, data, datavar=None, z=None, wcs=None):
        """
        Set the data for the FitCube object.

        Parameters:
        wave (array): Wavelength array.
        data (array): Data array.
        datavar (array, optional): Variance of the data. Defaults to None.
        z (float, optional): Redshift value. Defaults to None.
        """
        self.wave = wave
        self.data = data
        self.dataerr = np.sqrt(datavar) if datavar is not None else np.ones_like(data)
        self.z = z if z is not None else 0.0
        self._zcorrected = False
        self.wcs = wcs

    def zCorrect(self, z=None):
        """
        Correct the wavelength for redshift.

        Parameters:
        z (float, optional): Redshift value. Defaults to None.
        """
        if z is not None:
            self.z = z
        self.wave = self.wave / (1 + self.z)
        self.data = self.data * (1 + self.z)
        if not np.all(self.dataerr == 1):
            self.dataerr = self.dataerr * (1 + self.z)
        self._zcorrected = True
    
    def set_model(self, model):
        self.model = model
        self.parnames = models.get_model_free_params_names(self.model)

    def _check_spec4fit(self):
        if self.wave is None or self.data is None or self.dataerr is None:
            raise ValueError("Data and wavelength must be set before fitting.")
        if np.any(np.isnan(self.wave)) or np.any(np.isnan(self.data)) or np.any(np.isnan(self.dataerr)):
            raise ValueError("Data contains NaN values in wave or flux. Please clean the data before fitting.")

    def _fit_NOparallelized(self, flat_indices, model, niter, stat, method,
                            initpars, bounds):
        parscube = np.zeros((len(model.thawedpars), self.data.shape[1], self.data.shape[2]), dtype=float)
        statvals = np.zeros((self.data.shape[1], self.data.shape[2]), dtype=float)

        if initpars is not None:
            if initpars.shape != parscube.shape:
                raise ValueError(f"initpars shape is not correct: {initpars.shape} != {parscube.shape}")

        if bounds is not None:
            if bounds.shape != (2, parscube.shape[0], parscube.shape[1], parscube.shape[2]):
                raise ValueError(f"bounds shape is not correct: {bounds.shape} != {(2, *parscube.shape)}")

        for yi, xi in tqdm(flat_indices, desc="Fitting Cube"):
            flux = self.data[:, yi, xi]
            fluxerr = self.dataerr[:, yi, xi]
            d = Data1D("spec", self.wave, flux, fluxerr)
            model_i = copy.deepcopy(model)
            if initpars is not None:
                model_i.thawedpars = initpars[:, yi, xi]
            if bounds is not None:
                model_i.thawedparmins = bounds[0, :, yi, xi]
                model_i.thawedparmaxes = bounds[1, :, yi, xi]
            gfit = Fit(d, model_i, stat=stat, method=method)
            for _ in range(niter):
                gres = gfit.fit()
            pars = model_i.thawedpars
            parscube[:, yi, xi] = pars
            statvals[yi, xi] = gres.statval
        dof = gres.dof
        return parscube, statvals, dof

    def _fit_parallelized(self, flat_indices, model, niter, stat, method,
                        initpars, bounds, ncpu):
        parscube = np.zeros((len(model.thawedpars), self.data.shape[1], self.data.shape[2]), dtype=float)
        statvals = np.zeros((self.data.shape[1], self.data.shape[2]), dtype=float)

        if initpars is not None:
            if initpars.shape != parscube.shape:
                raise ValueError(f"initpars shape is not correct: {initpars.shape} != {parscube.shape}")

        if bounds is not None:
            if bounds.shape != (2, parscube.shape[0], parscube.shape[1], parscube.shape[2]):
                raise ValueError(f"bounds shape is not correct: {bounds.shape} != {(2, *parscube.shape)}")

        def fit_worker(args):
            yi, xi, wave, flux, fluxerr, model, niter, stat, method, initpars, bounds_ = args
            d = Data1D("spec", wave, flux, fluxerr)
            model_i = copy.deepcopy(model)
            if initpars is not None:
                model_i.thawedpars = initpars
            if bounds_ is not None:
                model_i.thawedparmins = bounds_[0]
                model_i.thawedparmaxes = bounds_[1]
            gfit = Fit(d, model_i, stat=stat, method=method)
            for _ in range(niter):
                gres = gfit.fit()
            pars = model_i.thawedpars
            return yi, xi, pars, gres.statval, gres.dof

        args_list = []
        for yi, xi in flat_indices:
            ipars = initpars[:, yi, xi] if initpars is not None else None
            ibounds = bounds[:, :, yi, xi] if bounds is not None else None
            args_list.append((yi, xi, self.wave, self.data[:, yi, xi], self.dataerr[:, yi, xi], model, niter, stat, method, ipars, ibounds))

        with mp.Pool(processes=ncpu) as pool:
            results = list(tqdm(pool.imap(fit_worker, args_list), total=len(args_list), desc="Fitting Cube"))
        
        for yi, xi, pars, statval, dof in results:
            parscube[:, yi, xi] = pars
            statvals[yi, xi] = statval
            dof = dof
        return parscube, statvals, dof

    def fit(self, model=None, niter=1, stat=Chi2(), method=LevMar(),
            initpars=None, bounds=None, ncpu=None):
        """
        Note: To repeat a set of bounds for each pixel, you can use:
        ```python
        import numpy as np
        
        bounds = np.zeros((2, len(model.thawedpars)))
        bounds[0, :] = model.thawedparmins
        bounds[1, :] = model.thawedparmaxes
        bounds_expanded = np.tile(bounds[:, :, None, None], (1, 1, ny, nx))
        ```
        """
        self._check_spec4fit()
        if model is not None:
            self.model = model
            self.parnames = models.get_model_free_params_names(self.model)
        if self.model is None:
            raise ValueError("Model must be specified for fitting.")
        
        if not self._zcorrected:
            warnings.warn("Wavelengths are not redshift-corrected. Consider calling zCorrect() before fitting.", UserWarning)

        xi, yi = np.indices(self.data.shape[1:])
        flat_indices = np.array(list(zip(yi.flatten(), xi.flatten())))

        if ncpu == 1.:
            parscube, statvals, dof = self._fit_NOparallelized(flat_indices, model=self.model, niter=niter, stat=stat, method=method, initpars=initpars, bounds=bounds)
        else:
            ncpu = ncpu or max(1, mp.cpu_count() - 3)
            parscube, statvals, dof = self._fit_parallelized(flat_indices, model=self.model, niter=niter, stat=stat, method=method, initpars=initpars, bounds=bounds, ncpu=ncpu)

        self.parscube = parscube
        self.stat = statvals
        self.dof = dof
        self.redstat = statvals / dof



    def compute_snr_line(self, par_amp, par_fwhm, continuum, fwhm_multiplier=0.5, continuum_subtracted_data=None, continuum_subtracted_wave=None, is_offs_free=True):
        """
        Compute the SNR of a fitted line for the whole cube.

        Parameters:
            par (str or int): Amplitude parameter name or index.
            par_fwhm (str or int): FWHM parameter name or index.
            continuum (list of tuple): List of (start, end) wavelength ranges for continuum.
            subtract_continuum (bool): Whether to subtract continuum from the line.
            fwhm_multiplier (float): Multiplier for FWHM to define the line region.

        Returns:
            snr_map (2D np.ndarray): Signal-to-noise ratio map.
        """
        if not self._zcorrected:
            warnings.warn("Wavelengths are not redshift-corrected. Consider calling zCorrect() before computing SNR.", UserWarning)

        if continuum_subtracted_data is not None and continuum_subtracted_wave is not None:
            data = continuum_subtracted_data
            wave = continuum_subtracted_wave
        elif (continuum_subtracted_data is None) != (continuum_subtracted_wave is None):
            raise ValueError("Both continuum_subtracted_data and continuum_subtracted_wave must be provided together.")
        else:
            wave = self.wave
            data = self.data

        # Get amplitude and FWHM indices
        if isinstance(par_amp, str):
            if par_amp not in self.parnames:
                raise ValueError(f"Parameter '{par_amp}' not found in the model.")
            par_idx = self.parnames.index(par_amp)
        elif isinstance(par_amp, int):
            par_idx = par_amp
            if par_idx < 0 or par_idx >= len(self.parnames):
                raise IndexError(f"Parameter index {par_idx} out of range.")
        else:
            raise TypeError("par must be a string or integer.")
        if isinstance(par_fwhm, str):
            if par_fwhm not in self.parnames:
                raise ValueError(f"Parameter '{par_fwhm}' not found in the model.")
            fwhm_idx = self.parnames.index(par_fwhm)
        elif isinstance(par_fwhm, int):
            fwhm_idx = par_fwhm
            if fwhm_idx < 0 or fwhm_idx >= len(self.parnames):
                raise IndexError(f"Parameter index {fwhm_idx} out of range.")
        else:
            raise TypeError("par_fwhm must be a string or integer.")

        amp_map = self.parscube[par_idx]
        fwhm_kms_map = self.parscube[fwhm_idx]

        component_name = self.parnames[par_idx].split('.')[0] # parnames have the form 'component.param'
        mod_comp = models.get_comp_from_name(self.model, component_name)
        if is_offs_free:
            if f'{component_name}.offs_kms' not in self.parnames:
                raise ValueError(f"Parameter '{component_name}.offs_kms' not found in the model. Ensure the model has an 'offs_kms' parameter for this component.")
            offs_kms_idx = self.parnames.index(f'{component_name}.offs_kms')
            offs_kms_map = self.parscube[offs_kms_idx]
        else:
            offs_kms_map = np.zeros_like(amp_map)

        if hasattr(mod_comp, 'pos'):
            position = mod_comp.pos.val
        elif hasattr(mod_comp, 'dft'):
            line_name = self.parnames[par_idx].split('.')[-1][4:] # remove 'amp_' prefix
            position = mod_comp.dft[mod_comp.dft['name'] == line_name].iloc[0]['position']
        else:
            raise ValueError("Model component does not have 'pos' or 'df' attribute to determine line position.")
        
        line_center_map = position + (offs_kms_map * position / c)
        fwhm_map = fwhm_kms_map * position / c  # Convert FWHM from km/s to wavelength units

        # Estimate RMS in the continuum region
        c_mask = tools.get_mask(wave, continuum, mask_inside=True)
        flux_cont = data[c_mask, :, :]
        noise_map = np.std(flux_cont, axis=0)  # RMS noise in the continuum region

        em_or_abs = np.sign(amp_map)  # Determine if emission or absorption line

        # Compute the line region as the center ± FWHM * fwhm_multiplier in the wave array for each spectrum in data
        ny, nx = amp_map.shape
        snr_map = np.full((ny, nx), np.nan)
        min_w_step = np.min(np.diff(wave))  # Minimum wavelength step
        for yi in range(ny):
            for xi in range(nx):
                if np.isnan(amp_map[yi, xi]) or np.isnan(fwhm_kms_map[yi, xi]) or noise_map[yi, xi] == 0:
                    continue
                line_center = line_center_map[yi, xi]
                dw = fwhm_map[yi, xi] * fwhm_multiplier
                if dw < min_w_step:
                    dw = 2*min_w_step
                line_region_mask = tools.get_mask(wave, [(line_center - dw, line_center + dw)], mask_inside=False)
                line_flux = data[line_region_mask, yi, xi]
                if em_or_abs[yi, xi] > 0:
                    snr_map[yi, xi] = np.abs(np.max(line_flux)) / noise_map[yi, xi]
                else:
                    snr_map[yi, xi] = np.abs(np.min(line_flux)) / noise_map[yi, xi]
        return snr_map

    def set_snr_maps(self, maps, linenames):
        if maps.ndim != 3:
            raise ValueError("Maps must be a 3D array with shape (n_lines, ny, nx).")
        if maps.shape[0] != len(linenames):
            raise ValueError("Number of maps must match number of line names.")
        if maps.shape[1] != self.data.shape[1] or maps.shape[2] != self.data.shape[2]:
            raise ValueError("Maps shape must match the data shape (ny, nx).")
        
        self.snr_maps = maps
        self.snr_lines = linenames

    def save(self, filename, save_model=True):
        """
        Save the fitted parameters and statistics to a FITS file.

        Parameters:
        filename (str): The name of the output FITS file.
        """
        if self.parscube is None or self.stat is None or self.dof is None:
            raise ValueError("No fitted parameters or statistics to save. Please fit the cube first.")
        if not filename.endswith('.fits'):
            filename += '.fits'

        # Create an empty primary HDU
        primary_hdu = fits.PrimaryHDU()

        # Store the parameter cube in an extension HDU with a name
        params_hdu = fits.ImageHDU(self.parscube)
        params_hdu.header['EXTNAME'] = 'PARAMS'
        # Add parameter names to the header
        if self.parnames is None and self.model is not None:
            self.parnames = models.get_model_free_params_names(self.model)
        params_hdu.header['PARNAMES'] = ', '.join(self.parnames)
        
        # Add WCS information if available
        if self.wcs is not None:
            wcs_header = self.wcs.to_header()
            params_hdu.header.update(wcs_header)

        # Statistic image HDU as before
        stat_hdu = fits.ImageHDU(self.stat)
        stat_hdu.header['EXTNAME'] = 'FITSTAT'
        stat_hdu.header['DOF'] = self.dof

        hdus = [primary_hdu, params_hdu, stat_hdu]

        if self.snr_maps is not None and self.snr_lines is not None:
            if self.snr_maps.ndim != 3:
                raise ValueError("SNR maps must be a 3D array with shape (n_lines, ny, nx).")
            if self.snr_maps.shape[1:] != self.data.shape[1:]:
                raise ValueError("SNR maps shape must match the data shape (ny, nx).")
            snr_hdu = fits.ImageHDU(self.snr_maps)
            snr_hdu.header['EXTNAME'] = 'SNR'
            snr_hdu.header['LINES'] = ', '.join(self.snr_lines)
            hdus.append(snr_hdu)
        
        if save_model:
            modobj_bytes = dill.dumps(self.model)
            col = fits.Column(name="MODEL", format='B', array=np.frombuffer(modobj_bytes, dtype='uint8'))
            hdu_model = fits.BinTableHDU.from_columns([col], name='PYMOD')
            hdus.append(hdu_model)

        # Write the HDUs to a FITS file
        hdul = fits.HDUList(hdus)
        hdul.writeto(filename, overwrite=True)



    def save_model(self, filename):
        if self.model is None:
            raise ValueError("No model to save. Please setup CubeAnalyzer.model first.")
        if not filename.endswith('.dill'):
            filename += '.dill'
        with open(filename, 'wb') as f:
            dill.dump(self.model, f)
    
    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.model = dill.load(f)
    
    def load_fit(self, file, model_file=None):
        """
        Load a fitted cube from a FITS file and an optional model file.

        Parameters:
        file (str): The name of the FITS file containing the fitted parameters.
        model_file (str, optional): The name of the file containing the model. Defaults to None.
        """
        if model_file is None:
            self._load_params_and_model(file)
        else:
            self._load_params(file)
            self._load_model(model_file)

    def _load_params_and_model(self, file):
        """
        Load the parameters and model from the FITS file.
        """
        with fits.open(file) as hdul:
            params_hdu = hdul['PARAMS']
            stat_hdu = hdul['FITSTAT']

            # Load the WCS if available in the header of the PARAMS HDU
            if any(k in params_hdu.header for k in ('WCSAXES', 'CTYPE1', 'CRPIX1')):
                self.wcs = WCS(params_hdu.header)

            self.parscube = params_hdu.data
            self.stat = stat_hdu.data
            self.dof = stat_hdu.header['DOF']

            # Extract parameter names from the header
            parnames_str = params_hdu.header['PARNAMES']
            self.parnames = parnames_str.split(', ')

            if 'PYMOD' in hdul:
                modobj_bytes = hdul['PYMOD'].data.tobytes()
                self.model = dill.loads(modobj_bytes)
            else:
                self.model = None
                warnings.warn("No model found in the FITS file. Model will be None.", UserWarning)

            if 'SNR' in hdul:
                snr_hdu = hdul['SNR']
                self.snr_maps = snr_hdu.data
                self.snr_lines = snr_hdu.header['LINES'].split(', ')
    
    def _load_params(self, file):
        """
        Load the parameters and statistics from the FITS file.
        """
        with fits.open(file) as hdul:
            params_hdu = hdul['PARAMS']
            stat_hdu = hdul['FITSTAT']
            
            # Load the WCS if available in the header of the PARAMS HDU
            if any(k in params_hdu.header for k in ('WCSAXES', 'CTYPE1', 'CRPIX1')):
                self.wcs = WCS(params_hdu.header)

            self.parscube = params_hdu.data
            self.stat = stat_hdu.data
            self.dof = stat_hdu.header['DOF']

            # Extract parameter names from the header
            parnames_str = params_hdu.header['PARNAMES']
            self.parnames = parnames_str.split(', ')

            if 'SNR' in hdul:
                snr_hdu = hdul['SNR']
                self.snr_maps = snr_hdu.data
                self.snr_lines = snr_hdu.header['LINES'].split(', ')

    def _load_model(self, file):
        """
        Load the model from a file if provided.
        """
        with open(file, 'rb') as f:
            self.model = dill.load(f)

    def get_model(self, yi, xi):
        """
        Get the model parameters for a specific pixel.

        Parameters:
        yi (int): Y index of the pixel.
        xi (int): X index of the pixel.

        Returns:
        dict: Dictionary of parameter names and their values for the specified pixel.
        """
        pars = self.parscube[:, yi, xi]
        model_i = copy.deepcopy(self.model)
        model_i.thawedpars = pars
        return model_i
    
    def get_par_map(self, par):
        """
        Get a 2D map of a specific parameter across the cube.

        Parameters:
        parname (str or int): Name or index of the parameter to retrieve.

        Returns:
        np.ndarray: 2D array of the parameter values.
        """
        if isinstance(par, str):
            if par not in self.parnames:
                raise ValueError(f"Parameter '{par}' not found in the model.")
            par_index = self.parnames.index(par)
        elif isinstance(par, int):
            par_index = par
            if par_index < 0 or par_index >= len(self.parnames):
                raise IndexError(f"Parameter index {par_index} out of range.")
        else:
            raise TypeError("parname must be a string or integer.")
        return self.parscube[par_index, :, :]
    
    def get_snr_map(self, line):
        """
        Get the SNR map for a specific line.

        Parameters:
        line (str): Name of the line to retrieve the SNR map for.

        Returns:
        np.ndarray: 2D array of the SNR values for the specified line.
        """
        if self.snr_maps is None or self.snr_lines is None:
            raise ValueError("SNR maps are not available. Please compute SNR maps first.")
        if line not in self.snr_lines:
            raise ValueError(f"Line '{line}' not found in the SNR maps.")
        line_index = self.snr_lines.index(line)
        return self.snr_maps[line_index, :, :]