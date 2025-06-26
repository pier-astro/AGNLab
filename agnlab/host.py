from astropy.io import fits
from astropy import units as u
from astropy import constants as const
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

from .tools import resample_spectrum, vac_to_air, get_mask, compute_reduced_chi2
from .spectrum import Spectrum
plt.rcParams['axes.xmargin'] = 0

script_dir = os.path.dirname(__file__)
pathEigen = os.path.join(script_dir, "eigenspectra")
c = const.c.to(u.km/u.s).value # Speed of light in km/s

class HostDecompose:
    def __init__(self, spec:Spectrum):
        self.spec = spec
        self.results = None
        self.wave_obs = spec.wave
        self.flux_obs = spec.flux
        self.fluxerr_obs = spec.fluxerr
        self.wave = None
        self.flux = None
        self.fluxerr = None

        self.default_mask_regions = [
            (4950, 4970),  # [O III] 4959
            (5000, 5020),  # [O III] 5007
            (6540, 6590),  # H-alpha
            (3717, 3737),  # Mg II
            (4852, 4872),  # Fe II
            (4330, 4350),  # Fe II
            (6680, 6750),  # Na I D
            (4091, 4111)   # Ca II H&K
        ]
        self._fit_success = None
        self._decomp_success = None
        self.fit_redchi2 = None
        self.coefficients = None
        self.n_galcomp = None
        self.n_agncomp = None

        self.host = None
        self.agn = None
        self._host = None # Host spectrum after decomposition no matter if the fit was successful or not
        self._agn = None  # AGN spectrum after decomposition no matter if the fit was successful or not

        if self.spec._zcorrected == False:
            warnings.warn("The spectrum is not redshift corrected. The result might be incorrect. Please correct for redshift before host decomposition.", UserWarning)

        self.path_qso_PCA = os.path.join(pathEigen, "SDSS_50AGNSpectra_Yip2004.fits")
        self.path_gal_PCA = os.path.join(pathEigen, "SDSS_10GalaxySpectra_Yip2004.fits")

    def _load_eigen_spectra(self, n_galaxy=5, n_agn=10):
        """
        Load the eigen spectra for galaxies and AGN from the predefined paths.
        Returns the wavelength grid and the eigen spectra for galaxies and AGN.
        """
        if n_galaxy < 1 or n_agn < 1:
            raise ValueError("Number of galaxy eigenspectra and QSO eigenspectra has to be at least 1.")

        glx = fits.open(self.path_gal_PCA)
        gal_wave = glx[0].data # Wavelength extension
        gal_wave = vac_to_air(gal_wave)
        gal_spectra = glx[1].data # PCA extension with the 10 galaxy eigenspectra

        qso = fits.open(self.path_qso_PCA)
        qso_wave = qso[0].data # Wavelength extension
        qso_wave = vac_to_air(qso_wave)
        qso_spectra = qso[1].data # PCA extension with the 50 AGN eigenspectra

        n_max_galaxy = gal_spectra.shape[0]
        n_max_agn = qso_spectra.shape[0]
        if n_galaxy > n_max_galaxy:
            raise ValueError(f"Number of galaxy eigenspectra ({n_galaxy}) exceeds the maximum available ({n_max_galaxy}).")
        if n_agn > n_max_agn:
            raise ValueError(f"Number of AGN eigenspectra ({n_agn}) exceeds the maximum available ({n_max_agn}).")
        
        self.galaxy_wave = gal_wave
        self.galaxy_eigenspectra = gal_spectra[:n_galaxy]  # Select the first n_galaxy eigenspectra
        self.qso_wave = qso_wave
        self.qso_eigenspectra = qso_spectra[:n_agn] # Select the first n_agn eigenspectra

    def _prepare(self, mask=None, apply_eigen_limits=True, fill=np.nan):
        """
        Prepare the host decomposition by loading eigen spectra and rebining them to the spectrum wavelength grid.
        Optionally apply a custom mask.
        """
        if apply_eigen_limits:
            eigen_min_wave = np.min(self.spec.wave) if np.min(self.spec.wave) > 3450 else 3450 # Minimum wavelength where all eigenspectra are defined
            eigen_max_wave = np.max(self.spec.wave) if np.max(self.spec.wave) < 7950 else 7950 # Maximum wavelength where all eigenspectra are defined
            eigen_limits_mask = (self.spec.wave > eigen_min_wave) & (self.spec.wave < eigen_max_wave)
            if mask is not None:
                mask = np.logical_and(mask, eigen_limits_mask)  # Combine with the eigen limits mask
            else:
                mask = eigen_limits_mask
        else:
            if mask is None:
                mask = np.ones_like(self.spec.wave, dtype=bool) # Use all wavelengths
        
        wave_hs = self.spec.wave[mask]
        flux_hs, fluxerr_hs = resample_spectrum(self.spec.wave, self.spec.flux, self.spec.fluxerr, new_wave=wave_hs, fill=fill, method='flux-conserving', verbose=False)
        gal_spectra_hs = resample_spectrum(wave=self.galaxy_wave, flux=self.galaxy_eigenspectra, new_wave=wave_hs, fill=fill, method='flux-conserving', verbose=False)
        qso_spectra_hs = resample_spectrum(wave=self.qso_wave, flux=self.qso_eigenspectra, new_wave=wave_hs, fill=fill, method='flux-conserving', verbose=False)
        return wave_hs, flux_hs, fluxerr_hs, gal_spectra_hs, qso_spectra_hs
    
    @staticmethod
    def _check_host(hst, negative_threshold=100):
        """
        Check if the host spectrum is valid by counting the number of negative values.
        Returns True if the number of negative values is less than or equal to the threshold.

        Parameters
        ----------
        hst : array
            Host spectrum array to check.
        negative_threshold : int, optional
            Maximum allowed number of negative values (default: 100)

        Returns
        -------
        bool
            True if valid, False otherwise.
        """
        num_negative = np.sum(hst < 0.0)  # Count negative values
        return num_negative <= negative_threshold
    
    @staticmethod
    def _host_fitter(host_templates, agn_templates, observed_flux, observed_fluxerr=None):
        """
        Fit a linear combination of host galaxy and AGN templates to the observed spectrum using Weighted Least Squares.

        Parameters
        ----------
        host_templates : array-like
            Array of host galaxy template spectra (each as a vector).
        agn_templates : array-like
            Array of AGN template spectra (each as a vector).
        observed_flux : array-like
            Observed spectrum flux values.

        Returns
        -------
        coefficients : ndarray
            Best-fit coefficients for the host and AGN templates.
        """
        if observed_fluxerr is None:
            observed_fluxerr = np.sqrt(np.abs(observed_flux))

        # Build a mask for valid (finite) data across all inputs
        valid = (
            np.all(np.isfinite(host_templates), axis=0) &
            np.all(np.isfinite(agn_templates), axis=0) &
            np.isfinite(observed_flux) &
            np.isfinite(observed_fluxerr)
        )

        # Apply mask
        host_templates = host_templates[:, valid]
        agn_templates = agn_templates[:, valid]
        observed_flux = observed_flux[valid]
        observed_fluxerr = observed_fluxerr[valid]

        design_matrix = np.vstack([host_templates, agn_templates]).T # Stack host and AGN templates into a design matrix (columns = templates)
        weights = 1 / observed_fluxerr**2 # Calculate weights
        weighted_matrix = design_matrix * np.sqrt(weights[:, np.newaxis]) # Apply weights to the design matrix
        weighted_flux = observed_flux * np.sqrt(weights) # Apply weights to the observed data
        coefficients = np.linalg.lstsq(weighted_matrix, weighted_flux, rcond=None)[0] # Solve the weighted least squares problem
        return coefficients
    

    def fit(self, n_galaxy=5, n_agn=10, mask=None, use_default_mask=False):
        """
        Fit a fixed combination of n_galaxy host galaxy and n_qso AGN SDSS eigenspectra to the observed spectrum.
        Optionally mask emission lines in the host spectrum after fitting.

        Parameters
        ----------
        n_galaxy : int, optional
            Number of host galaxy eigenspectra to use (default: 5).
        n_agn : int, optional
            Number of AGN eigenspectra to use (default: 10).
        mask : array-like, optional
            Custom mask to apply to the observed spectrum (default: None).
        use_default_mask : bool, optional
            Whether to use the default mask regions for emission lines (default: False).
            The default mask is defined in the self.default_mask_regions atrribute and includes common emission lines like [O III], H-alpha, Mg II, Fe II, Na I D, and Ca II H&K.

        Raises
        ------
        ValueError
            If no valid data points are found after applying the mask, or if n_galaxy or n_agn are out of bounds.
        
        Notes
        -----
        This method prepares the host galaxy and AGN eigenspectra, applies the mask,
        and fits a linear combination of the eigenspectra to the observed spectrum.
        The fit is performed using a weighted least squares approach, and the results are stored in the
        `self.host` and `self.agn` attributes.
        Note that the fit is only performed if the number of negative values in the host and AGN spectra is below a certain threshold (default: 100).
        Concerning the mask, if `mask` is provided, it will be used directly, no matter if `use_default_mask` is set to True or False.
        The tools.get_mask() function can be used to easily create a mask based on the wavelength array and the regions to mask.
        """
        self.n_galcomp  = n_galaxy
        self.n_agncomp = n_agn
        self._load_eigen_spectra(n_galaxy=n_galaxy, n_agn=n_agn)

        wave_hs_fit, flux_hs_fit, fluxerr_hs_fit, gal_spectra_fit, qso_spectra_fit = self._prepare(apply_eigen_limits=True)

        # Fit host + AGN templates
        coefficients = self._host_fitter(gal_spectra_fit, qso_spectra_fit, flux_hs_fit, fluxerr_hs_fit)

        host_fit = sum(coefficients[i] * gal_spectra_fit[i] for i in range(n_galaxy))
        agn_fit = sum(coefficients[n_galaxy + i] * qso_spectra_fit[i] for i in range(n_agn))
        redchi2 = compute_reduced_chi2(flux_hs_fit, fluxerr_hs_fit, host_fit+agn_fit, n_galaxy + n_agn)
        self.fit_redchi2 = redchi2

        redchi2_limit = 50
        if redchi2 > redchi2_limit:
            self._fit_success = False
            print(f"Warning: Reduced chi-squared ({redchi2:.2f}) is greater than the limit ({redchi2_limit}). The fit might not be good.")
        else:
            self._fit_success = True
        
        negative_threshold = 100  # Threshold for number of negative values in the host and AGN spectra
        if not self._check_host(agn_fit, negative_threshold):
            print("Warning: The AGN component has a non-positive spectrum.\nThe AGN contribution might be negligible or the fit is not good.")
            self._decomp_success = False
        elif not self._check_host(host_fit, negative_threshold):
            print("Warning: The host galaxy component has a non-positive spectrum.\nThe host contribution might be negligible or the fit is not good.")
            self._decomp_success = False
        else:
            self._decomp_success = True

        self.wave, self.flux, self.fluxerr, host_eigen, agn_eigen = self._prepare(mask=None, apply_eigen_limits=False)
        self.coefficients = coefficients
        self._host = sum(coefficients[i] * host_eigen[i] for i in range(n_galaxy))
        self._agn = sum(coefficients[n_galaxy + i] * agn_eigen[i] for i in range(n_agn))

        if self._fit_success and self._decomp_success:
            self.host = self._host
            self.agn = self._agn
            self.spec.host = self.host
            self.spec.agn = self.spec.flux - self.host # SAVE THE DIFFERENCE TO REDUCE THE ERROR!
        else:
            # print("Not satisfying the decomposition internal conditions. Not saving the decomposition.\nIf needed the best fit Galaxy and AGN components can be created using the build_decomposition_spectra() method.")
            pass

    def build_decomposition_spectra(self, wave=None):
        """
        Build the host galaxy and AGN spectra based on the fitted coefficients.
        If wave is provided, resample the spectra to the new wavelength grid.
        """
        if wave is None:
            wave = self.spec.wave
        gal_spectra = resample_spectrum(wave=self.galaxy_wave, flux=self.galaxy_eigenspectra, new_wave=wave)
        qso_spectra = resample_spectrum(wave=self.qso_wave, flux=self.qso_eigenspectra, new_wave=wave)
        host_spectrum = sum(self.coefficients[i] * gal_spectra[i] for i in range(self.n_galcomp))
        agn_spectrum = sum(self.coefficients[self.n_galcomp + i] * qso_spectra[i] for i in range(self.n_agncomp))
        return host_spectrum, agn_spectrum
    
    def resample_eigenspectra(self, wave=None):
        """
        Resample the host galaxy and AGN eigenspectra to a new wavelength grid.
        If wave is None, use the spectrum's wavelength grid.
        """
        if wave is None:
            wave = self.spec.wave
        gal_spectra_resampled = resample_spectrum(wave=self.galaxy_wave, flux=self.galaxy_eigenspectra, new_wave=wave)
        qso_spectra_resampled = resample_spectrum(wave=self.qso_wave, flux=self.qso_eigenspectra, new_wave=wave)
        return gal_spectra_resampled, qso_spectra_resampled

    def plot_decomposition(self, ax=None):
        """
        Plot the observed spectrum, best-fit host galaxy, AGN, and total fit.
        """
        if not self._fit_success:
            print("Warning! The fit was not successful.")
        if not self._decomp_success:
            print("Warning! The decomposition was not successful.")
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        ax.plot(self.wave, self.spec.flux, label="Observed Spectrum", color='black')
        ax.plot(self.wave, self._host, label="Host Galaxy Fit", color='gray', linestyle='-')
        ax.plot(self.wave, self._agn, label="AGN Fit", color='blue', linestyle='-')
        ax.plot(self.wave, self._host+self._agn, label="Total Fit", color='red', linestyle='-')
        ax.set_title("Host Galaxy and AGN Decomposition")
        ax.set_xlabel("Wavelengt")
        ax.set_ylabel("Flux")
        ax.legend()
        if created_fig:
            plt.show()

    def plot_galaxy_components(self, ax=None):
        """
        Plot the individual galaxy eigenspectra multiplied by their coefficients.
        """
        if not self._decomp_success:
            print("Warning! The decomposition was not successful.")
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        gal_spectra, _ = self.resample_eigenspectra(self.wave)
        for i in range(self.n_galcomp):
            ax.plot(self.wave, gal_spectra[i] * self.coefficients[i], label=f"Component {i+1}", alpha=0.7)
        ax.set_title("Galaxy Components")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.legend(fontsize='small')
        if created_fig:
            plt.show()

    def plot_agn_components(self, ax=None):
        """
        Plot the individual AGN eigenspectra multiplied by their coefficients.
        """
        if not self._decomp_success:
            print("Warning! The decomposition was not successful.")
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        _, qso_spectra = self.resample_eigenspectra(self.wave)
        for i in range(self.n_agncomp):
            ax.plot(self.wave, qso_spectra[i] * self.coefficients[self.n_galcomp + i], label=f"Component {i+1}", alpha=0.7)
        ax.set_title("AGN Components")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.legend(fontsize='small')
        if created_fig:
            plt.show()