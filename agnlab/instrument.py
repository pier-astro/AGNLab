import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, issparse
from scipy.special import erf
from astropy.io import fits
from sherpa.models import Model, ArithmeticModel, CompositeModel
import warnings
import os
import yaml

def _get_rsp_path():
    """
    Returns the path to the response matrix file.
    This function is a placeholder and should be replaced with actual logic to get the response path.
    """
    # Replace with actual logic to get the response path
    return os.path.join(os.path.dirname(__file__), "responses")

def _get_responses_yaml_path():
    """Return the path to the responses.yaml file."""
    return os.path.join(os.path.dirname(__file__), "responses", "rsps.yaml")

def load_responses_mapping():
    """Load the instrument-to-FITS mapping from YAML."""
    yaml_path = _get_responses_yaml_path()
    if not os.path.exists(yaml_path):
        return {}
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f) or {}

def _save_responses_mapping(mapping):
    """Save the instrument-to-FITS mapping to YAML."""
    yaml_path = _get_responses_yaml_path()
    with open(yaml_path, "w") as f:
        yaml.safe_dump(mapping, f)

def add_response_to_archive(inst_name, fits_filename, clobber=False):
    """Add or update an instrument-FITS mapping in the archive.
    """
    mapping = load_responses_mapping()
    if inst_name in mapping and not clobber:
        raise ValueError(f"Instrument '{inst_name}' already exists in the responses archive. Use clobber=True to overwrite.")
    mapping[inst_name] = fits_filename
    _save_responses_mapping(mapping)


def crop_response_matrix(matrix, matrix_wave, target_wave, renormalize=True):
    """
    Crop a response matrix to match the target wavelengths.

    Parameters:
        matrix: 2D array or sparse matrix (N x N)
        matrix_wave: 1D array of wavelengths corresponding to the matrix
        target_wave: 1D array of desired wavelengths
        renormalize: bool, whether to renormalize rows for flux conservation

    Returns:
        Cropped (and optionally renormalized) matrix
    """
    if not np.all([np.any(np.isclose(dw, matrix_wave)) for dw in target_wave]):
        raise ValueError("Data wavelengths must be within the wavelength grid")
    # Ensure the data wavelengths have the same step size as the response matrix
    if not np.all(np.isclose(np.diff(target_wave), np.diff(matrix_wave)[0])):
        warnings.warn("Wavelengths do not have always the same step size as the response matrix.")

    indices = np.where([np.any(np.isclose(w, matrix_wave)) for w in target_wave])[0]
    cropped = matrix[np.ix_(indices, indices)]

    if renormalize:
        # Ensure flux conservation
        row_sums = cropped.sum(axis=1).A1 if issparse(cropped) else cropped.sum(axis=1)
        cropped = cropped.multiply(1 / row_sums[:, np.newaxis]) if issparse(cropped) else cropped / row_sums[:, np.newaxis]

    return cropped







class InstrumentResponse:
    """
    Unified class for building, loading, saving, cropping, and registering instrumental response matrices.
    Use classmethods to construct from parameters, FITS files, or instrument names.
    """
    def __init__(self, wavelength_grid, response_matrix):
        self.wavelength_grid = np.asarray(wavelength_grid)
        self.response_matrix = response_matrix

    # ---- BUILDERS ----
    @classmethod
    def from_variable_gaussian_resolution(cls, wavelength_grid, lambda_R, R_values, interp_kind='linear'):
        """Build from variable resolution R(lambda)."""
        sigmas = cls._compute_sigmas_variable(wavelength_grid, lambda_R, R_values, interp_kind)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_fwhm(cls, wavelength_grid, fwhm):
        """Build from fixed FWHM."""
        if fwhm <= 0:
            raise ValueError("FWHM must be positive.")
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigmas = np.full(len(wavelength_grid), sigma)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_resolution(cls, wavelength_grid, R):
        """Build from fixed resolution R."""
        if R <= 0:
            raise ValueError("Resolution R must be positive.")
        delta_lam = wavelength_grid / R
        sigmas = delta_lam / (2 * np.sqrt(2 * np.log(2)))
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_fixed_sigma(cls, wavelength_grid, sigma):
        """Build from fixed sigma."""
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        sigmas = np.full(len(wavelength_grid), sigma)
        matrix = cls._build_sparse_gaussian_matrix(wavelength_grid, sigmas)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_array(cls, wavelength_grid, matrix):
        """Create from a dense or sparse matrix."""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Response matrix must be square.")
        if matrix.shape[0] != len(wavelength_grid):
            raise ValueError("Response matrix dimensions do not match wavelength grid.")
        if not issparse(matrix):
            matrix = csr_matrix(matrix)
        return cls(wavelength_grid, matrix)

    # ---- LOADERS ----
    @classmethod
    def from_fits(cls, filename):
        """Load from FITS file."""
        with fits.open(filename) as hdul:
            wavelength_grid = hdul[1].data
            dense_matrix = hdul[2].data.astype(np.float64)
            matrix = csr_matrix(dense_matrix)
        return cls(wavelength_grid, matrix)

    @classmethod
    def from_instrument(cls, instrument):
        """Load from instrument name in the archive."""
        mapping = load_responses_mapping()
        if instrument not in mapping:
            raise ValueError(f"Unknown instrument: {instrument}. Available: {list(mapping.keys())}")
        filename = os.path.join(_get_rsp_path(), mapping[instrument])
        return cls.from_fits(filename)

    # ---- INTERNALS ----
    @staticmethod
    def _compute_sigmas_variable(wavelength_grid, lambda_R, R_values, interp_kind):
        _R_interp = interp1d(lambda_R, R_values, kind=interp_kind, bounds_error=False, fill_value=(R_values[0], R_values[-1]))
        R = np.maximum(_R_interp(wavelength_grid), 1.0)
        delta_lam = wavelength_grid / R
        return delta_lam / (2 * np.sqrt(2 * np.log(2)))

    @staticmethod
    def _build_sparse_gaussian_matrix(wavelength_grid, sigmas):
        N = len(wavelength_grid)
        wstep = np.diff(wavelength_grid)[0]
        data, row_indices, col_indices = [], [], []
        lower_edges = wavelength_grid - wstep / 2
        upper_edges = wavelength_grid + wstep / 2
        for i, (lambda_real, sigma) in enumerate(zip(wavelength_grid, sigmas)):
            a = (lower_edges - lambda_real) / (sigma * np.sqrt(2))
            b = (upper_edges - lambda_real) / (sigma * np.sqrt(2))
            integrals = 0.5 * (erf(b) - erf(a))
            row_sum = integrals.sum()
            if row_sum > 1e-9:
                integrals /= row_sum
            non_zero = integrals > 1e-10
            data.extend(integrals[non_zero])
            row_indices.extend([i] * np.sum(non_zero))
            col_indices.extend(np.where(non_zero)[0])
        return csr_matrix((data, (row_indices, col_indices)), shape=(N, N))

    # ---- SAVE, REGISTER, CROP ----
    def save_fits(self, filename, compress=True):
        """Save wavelength grid and matrix to FITS file."""
        primary_hdu = fits.PrimaryHDU()
        wavelength_hdu = fits.ImageHDU(self.wavelength_grid, name='WAVELENGTH')
        dense_matrix = self.response_matrix.toarray()
        if compress:
            matrix_hdu = fits.CompImageHDU(dense_matrix, name='RESPONSE')
        else:
            matrix_hdu = fits.ImageHDU(dense_matrix, name='RESPONSE')
        hdul = fits.HDUList([primary_hdu, wavelength_hdu, matrix_hdu])
        hdul.writeto(filename, overwrite=True)

    def save_and_register(self, filename, instrument_name, compress=True, clobber=False):
        """Save the response matrix and register it in the YAML mapping."""
        archive_filename = os.path.join(_get_rsp_path(), os.path.basename(filename))
        if os.path.exists(archive_filename) and not clobber:
            raise ValueError(f"Response file '{archive_filename}' already exists. Use clobber=True to overwrite.")
        self.save_fits(archive_filename, compress=compress)
        add_response_to_archive(instrument_name, filename, clobber=clobber)

    def crop(self, new_wavelengths, renormalize=True):
        """Crop the response matrix to match a new wavelength grid."""
        cropped_matrix = crop_response_matrix(self.response_matrix, self.wavelength_grid, new_wavelengths, renormalize=renormalize)
        return InstrumentResponse(new_wavelengths, cropped_matrix)

    def __repr__(self):
        return f"<InstrumentResponse(wavelength_grid={self.wavelength_grid.shape}, response_matrix={'set' if self.response_matrix is not None else 'unset'})>"







class ConvolvedModel(CompositeModel, ArithmeticModel):
    def __init__(self, response_model, source_model):
        self.response_model = response_model
        self.source_model = source_model
        # If the model has parts as an attribute, use CompositeModel
        CompositeModel.__init__(self, f'{response_model.name}({source_model.name})', (source_model, ))

    @property
    def pars(self):
        return self.source_model.pars

    def calc(self, pars, x, *args, **kwargs):
        source_eval = self.source_model.calc(pars, x, *args, **kwargs)
        return self.response_model.response_matrix.dot(source_eval)

class SpectralRsp(Model):
    """Sherpa model for instrumental spectral response.
    
    Parameters
    ----------
    response_matrix : array-like or sparse matrix
        2D response matrix (N x N) representing the instrumental response
    name : str, optional
        Model name (default: 'instrsp')
    """
    def __init__(self, response_matrix, name='instrsp'):
        if response_matrix is None:
            raise ValueError("response_matrix must be provided.")
        if not issparse(response_matrix):
            response_matrix = csr_matrix(response_matrix)
        self.response_matrix = response_matrix
        super().__init__(name)

    @classmethod
    def from_instrument(cls, instrument_name, wave=None, spectrum=None, renormalize=True, name='instrsp'):
        """Construct SpectralRsp using a registered instrument response.
        
        Parameters
        ----------
        instrument_name : str
            Name of the instrument (e.g., 'MUSE')
        wave : array-like, optional
            Wavelength grid. If not provided, must provide spectrum.
        spectrum : Spectrum, optional
            Spectrum object. Uses spectrum.observed_wave if available.
        renormalize : bool, optional
            Whether to renormalize response matrix rows (default: True)
        name : str, optional
            Model name
            
        Returns
        -------
        SpectralRsp
            Configured spectral response model
            
        Examples
        --------
        >>> # Pass spectrum object (recommended)
        >>> spec = Spectrum.from_txt('data.txt', z=0.1)
        >>> spec.zCorrect()
        >>> rsp = SpectralRsp.from_instrument('MUSE', spectrum=spec)
        """
        if wave is None and spectrum is None:
            raise ValueError("Either 'wave' or 'spectrum' must be provided.")
        
        if wave is None:
            if hasattr(spectrum, 'observed_wave'):
                wave = spectrum.observed_wave
            else:
                raise ValueError("Provided spectrum does not have 'observed_wave' property.")
        
        response = InstrumentResponse.from_instrument(instrument_name)
        cropped = response.crop(wave, renormalize=renormalize)
        
        return cls(cropped.response_matrix, name=name)

    @classmethod
    def from_fits(cls, filename, wave=None, spectrum=None, renormalize=True, name='instrsp'):
        """Construct SpectralRsp by loading from FITS file.
        
        Parameters
        ----------
        filename : str
            Path to FITS file containing response matrix
        wave : array-like, optional
            Wavelength grid. If not provided, must provide spectrum.
        spectrum : Spectrum, optional
            Spectrum object. Uses spectrum.observed_wave if available.
        renormalize : bool, optional
            Whether to renormalize response matrix rows (default: True)
        name : str, optional
            Model name
            
        Returns
        -------
        SpectralRsp
            Configured spectral response model
        """
        if wave is None and spectrum is None:
            raise ValueError("Either 'wave' or 'spectrum' must be provided.")
        
        if wave is None:
            if hasattr(spectrum, 'observed_wave'):
                wave = spectrum.observed_wave
            else:
                raise ValueError("Provided spectrum does not have 'observed_wave' property.")
        
        response = InstrumentResponse.from_fits(filename)
        cropped = response.crop(wave, renormalize=renormalize)
        
        return cls(cropped.response_matrix, name=name)

    def __call__(self, source_model):
        """Return a Sherpa model convolved with the stored response."""
        return ConvolvedModel(self, source_model)
    






# Example of usage
if __name__=='__main__':

    from sherpa.models.basic import Gauss1D
    import matplotlib.pyplot as plt

    # Create the response matrix
    rsp_wave = np.arange(5000, 7000, 1)

    resp_lambda = np.array([5000.0, 5500.0, 6000.0, 6500.0, 7000.0])
    resp_R = np.array([1695.0, 1750.0, 1978.0, 2227.0, 2484.0])

    # Create the response matrix using the unified InstrumentResponse class
    # Option 1: Build from variable resolution
    response = InstrumentResponse.from_variable_gaussian_resolution(rsp_wave, resp_lambda, resp_R)
    # Option 2: Build from fixed FWHM
    response = InstrumentResponse.from_fixed_fwhm(rsp_wave, 10.0)

    # # Save and load the response matrix
    # response.save_fits('response_matrix.fits')
    # response = InstrumentResponse.from_fits('response_matrix.fits')

    # Define the unfolded model and the energy grid
    gauss = Gauss1D('gauss')
    gauss.ampl = 1.0
    gauss.pos = 6000.0
    gauss.fwhm = 5
    wave = np.arange(5950, 6050, 1)

    # Crop the response matrix to match the wavelength grid
    cropped_response = response.crop(wave)
    rsp_matrix = cropped_response.response_matrix

    # Define the spectral response model and apply it to the unfolded model
    rsp = SpectralRsp(rsp_matrix)
    convolved_model = rsp(gauss)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(wave, gauss(wave), label='Original Flux', color='blue')
    plt.plot(wave, convolved_model(wave), label='Convolved Flux', color='red')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')
    plt.title('Instrumental Response Convolution')
    plt.margins(x=0.)
    plt.legend()
    plt.show()