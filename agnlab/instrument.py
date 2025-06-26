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







class InstRspBuilder:
    """Handles creation and storage of the instrumental response matrix"""
    def __init__(self, wave_grid):
        self.wave_grid = wave_grid
        self.response_matrix = None

        self.lambda_R = None
        self.R_values = None

        w_diff = np.diff(wave_grid)
        if not np.all(np.isclose(w_diff, w_diff[0])):
            raise ValueError("Wavelength grid must be uniform")
        self.wstep = w_diff[0]
    
    def _build_sparse_gaussian_matrix(self, sigmas):
        """
        Private helper to build a sparse Gaussian response matrix given sigma values for each row.

        Args:
            sigmas (np.ndarray): An array of sigma values, one for each row in the response matrix.
        """
        N = len(self.wave_grid)
        data, row_indices, col_indices = [], [], []
        lower_edges = self.wave_grid - self.wstep / 2
        upper_edges = self.wave_grid + self.wstep / 2

        for i, (lambda_real, sigma) in enumerate(zip(self.wave_grid, sigmas)):
            # Compute Gaussian integral over each bin
            a = (lower_edges - lambda_real) / (sigma * np.sqrt(2))
            b = (upper_edges - lambda_real) / (sigma * np.sqrt(2))
            integrals = 0.5 * (erf(b) - erf(a))

            # Normalize row to account for any truncation and prevent division by zero
            row_sum = integrals.sum()
            if row_sum > 1e-9:
                integrals /= row_sum

            # Store non-zero values and their indices
            non_zero = integrals > 1e-10  # Threshold to ignore very small values
            data.extend(integrals[non_zero])
            row_indices.extend([i] * np.sum(non_zero))
            col_indices.extend(np.where(non_zero)[0])

        # Create and store the sparse matrix
        self.response_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
        return self.response_matrix

    def build_gaussian_matrix(self, lambda_R, R_values, interp_kind='linear'):
        """Build the response matrix using a variable resolution R."""
        self.lambda_R = np.array(lambda_R)
        self.R_values = np.array(R_values)
        _R_interp = interp1d(self.lambda_R, self.R_values, kind=interp_kind, bounds_error=False, fill_value=(self.R_values[0], self.R_values[-1]))

        R = np.maximum(_R_interp(self.wave_grid), 1.0)
        delta_lam = self.wave_grid / R
        sigmas = delta_lam / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        return self._build_sparse_gaussian_matrix(sigmas)

    def build_fixed_fwhm_matrix(self, fwhm):
        """Build the response matrix using a fixed FWHM for the Gaussian response."""
        if fwhm <= 0:
            raise ValueError("FWHM must be positive.")
        
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigmas = np.full(len(self.wave_grid), sigma)
        return self._build_sparse_gaussian_matrix(sigmas)

    def build_fixed_r_matrix(self, R):
        """Build the response matrix using a fixed resolution R."""
        if R <= 0:
            raise ValueError("Resolution R must be positive.")

        delta_lam = self.wave_grid / R
        sigmas = delta_lam / (2 * np.sqrt(2 * np.log(2)))
        return self._build_sparse_gaussian_matrix(sigmas)

    def build_fixed_sigma_matrix(self, sigma):
        """Build the response matrix using a fixed sigma for the Gaussian response."""
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        
        sigmas = np.full(len(self.wave_grid), sigma)
        return self._build_sparse_gaussian_matrix(sigmas)
    
    def load_matrix_from_array(self, matrix):
        """Load response matrix from a 2D numpy array or sparse matrix"""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Response matrix must be square")
        if matrix.shape[0] != len(self.wave_grid):
            raise ValueError("Response matrix dimensions do not match wavelength grid")
        
        # Convert to sparse matrix if it's not already sparse
        if not issparse(matrix):
            matrix = csr_matrix(matrix)
        
        self.response_matrix = matrix

    def save_to_fits(self, filename, compress=True):
        """Save wavelength grid and matrix to FITS file"""
        if self.response_matrix is None:
            raise ValueError("Build matrix first using build_matrix()")
            
        primary_hdu = fits.PrimaryHDU()
        wavelength_hdu = fits.ImageHDU(self.wave_grid, name='WAVELENGTH')
        
        # Convert sparse matrix to dense before saving
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
        self.save_to_fits(archive_filename, compress=compress)
        add_response_to_archive(instrument_name, filename, clobber=clobber)

    def crop_matrix(self, new_wave, renormalize=True):
        return crop_response_matrix(self.response_matrix, self.wave_grid, new_wave, renormalize=renormalize)
    

class InstRspLoader:
    """Handles loading and preparation of response matrices"""
    def __init__(self, filename=None, inst=None):
        """
        Initialize the loader. Either `filename` or `inst` must be provided.
        
        - filename: Path to the FITS file containing the response matrix.
        - inst: Instrument name to load the precomputed response file.
        """
        if filename is None and inst is None:
            raise ValueError("Either `filename` or `inst` must be provided.")
        
        if inst is not None:
            responses_dir = os.path.join(os.path.dirname(__file__), "responses")
            mapping = load_responses_mapping()
            if inst not in mapping:
                raise ValueError(f"Unknown instrument: {inst}. Available: {list(mapping.keys())}")
            self.filename = os.path.join(responses_dir, mapping[inst])
        else:
            self.filename = filename
        
        self.wave = None
        self.full_matrix = None
        self._load_matrix()
    
    def _load_matrix(self):
        """Load matrix and wavelengths from FITS file"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"Response file not found: {self.filename}")
        
        with fits.open(self.filename) as hdul:
            self.wave = hdul[1].data
            dense_matrix = hdul[2].data
            # Ensure the matrix has a supported data type
            dense_matrix = dense_matrix.astype(np.float64)  # Convert to float64 if necessary
            # Convert the dense matrix to a sparse matrix
            self.full_matrix = csr_matrix(dense_matrix)
    
    def crop_matrix(self, wave, renormalize=True):
        return crop_response_matrix(self.full_matrix, self.wave, wave, renormalize=renormalize)







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
    def __init__(self, response_matrix, name='instrsp'):
        self.response_matrix = response_matrix
        super().__init__(name)

    def __call__(self, source_model):
        """
        Returns a callable object that applies the response matrix to the source model.
        """
        return ConvolvedModel(self, source_model)
    






# Example of usage
if __name__=='__main__':

    from sherpa.models.basic import Gauss1D
    import matplotlib.pyplot as plt

    # Create the response matrix
    rsp_wave = np.arange(5000, 7000, 1)

    resp_lambda = np.array([5000.0, 5500.0, 6000.0, 6500.0, 7000.0])
    resp_R = np.array([1695.0, 1750.0, 1978.0, 2227.0, 2484.0])

    # Create the response matrix
    builder = InstRspBuilder(rsp_wave)
    full_response_matrix = builder.build_gaussian_matrix(resp_lambda, resp_R)
    full_response_matrix = builder.build_fixed_fwhm_matrix(10.0)

    # # Save and load the response matrix
    # builder.save_to_fits('response_matrix.fits')
    # loader = InstRspLoader('response_matrix.fits')

    # Define the unfolded model and the energy grid
    gauss = Gauss1D('gauss')
    gauss.ampl = 1.0
    gauss.pos = 6000.0
    gauss.fwhm = 5
    wave = np.arange(5950, 6050, 1)

    # Crop the response matrix to match the wavelength grid
    rsp_matrix = builder.crop_matrix(wave)

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