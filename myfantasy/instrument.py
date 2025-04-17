import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse import csr_matrix, issparse
from scipy.special import erf
from astropy.io import fits
from sherpa.models import Model, ArithmeticModel, CompositeModel
import warnings
import os


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
    
    def build_gaussian_matrix(self, lambda_R, R_values, interp_kind='linear'):
        """Build the response matrix using a Gaussian response function (sparse version)"""
        self.lambda_R = np.array(lambda_R)
        self.R_values = np.array(R_values)
        _R_interp = interp1d(self.lambda_R, self.R_values, kind=interp_kind, bounds_error=False, fill_value=(self.R_values[0], self.R_values[-1]))
        N = len(self.wave_grid)
        data, row_indices, col_indices = [], [], []
        for i, lambda_real in enumerate(self.wave_grid):
            R = max(_R_interp(lambda_real), 1.0)
            delta_lam = lambda_real / R
            sigma = delta_lam / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
            # Compute Gaussian integral over each bin
            lower_edges = self.wave_grid - self.wstep / 2
            upper_edges = self.wave_grid + self.wstep / 2
            a = (lower_edges - lambda_real) / (sigma * np.sqrt(2))
            b = (upper_edges - lambda_real) / (sigma * np.sqrt(2))
            integrals = 0.5 * (erf(b) - erf(a))
            # Normalize row to account for any truncation
            integrals /= integrals.sum()
            # Store non-zero values and their indices
            non_zero = integrals > 1e-10  # Threshold to ignore very small values
            data.extend(integrals[non_zero])
            row_indices.extend([i] * np.sum(non_zero))
            col_indices.extend(np.where(non_zero)[0])
        # Create sparse matrix
        self.response_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
        return self.response_matrix
    
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

    def crop_matrix(self, new_wave, renormalize=True):
        """Crop the response matrix to match data wavelengths"""
        if not np.all([np.any(np.isclose(dw, self.wave_grid)) for dw in new_wave]):
            raise ValueError("Data wavelengths must be within the wavelength grid")
        # Ensure the data wavelengths have the same step size as the response matrix
        if not np.all(np.isclose(np.diff(new_wave), self.wstep)):
            warnings.warn("Data wavelengths do not have the same step size as the response matrix. It would be better to rebin the response of the instrument.")

        indices = np.where([np.any(np.isclose(w, self.wave_grid)) for w in new_wave])[0]
        cropped = self.response_matrix[np.ix_(indices, indices)]
        
        if renormalize:
            # Ensure flux conservation
            row_sums = cropped.sum(axis=1)
            cropped /= row_sums[:, np.newaxis]
            
        return cropped
    

class InstRspLoader:
    """Handles loading and preparation of response matrices"""
    def __init__(self, filename=None, inst=None):
        """
        Initialize the loader. Either `filename` or `inst` must be provided.
        
        - filename: Path to the FITS file containing the response matrix.
        - inst: Instrument name to load the precomputed response file. Options are 'MUSE' and that's it :)
        """
        if filename is None and inst is None:
            raise ValueError("Either `filename` or `inst` must be provided.")
        
        if inst is not None:
            # Construct the path to the response file based on the instrument name
            base_dir = os.path.dirname(__file__)  # Directory of the current file
            responses_dir = os.path.join(base_dir, "responses")
            if inst == 'MUSE':
                self.filename = os.path.join(responses_dir, "MUSERSP.fits")
            else:
                raise ValueError(f"Unknown instrument: {inst}. Supported instruments are: 'MUSE'.")
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
        """Crop the response matrix to match data wavelengths"""
        if not np.all([np.any(np.isclose(dw, self.wave)) for dw in wave]):
            raise ValueError("Data wavelengths must be within the wavelength grid")
        # Ensure the data wavelengths have the same step size as the response matrix
        if not np.all(np.isclose(np.diff(wave), np.diff(self.wave)[0])):
            warnings.warn("Data wavelengths do not have the same step size as the response matrix. It would be better to rebin the response of the instrument.")
        
        indices = np.where([np.any(np.isclose(w, self.wave)) for w in wave])[0]
        cropped = self.full_matrix[np.ix_(indices, indices)]
        
        if renormalize:
            # Ensure flux conservation
            row_sums = cropped.sum(axis=1).A1  # Convert sparse matrix row sums to a 1D array
            cropped = cropped.multiply(1 / row_sums[:, np.newaxis])
            
        return cropped




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

    # # Save and load the response matrix
    # builder.save_to_fits('response_matrix.fits')
    # loader = InstRspLoader('response_matrix.fits')

    # Define the unfolded model and the energy grid
    gauss = Gauss1D('gauss')
    gauss.ampl = 1.0
    gauss.pos = 6000.0
    gauss.fwhm = 5
    wave = np.arange(5800, 6200, 1)

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
    plt.legend()
    plt.show()