# AGNFitLab

**AGNFitLab** is an extension of the [`AGNFantasy`](https://fantasy-agn.readthedocs.io/en/latest/) package, designed to provide enhanced tools for modeling AGN spectra using the [`sherpa`](https://sherpa.readthedocs.io/en/4.17.0/install.html) fitting environment in Python.

It wraps and extends the original AGNFantasy package, offering additional modeling features and improved flexibility, especially useful for handling instrument responses and alternate line profiles like Lorentzian and Voigt.

---

## ⚙️ Requirements

Before using AGNFitLab, make sure you have the following packages installed:

- [`sherpa`](https://parameter-sherpa.readthedocs.io/en/latest/gettingstarted/installation.html)
- [`AGNFantasy`](https://fantasy-agn.readthedocs.io/en/latest/install.html)

> ⚠️ **Important Note for macOS/Apple Silicon users**  
> AGNFantasy has strict and sometimes incompatible requirements on certain platforms (e.g. Apple Silicon).  
> However, the PyPI distribution only enforces dependencies from `sherpa`, and once `sherpa` is installed, most AGNFantasy functionality works fine.  
> AGNFitLab is designed to minimize dependence on AGNFantasy internals, so you may safely bypass some of the strict requirements.

---
## 🚀 Quickstart

### Install the modules
Clone or download the repository.
Place it in a directory visible to Python (`PYTHONPATH``) or install it locally via:
```bash
pip install -e /path/to/AGNFitLab
```

### 📦 Model Setup Example
Import the models in your code and use them as standard `sherpa` models:
```python
import fantasy_agn.models as FantasyModels
import agnfitlab.models as AGNFitLabModels

# Set up folder structure
path_to_folder = 'testfit/'
FantasyModels.create_input_folder(path_to_folder=path_to_folder)
AGNFitLabModels.set_path(path_to_folder)

# Default Fantasy model
narrow = FantasyModels.create_fixed_model(
    ['hydrogen.csv'], name='Narrow',
    fwhm=500, min_fwhm=0., max_fwhm=800.,
    offset=0, min_offset=-300, max_offset=300,
    amplitude=1., min_amplitude=0., max_amplitude=100)

# Improved Fantasy model
narrow_new = AGNFitLabModels.create_fixed_model(
    ['hydrogen.csv'], name='Narrow',
    fwhm=500, min_fwhm=0., max_fwhm=800.,
    offset=0, min_offset=-300, max_offset=300,
    amplitude=1., min_amplitude=0., max_amplitude=100,
    profile='gauss')

# Voigt profile model
narrow_voigt = AGNFitLabModels.create_voigt_fixed_model(
    ['hydrogen.csv'], name='Narrow',
    fwhm_g=100, min_fwhm_g=100, max_fwhm_g=3000,
    fwhm_l=100, min_fwhm_l=100, max_fwhm_l=3000,
    offset=0, min_offset=-3000, max_offset=3000,
    amplitude=10, min_amplitude=0, max_amplitude=100)

model = narrow_new  # Choose any model or a combination of them
```

### 📦 Instrument Response Integration

```python
import agnfitlab.instrument as inst

# Load instrument response (e.g., MUSE)
rsploader = inst.InstRspLoader(inst='MUSE')

# Assume wav is your wavelength array in the instrument frame
rsp_matrix = rsploader.crop_matrix(wave=wav)

# Apply spectral response to the model
rsp = inst.SpectralRsp(rsp_matrix)
fitmodel = rsp(model)
```

### 🔧 Build Instrument Response and Add to the Local Archive
The instrumental response must be a square matrix that has on the x-axis the 'true' energy and on the y-axis the energy detected by the instrument. It acts as a redistribution matrix, redistributing the model flux according to the instrument's response.

In this implementation, both the step size of the 'true' energy and the instrumental energy must be the same and must match the energy grid of the instrument.

The instrument module has implemented a response builder that allows you to easily construct the response and save it in the local archive of instrumental responses.

The `InstRspBuilder.build_gaussian_matrix()` takes as arguments an array of wavelengths and an array of spectral resolutions $R = \dfrac{\lambda}{\Delta \lambda}$. Based on these, it interpolates along the wavelength axis and builds the response assuming the Line Response Function is Gaussian.

```python
import agnfitlab.instrument as inst

# Define resp_lambda and resp_R as 1D arrays
# e.g. resp_lambda = np.array([5000, 5500, 6000])
#      resp_R = np.array([1600, 1750, 2000])
# Define the instrument wavelength grid as inst_wav

# Create the response matrix
builder = inst.InstRspBuilder(inst_wav)
full_response_matrix = builder.build_gaussian_matrix(resp_lambda, resp_R)

# Save and load the response matrix
builder.save_and_register('INSTRSP.fits', instrument_name='INST', clobber=True)
```

Alternatively, the user can build the response matrix manually as a 2D array and load it:
```python
import agnfitlab.instrument as inst

# Define the instrument wavelength grid as inst_wav and the matrix as rsp_matrix

# Create the response matrix
builder = inst.InstRspBuilder(inst_wav)
full_response_matrix = builder.load_matrix_from_array(rsp_matrix)

# Save and load the response matrix
builder.save_and_register('INSTRSP.fits', instrument_name='INST', clobber=True)
```
