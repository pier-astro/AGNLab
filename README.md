# AGNFitLab

**AGNFitLab** is an extension of the [`AGNFantasy`](https://fantasy-agn.readthedocs.io/en/latest/) package, designed to provide enhanced tools for modeling AGN spectra using the [`sherpa`](https://parameter-sherpa.readthedocs.io/en/latest/gettingstarted/installation.html) fitting environment in Python.

It wraps and extends the original AGNFantasy package, offering additional modeling features and improved flexibility, especially useful for handling instrument responses and alternate line profiles like Voigt.

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

### 📦 Model Setup Example
1. Clone or download the repository.
2. Place it in a directory visible to Python (`PYTHONPATH`) or install it locally via:

```bash
pip install -e /path/to/AGNFitLab
```

3. Then import the models in your code:

```python
import myfantasy.models as mymodels
from myfantasy import FantasyModels
from myfantasy.models import create_voigt_fixed_model

# Set up folder structure
path_to_folder = 'fit/'
FantasyModels.create_input_folder(path_to_folder=path_to_folder)
mymodels.set_path(path_to_folder)

# Default Fantasy model
narrow = FantasyModels.create_fixed_model(
    ['hydrogen.csv'], name='Narrow',
    fwhm=500, min_fwhm=0., max_fwhm=800.,
    offset=0, min_offset=-300, max_offset=300,
    amplitude=1., min_amplitude=0., max_amplitude=100)

# Improved Fantasy model
narrow_new = mymodels.create_fixed_model(
    ['hydrogen.csv'], name='Narrow',
    fwhm=500, min_fwhm=0., max_fwhm=800.,
    offset=0, min_offset=-300, max_offset=300,
    amplitude=1., min_amplitude=0., max_amplitude=100,
    profile='gauss')

# Voigt profile model
narrow_voigt = create_voigt_fixed_model(
    ['hydrogen.csv'], name='Narrow',
    fwhm_g=100, min_fwhm_g=100, max_fwhm_g=3000,
    fwhm_l=100, min_fwhm_l=100, max_fwhm_l=3000,
    offset=0, min_offset=-3000, max_offset=3000,
    amplitude=10, min_amplitude=0, max_amplitude=100)

model = narrow_new  # Choose any model
```

### 🔧 Instrument Response Integration
```python
import myfantasy.instrument as inst

# Load instrument response (e.g., MUSE)
rsploader = inst.InstRspLoader(inst='MUSE')
rsp_matrix = rsploader.crop_matrix(wave=wav)

# Apply spectral response to the model
rsp = inst.SpectralRsp(rsp_matrix)
fitmodel = rsp(model)
```
