import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import string
import yaml

import astropy.units as u
import astropy.constants as const
from astropy.modeling.models import BlackBody

from sherpa.models import model
from sherpa.models.parameter import Parameter, tinyval
from sherpa.models import ArithmeticModel, modelCacher1d, CompositeModel, ArithmeticFunctionModel, RegriddableModel1D
from sherpa.models.basic import clean_kwargs1d
from sherpa.utils.guess import get_position, guess_amplitude, guess_fwhm, param_apply_limits, _guess_ampl_scale
from sherpa.models import _modelfcts as _basic_modelfuncs
from sherpa.utils.numeric_types import SherpaFloat

from sherpa.astro.models import _modelfcts as _astro_modelfuncs # Sherpa implementation uses a lorentzian function not normalized to the peak

c = const.c.to(u.km/u.s).value # Speed of light in km/s
script_dir = os.path.dirname(__file__) # get the directory of the current script
input_path = os.path.join(script_dir, "input")

def get_model_free_params(model):
    thawed_pars = model.get_thawed_pars()
    cp_names = []
    parkeys = []
    parvals = []
    for p in thawed_pars:
        comp = p.modelname
        name = p.name
        cp_name = f'{comp}.{name}'
        if cp_name in cp_names:
            i = 2
            cp_name = f'{comp}_{i}.{name}'
            while cp_name in cp_names:
                i += 1
                cp_name = f'{comp}_{i}.{name}'
        cp_names.append(cp_name)
        parkeys.append(cp_name)
        parvals.append(p.val)
    pars = dict(zip(parkeys, parvals))
    return pars

def _convert_np(obj):
    if isinstance(obj, dict):
        return {k: _convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.item()
    else:
        return obj
def save_params(model, filename):
    pars_dict = get_model_free_params(model)
    with open(filename, 'w') as f:
        yaml.dump(_convert_np(pars_dict), f, sort_keys=False)

def read_params(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")
    with open(filename, 'r') as f:
        pars_dict = yaml.safe_load(f)
    return pars_dict

def load_params(model, filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")
    params_dict = read_params(filename)
    # Get an array from all the values
    values = np.array(list(params_dict.values()))
    model.thawedpars = values



def init_lines_csv(wmin=4000, wmax=7000, dirpath='./lines', overwrite=False):
    """
    The init_lines function initializes the lines by reading the csv files from the input folder and filtering them based on the wavelength range.
    The function takes three arguments:
        1) wmin - the minimum wavelength value to filter the lines, default is 4000
        2) wmax - the maximum wavelength value to filter the lines, default is 7000
        3) dirpath - The path where you want to save the filtered lines. Default is './lines'
    
    :param wmin=4000: Used to Set the minimum wavelength value for filtering the lines.
    :param wmax=7000: Used to Set the maximum wavelength value for filtering the lines.
    :param dirpath='./lines': Used to Specify the directory where the filtered lines will be saved.
    :param overwrite=False: Used to Overwrite the existing files in the directory.
    :return: The path to the directory where the filtered lines are saved.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        is_created = True
        print(f"Directory {dirpath} created.")
    else:
        # print(f"Directory {dirpath} already exists.")
        is_created = False
    global csv_lines_path
    csv_lines_path = dirpath
    if overwrite or is_created:
        for files in glob.glob(input_path + "/*.csv"):
            df = pd.read_csv(files)
            try:
                df = df[df.position > wmin]
                df = df[df.position < wmax]
            except:
                df = df[df.wav > wmin]
                df = df[df.wav < wmax]

            name = os.path.join(dirpath, Path(files).name)
            df.to_csv(name, index=False)


def gaussian(pars, x):
    """
    Gaussian function for 1D data.
    
    :param pars: tuple of parameters (fwhm, center, amplitude)
    :param x: array of x values
    :return: array of y values
    """
    fwhm, center, ampl = pars
    return ampl * np.exp(-0.5 * ((x - center) / (fwhm / 2.35482)) ** 2)

class GaussianLine(model.ArithmeticModel):
    def __init__(self, name='gauss',
                 pos=5000,
                 amp=5, min_amp=-500, max_amp=500,
                 fwhm=1000, min_fwhm=5, max_fwhm=10000,
                 offset=0, min_offset=-3000, max_offset=3000):
        
        self.amp = model.Parameter(modelname=name, name="amp", val=amp, min=min_amp, max=max_amp)
        self.pos = model.Parameter(modelname=name, name="pos", val=pos, min=0, frozen=True, units="Å")
        self.offs_kms = model.Parameter(modelname=name, name="offs_kms", val=offset, min=min_offset, max=max_offset, units="km/s")
        self.fwhm = model.Parameter(modelname=name, name="fwhm", val=fwhm, min=min_fwhm, max=max_fwhm, hard_min=tinyval, units="km/s")
        p = (self.amp, self.pos, self.offs_kms, self.fwhm)
        super().__init__(name, p)

    # def get_center(self):
    #     center = self.pos.val + self.offs_kms.val/c * self.pos.val
    #     return (center, )

    # def set_center(self, center, *args, **kwargs):
    #     offs_kms = (center - self.pos.val) / self.pos.val * c
    #     self.offs_kms.set(offs_kms)

    # def guess(self, dep, *args, **kwargs):
    #     norm = guess_amplitude(dep, *args)
    #     param_apply_limits(norm, self.ampl, **kwargs)
    #     pos = get_position(dep, *args)
    #     offs_kms = (pos - self.pos.val) / self.pos.val * c
    #     param_apply_limits(offs_kms, self.offs_kms, **kwargs)
    #     fwhm = guess_fwhm(dep, *args)
    #     fwhm_kms = fwhm / pos * c
    #     param_apply_limits(fwhm_kms, self.fwhm, **kwargs)

    # @modelCacher1d
    def calc(self, p, x, *args, **kwargs):
        kwargs = clean_kwargs1d(self, kwargs)
        ampl = p[0]
        pos = p[1]
        offs_kms = p[2]
        fwhm_kms = p[3]
        center = pos + offs_kms/c * pos
        fwhm = fwhm_kms/c * center
        p = (fwhm, center, ampl)
        return gaussian(pars=p, x=x)
    
class GaussEmLine(GaussianLine):
    def __init__(self, name='em_gauss',
                 pos=5000,
                 amp=5, min_amp=0, max_amp=500,
                 fwhm=1000, min_fwhm=5, max_fwhm=10000,
                 offset=0, min_offset=-3000, max_offset=3000):

        super().__init__(name=name, pos=pos, amp=amp, min_amp=min_amp, max_amp=max_amp,
                         fwhm=fwhm, min_fwhm=min_fwhm, max_fwhm=max_fwhm,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)
        self.amp.hard_min = 0 # Set the hard minimum for the amplitude to 0, so it cannot be an absorption line

class GaussAbsLine(GaussianLine): # Inherit everything from GaussianLine and just change the hard_maximum for the amplitude to 0
    def __init__(self, name='abs_gauss',
                 pos=5000,
                 amp=-5, min_amp=-500, max_amp=0,
                 fwhm=1000, min_fwhm=5, max_fwhm=10000,
                 offset=0, min_offset=-3000, max_offset=3000):

        super().__init__(name=name, pos=pos, amp=amp, min_amp=min_amp, max_amp=max_amp,
                         fwhm=fwhm, min_fwhm=min_fwhm, max_fwhm=max_fwhm,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)
        self.amp.hard_max = 0 # Set the hard maximum for the amplitude to 0, so it cannot be an emission line





# NOTE: The lorentzian() implementation of the Lorentzian profile uses the peak flux as a parameter, while the Sherpa implementation uses the integrated flux.
def lorentzian(pars, x):
    fwhm, center, ampl = pars
    return ampl * ((fwhm / 2.0) ** 2) / ((x - center) ** 2 + (fwhm / 2.0) ** 2)

class LorentzianLine(model.ArithmeticModel):
    def __init__(self, name='lorentz',
                 pos=5000,
                 amp=5, min_amp=-500, max_amp=500,
                 fwhm=1000, min_fwhm=5, max_fwhm=10000,
                 offset=0, min_offset=-3000, max_offset=3000):
        
        self.amp = model.Parameter(modelname=name, name="amp", val=amp, min=min_amp, max=max_amp)
        self.pos = model.Parameter(modelname=name, name="pos", val=pos, min=0, frozen=True, units="Å")
        self.offs_kms = model.Parameter(modelname=name, name="offs_kms", val=offset, min=min_offset, max=max_offset, units="km/s")
        self.fwhm = model.Parameter(modelname=name, name="fwhm", val=fwhm, min=min_fwhm, max=max_fwhm, hard_min=tinyval, units="km/s")
        p = (self.amp, self.pos, self.offs_kms, self.fwhm)
        super().__init__(name, p)

    # def get_center(self):
    #     center = self.pos.val + self.offs_kms.val/c * self.pos.val
    #     return (center, )

    # def set_center(self, center, *args, **kwargs):
    #     offs_kms = (center - self.pos.val) / self.pos.val * c
    #     self.offs_kms.set(offs_kms)

    # def guess(self, dep, *args, **kwargs):
    #     pos = get_position(dep, *args)
    #     offs_kms = (pos - self.pos.val) / self.pos.val * c
    #     param_apply_limits(offs_kms, self.offs_kms, **kwargs)
    #     fwhm = guess_fwhm(dep, *args)
    #     fwhm_kms = fwhm / pos * c
    #     param_apply_limits(fwhm_kms, self.fwhm, **kwargs)

    #     norm = guess_amplitude(dep, *args)
    #     if fwhm != 10:
    #         aprime = norm['val'] * self.fwhm.val * np.pi / 2.
    #         ampl = {'val': aprime,
    #                 'min': aprime / _guess_ampl_scale,
    #                 'max': aprime * _guess_ampl_scale}
    #         param_apply_limits(ampl, self.ampl, **kwargs)
    #     else:
    #         param_apply_limits(norm, self.ampl, **kwargs)

    # ## LORENTZIAN WITH INTEGRATED FLUX AS PARAMETER
    # @modelCacher1d
    # def calc(self, p, *args, **kwargs):
    #     kwargs = clean_kwargs1d(self, kwargs)
    #     ampl = p[0]
    #     pos = p[1]
    #     offs_kms = p[2]
    #     fwhm_kms = p[3]
    #     center = pos + offs_kms/c * pos
    #     fwhm = fwhm_kms/c * center
    #     p = (fwhm, center, ampl)
    #     return _astro_modelfuncs.lorentz1d(p, *args, **kwargs) # Sherpa Implementation uses the intergated flux as parameter

    ## LORENTZIAN WITH PEAK FLUX AS PARAMETER
    def calc(self, p, x, *args, **kwargs):
        ampl = p[0]
        pos = p[1]
        offs_kms = p[2]
        fwhm_kms = p[3]
        center = pos + offs_kms/c * pos
        fwhm = fwhm_kms/c * center
        p = (fwhm, center, ampl)
        return lorentzian(pars=p, x=x)

class LorentzEmLine(LorentzianLine):
    def __init__(self, name='em_lorentz',
                 pos=5000,
                 amp=5, min_amp=0, max_amp=500,
                 fwhm=1000, min_fwhm=5, max_fwhm=10000,
                 offset=0, min_offset=-3000, max_offset=3000):

        super().__init__(name=name, pos=pos, amp=amp, min_amp=min_amp, max_amp=max_amp,
                         fwhm=fwhm, min_fwhm=min_fwhm, max_fwhm=max_fwhm,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)
        self.amp.hard_min = 0 # Set the hard minimum for the amplitude to 0, so it cannot be an absorption line

class LorentzAbsLine(LorentzianLine): 
    def __init__(self, name='abs_lorentz',
                 pos=5000,
                 amp=-5, min_amp=-500, max_amp=0,
                 fwhm=1000, min_fwhm=5, max_fwhm=10000,
                 offset=0, min_offset=-3000, max_offset=3000):

        super().__init__(name=name, pos=pos, amp=amp, min_amp=min_amp, max_amp=max_amp,
                         fwhm=fwhm, min_fwhm=min_fwhm, max_fwhm=max_fwhm,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)
        self.amp.hard_max = 0 # Set the hard maximum for the amplitude to 0, so it cannot be an emission line





# NOTE: As for the Loretzian, the following voigt() implementation of the Lorentzian profile uses the peak flux as a parameter, while the Sherpa implementation uses the integrated flux.
def voigt(pars, x): # From astropy Voigt profile definition
    """Efficient Voigt profile using Humlicek2 rational approximation."""

    fwhm_G, fwhm_L, x_0, amplitude_L = pars

    sqrt_ln2 = np.sqrt(np.log(2))
    sqrt_ln2pi = np.sqrt(np.log(2) * np.pi)
    z = np.atleast_1d(2 * (x - x_0) + 1j * fwhm_L) * sqrt_ln2 / fwhm_G

    # Humlicek region I rational approximation for n=16, delta=1.35
    AA = np.array([
        +46236.3358828121,   -147726.58393079657j,
        -206562.80451354137,  281369.1590631087j,
        +183092.74968253175, -184787.96830696272j,
        -66155.39578477248,   57778.05827983565j,
        +11682.770904216826, -9442.402767960672j,
        -1052.8438624933142,  814.0996198624186j,
        +45.94499030751872,  -34.59751573708725j,
        -0.7616559377907136,  0.5641895835476449j,
    ])
    bb = np.array([
        +7918.06640624997,
        -126689.0625,
        +295607.8125,
        -236486.25,
        +84459.375,
        -15015.0,
        +1365.0,
        -60.0,
        +1.0,
    ])

    def hum2zpf16c(z, s=10.0):
        sqrt_piinv = 1.0 / np.sqrt(np.pi)
        zz = z * z
        w = 1j * (z * (zz * sqrt_piinv - 1.410474)) / (0.75 + zz * (zz - 3.0))
        mask = abs(z.real) + z.imag < s
        if np.any(mask):
            Z = z[mask] + 1.35j
            ZZ = Z * Z
            numer = (((((((((((((((AA[15]*Z + AA[14])*Z + AA[13])*Z + AA[12])*Z + AA[11])*Z +
                               AA[10])*Z + AA[9])*Z + AA[8])*Z + AA[7])*Z + AA[6])*Z +
                          AA[5])*Z + AA[4])*Z+AA[3])*Z + AA[2])*Z + AA[1])*Z + AA[0])
            denom = (((((((ZZ + bb[7])*ZZ + bb[6])*ZZ + bb[5])*ZZ+bb[4])*ZZ + bb[3])*ZZ +
                      bb[2])*ZZ + bb[1])*ZZ + bb[0]
            w[mask] = numer / denom
        return w

    w = hum2zpf16c(z)
    return w.real * sqrt_ln2pi / fwhm_G * fwhm_L * amplitude_L

class VoigtLine(model.ArithmeticModel):
    def __init__(self, name='voigt',
                 pos=5000,
                 amp=5, min_amp=-500, max_amp=500,
                 fwhm_g=1000, min_fwhm_g=5, max_fwhm_g=10000,
                 fwhm_l=1000, min_fwhm_l=5, max_fwhm_l=10000,
                 offset=0, min_offset=-3000, max_offset=3000):
        
        self.amp = model.Parameter(modelname=name, name="amp", val=amp, min=min_amp, max=max_amp)
        self.pos = model.Parameter(modelname=name, name="pos", val=pos, min=0, frozen=True, units="Å")
        self.offs_kms = model.Parameter(modelname=name, name="offs_kms", val=offset, min=min_offset, max=max_offset, units="km/s")
        self.fwhm_g = model.Parameter(modelname=name, name="fwhm_g", val=fwhm_g, min=min_fwhm_g, max=max_fwhm_g, hard_min=tinyval, units="km/s")
        self.fwhm_l = model.Parameter(modelname=name, name="fwhm_l", val=fwhm_l, min=min_fwhm_l, max=max_fwhm_l, hard_min=tinyval, units="km/s")
        p = (self.amp, self.pos, self.offs_kms, self.fwhm_g, self.fwhm_l)
        super().__init__(name, p)

    def get_center(self):
        center = self.pos.val + self.offs_kms.val/c * self.pos.val
        return (center, )
    
    def set_center(self, center, *args, **kwargs):
        offs_kms = (center - self.pos.val) / self.pos.val * c
        self.offs_kms.set(offs_kms)

    def guess(self, dep, *args, **kwargs):
        pos = get_position(dep, *args)
        offs_kms = (pos - self.pos.val) / self.pos.val * c
        param_apply_limits(offs_kms, self.offs_kms, **kwargs)
        fwhm = guess_fwhm(dep, *args)
        fwhm_kms = fwhm / pos * c
        param_apply_limits(fwhm_kms, self.fwhm_g, **kwargs)
        param_apply_limits(fwhm_kms, self.fwhm_l, **kwargs)

        norm = guess_amplitude(dep, *args)
        aprime = norm['val'] * fwhm['val'] * np.pi / 2.
        ampl = {'val': aprime,
                'min': aprime / _guess_ampl_scale,
                'max': aprime * _guess_ampl_scale}
        param_apply_limits(ampl, self.ampl, **kwargs)
    
    # ## VOIGT WITH INTEGRATED FLUX AS PARAMETER
    # @modelCacher1d
    # def calc(self, p, *args, **kwargs):
    #     kwargs = clean_kwargs1d(self, kwargs)
    #     ampl = p[0]
    #     pos = p[1]
    #     offs_kms = p[2]
    #     fwhm_g_kms = p[3]
    #     fwhm_l_kms = p[4]
    #     center = pos + offs_kms/c * pos
    #     fwhm_g = fwhm_g_kms/c * center
    #     fwhm_l = fwhm_l_kms/c * center
    #     p = (fwhm_g, fwhm_l, center, ampl)
    #     return _astro_modelfuncs.wofz(p, *args, **kwargs)

    ## VOIGT WITH PEAK FLUX AS PARAMETER
    def calc(self, p, x, *args, **kwargs):
        ampl = p[0]
        pos = p[1]
        offs_kms = p[2]
        fwhm_g_kms = p[3]
        fwhm_l_kms = p[4]
        center = pos + offs_kms/c * pos
        fwhm_g = fwhm_g_kms/c * center
        fwhm_l = fwhm_l_kms/c * center
        p = (fwhm_g, fwhm_l, center, ampl)
        return voigt(pars=p, x=x)

class VoigtEmLine(VoigtLine):
    def __init__(self, name='em_voigt',
                 pos=5000,
                 amp=5, min_amp=0, max_amp=500,
                 fwhm_g=1000, min_fwhm_g=5, max_fwhm_g=10000,
                 fwhm_l=1000, min_fwhm_l=10, max_fwhm_l=10000,
                 offset=0, min_offset=-3000, max_offset=3000):

        super().__init__(name=name, pos=pos, amp=amp, min_amp=min_amp, max_amp=max_amp,
                         fwhm_g=fwhm_g, min_fwhm_g=min_fwhm_g, max_fwhm_g=max_fwhm_g,
                         fwhm_l=fwhm_l, min_fwhm_l=min_fwhm_l, max_fwhm_l=max_fwhm_l,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)
        self.amp.hard_min = 0 # Set the hard minimum for the amplitude to 0, so it cannot be an absorption line

class VoigtAbsLine(VoigtLine):
    def __init__(self, name='abs_voigt',
                 pos=5000,
                 amp=-5, min_amp=-500, max_amp=0,
                 fwhm_g=1000, min_fwhm_g=5, max_fwhm_g=10000,
                 fwhm_l=1000, min_fwhm_l=10, max_fwhm_l=10000,
                 offset=0, min_offset=-3000, max_offset=3000):

        super().__init__(name=name, pos=pos, amp=amp, min_amp=min_amp, max_amp=max_amp,
                         fwhm_g=fwhm_g, min_fwhm_g=min_fwhm_g, max_fwhm_g=max_fwhm_g,
                         fwhm_l=fwhm_l, min_fwhm_l=min_fwhm_l, max_fwhm_l=max_fwhm_l,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)
        self.amp.hard_max = 0 # Set the hard maximum for the amplitude to 0, so it cannot be an emission line


def _make_unique(names):
    """
    Make a list of unique names by appending suffixes to duplicates, starting from '_b'.
    """
    import string
    counts = {}
    result = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            result.append(name)
        else:
            counts[name] += 1
            suffix = '_' + string.ascii_lowercase[counts[name]] # Start from '_b' for the first duplicate (counts[name] == 1)
            result.append(name + suffix)
    return result

class _TiedLinesBase(model.ArithmeticModel):
    def __init__(self, files, name, amplitude, fwhm, offset, min_offset, max_offset, min_amplitude, max_amplitude, min_fwhm, max_fwhm):
        if len(files) > 0:
            F = [pd.read_csv(os.path.join(csv_lines_path, file)) for file in files]
            df = pd.concat(F)
            df.reset_index(drop=True, inplace=True)
        else:
            raise ValueError("List of csv files should be given to create model")
        df['safe_line'] = df['line'].str.replace(r'[\[\]<>]', '', regex=True).str.replace(' ', '_')
        df['name'] = df.safe_line + '_' + df.position.round(0).astype(int).astype(str)
        df['name'] = _make_unique(df['name'].tolist())
        uniq = df.name.tolist()
        pars = []
        for i in range(len(uniq)):
            pars.append(
                Parameter(name, "amp_" + uniq[i], amplitude, min=min_amplitude, max=max_amplitude, frozen=False)
            )
        for p in pars:
            setattr(self, p.name, p)
        self.offs_kms = model.Parameter(
            name, "offs_kms", offset, min=min_offset, max=max_offset, units='km/s'
        )
        self.fwhm = model.Parameter(
            name, "fwhm", fwhm, min=min_fwhm, max=max_fwhm, units='km/s'
        )
        pars.append(self.offs_kms)
        pars.append(self.fwhm)
        self.dft = df
        super().__init__(name, pars)

    def _get_positions(self):
        return self.dft.position.to_numpy()

class TiedGaussLines(_TiedLinesBase):
    def __init__(self, files, name='', amplitude=2, min_amplitude=0, max_amplitude=600, fwhm=3000, min_fwhm=100, max_fwhm=7000, offset=0, min_offset=-300, max_offset=300):
        super().__init__(files, name, amplitude, fwhm, offset, min_offset, max_offset, min_amplitude, max_amplitude, min_fwhm, max_fwhm)

    def calc(self, pars, x, *args, **kwargs):
        f = 0
        dft = self.dft
        fwhm = pars[-1]
        offs_kms = pars[-2]
        pos = self._get_positions()
        for i in range(len(pos)):
            offset = pos[i] * offs_kms / c
            center = pos[i] + offset
            ampl = pars[i]
            fwhm_func = fwhm / c * center
            p = (fwhm_func, center, ampl)
            # f += _basic_modelfuncs.gauss1d(p, x)
            f += gaussian(p, x)
        return f

class TiedLorentzLines(_TiedLinesBase):
    def __init__(self, files, name='', amplitude=2, min_amplitude=0, max_amplitude=600, fwhm=3000, min_fwhm=100, max_fwhm=7000, offset=0, min_offset=-300, max_offset=300):
        super().__init__(files, name, amplitude, fwhm, offset, min_offset, max_offset, min_amplitude, max_amplitude, min_fwhm, max_fwhm)

    def calc(self, pars, x, *args, **kwargs):
        f = 0
        dft = self.dft
        fwhm = pars[-1]
        offs_kms = pars[-2]
        pos = self._get_positions()
        for i in range(len(pos)):
            offset = pos[i] * offs_kms / c
            center = pos[i] + offset
            ampl = pars[i]
            fwhm_func = fwhm / c * center
            p = (fwhm_func, center, ampl)
            # f += _astro_modelfuncs.lorentz1d(p, x)
            f += lorentzian(p, x)
        return f

class TiedVoigtLines(_TiedLinesBase):
    def __init__(self, files, name='',
                 amplitude=2, min_amplitude=0, max_amplitude=600,
                 fwhm_g=3000, min_fwhm_g=100, max_fwhm_g=7000,
                 fwhm_l=3000, min_fwhm_l=100, max_fwhm_l=7000,
                 offset=0, min_offset=-300, max_offset=300):
        # Call base class to set up amplitudes and offset
        super().__init__(files, name, amplitude, fwhm_g, offset, min_offset, max_offset, min_amplitude, max_amplitude, min_fwhm_g, max_fwhm_g)
        
        # Add Lorentzian FWHM parameter
        self.fwhm_l = model.Parameter(name, "fwhm_l", fwhm_l, min=min_fwhm_l, max=max_fwhm_l, units='km/s')
        # Insert Lorentzian FWHM as the last parameter
        self.pars.append(self.fwhm_l)

    def calc(self, pars, x, *args, **kwargs):
        f = 0
        dft = self.dft
        offs_kms = pars[-3]
        fwhm_g = pars[-2]
        fwhm_l = pars[-1]
        pos = self._get_positions()
        for i in range(len(pos)):
            offset = pos[i] * offs_kms / c
            center = pos[i] + offset
            ampl = pars[i]
            fwhm_g_func = fwhm_g / c * center
            fwhm_l_func = fwhm_l / c * center
            p = (fwhm_g_func, fwhm_l_func, center, ampl)
            # f += _astro_modelfuncs.wofz(p, x)
            f += voigt(p, x)
        return f








class _BaseFeII(model.ArithmeticModel):
    def __init__(self, name="feii", csv_path=None, extra_params=None, offset=0, min_offset=-1000, max_offset=1000):
        if csv_path is None:
            csv_path = os.path.join(csv_lines_path, "feII_model.csv")
        dft = pd.read_csv(csv_path)
        uniq = pd.unique(dft.ime)
        pars = []
        for u in uniq:
            pars.append(Parameter(name, f"amp_{u}", 2, min=0, max=1000, frozen=False))
        for p in pars:
            setattr(self, p.name, p)
        self.offs_kms = model.Parameter(name, "offs_kms", offset, min=min_offset, hard_min=min_offset, max=max_offset)
        pars.append(self.offs_kms)
        if extra_params:
            for param in extra_params:
                pars.append(param)
        self.dft = dft
        self.uniq = uniq
        super().__init__(name, pars)

class GaussFeII(_BaseFeII):
    def __init__(self, name="feii", csv_path=None, fwhm=2000, min_fwhm=10, max_fwhm=10000, offset=0, min_offset=-1000, max_offset=1000):
        fwhm_param = model.Parameter(name, "fwhm", fwhm, min=min_fwhm, hard_min=min_fwhm, max=max_fwhm)
        self.fwhm = fwhm_param
        super().__init__(name, csv_path, extra_params=[fwhm_param], offset=offset, min_offset=min_offset, max_offset=max_offset)

    def calc(self, pars, x, *args, **kwargs):
        dft = self.dft
        uniq = self.uniq
        offs_kms = pars[len(uniq)]
        fwhm = pars[len(uniq) + 1]
        f = 0
        for i, u in enumerate(uniq):
            df = dft[dft["ime"] == u]
            rxc = df.wav.to_numpy()
            Int = df.Int.to_numpy()
            par = pars[i]
            for j in range(len(rxc)):
                offset = rxc[j] * offs_kms / c
                center = rxc[j] + offset
                ampl = Int[j]
                fwhm_func = fwhm / c * center
                p = (fwhm_func, center, ampl)
                # f += par * _basic_modelfuncs.gauss1d(p, x)
                f += par * gaussian(p, x)
        return f

class LorentzFeII(_BaseFeII):
    def __init__(self, name="feii_lorentz", csv_path=None, fwhm=2000, min_fwhm=10, max_fwhm=10000, offset=0, min_offset=-1000, max_offset=1000):
        fwhm_param = model.Parameter(name, "fwhm", fwhm, min=min_fwhm, hard_min=min_fwhm, max=max_fwhm)
        self.fwhm = fwhm_param
        super().__init__(name, csv_path, extra_params=[fwhm_param], offset=offset, min_offset=min_offset, max_offset=max_offset)

    def calc(self, pars, x, *args, **kwargs):
        dft = self.dft
        uniq = self.uniq
        offs_kms = pars[len(uniq)]
        fwhm = pars[len(uniq) + 1]
        f = 0
        for i, u in enumerate(uniq):
            df = dft[dft["ime"] == u]
            rxc = df.wav.to_numpy()
            Int = df.Int.to_numpy()
            par = pars[i]
            for j in range(len(rxc)):
                offset = rxc[j] * offs_kms / c
                center = rxc[j] + offset
                ampl = Int[j]
                fwhm_func = fwhm / c * center
                p = (fwhm_func, center, ampl)
                f += par * lorentzian(p, x)
        return f

class VoigtFeII(_BaseFeII):
    def __init__(self, name="feii_voigt", csv_path=None, fwhm_g=2000, min_fwhm_g=10, max_fwhm_g=10000, fwhm_l=2000, min_fwhm_l=10, max_fwhm_l=10000, offset=0, min_offset=-1000, max_offset=1000):
        fwhm_g_param = model.Parameter(name, "fwhm_g", fwhm_g, min=min_fwhm_g, hard_min=min_fwhm_g, max=max_fwhm_g)
        fwhm_l_param = model.Parameter(name, "fwhm_l", fwhm_l, min=min_fwhm_l, hard_min=min_fwhm_l, max=max_fwhm_l)
        self.fwhm_g = fwhm_g_param
        self.fwhm_l = fwhm_l_param
        super().__init__(name, csv_path, extra_params=[fwhm_g_param, fwhm_l_param], offset=offset, min_offset=min_offset, max_offset=max_offset)

    def calc(self, pars, x, *args, **kwargs):
        dft = self.dft
        uniq = self.uniq
        offs_kms = pars[len(uniq)]
        fwhm_g = pars[len(uniq) + 1]
        fwhm_l = pars[len(uniq) + 2]
        f = 0
        for i, u in enumerate(uniq):
            df = dft[dft["ime"] == u]
            rxc = df.wav.to_numpy()
            Int = df.Int.to_numpy()
            par = pars[i]
            for j in range(len(rxc)):
                offset = rxc[j] * offs_kms / c
                center = rxc[j] + offset
                ampl = Int[j]
                fwhm_g_func = fwhm_g / c * center
                fwhm_l_func = fwhm_l / c * center
                p = (fwhm_g_func, fwhm_l_func, center, ampl)
                f += par * voigt(p, x)
        return f

class GaussUVFeII(GaussFeII):
    def __init__(self, name="uv_feii_gauss", csv_path=None, fwhm=2000, min_fwhm=10, max_fwhm=10000,
                 offset=0, min_offset=-1000, max_offset=1000):
        if csv_path is None:
            csv_path = os.path.join(csv_lines_path, "uvfe.csv")
        super().__init__(name=name, csv_path=csv_path, fwhm=fwhm, min_fwhm=min_fwhm, max_fwhm=max_fwhm,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)

class LorentzUVFeII(LorentzFeII):
    def __init__(self, name="uv_feii_lorentz", csv_path=None, fwhm=2000, min_fwhm=10, max_fwhm=10000,
                 offset=0, min_offset=-1000, max_offset=1000):
        if csv_path is None:
            csv_path = os.path.join(csv_lines_path, "uvfe.csv")
        super().__init__(name=name, csv_path=csv_path, fwhm=fwhm, min_fwhm=min_fwhm, max_fwhm=max_fwhm,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)

class VoigtUVFeII(VoigtFeII):
    def __init__(self, name="uv_feii_voigt", csv_path=None, fwhm_g=2000, min_fwhm_g=10, max_fwhm_g=10000,
                 fwhm_l=2000, min_fwhm_l=10, max_fwhm_l=10000,
                 offset=0, min_offset=-1000, max_offset=1000):
        if csv_path is None:
            csv_path = os.path.join(csv_lines_path, "uvfe.csv")
        super().__init__(name=name, csv_path=csv_path, fwhm_g=fwhm_g, min_fwhm_g=min_fwhm_g, max_fwhm_g=max_fwhm_g,
                         fwhm_l=fwhm_l, min_fwhm_l=min_fwhm_l, max_fwhm_l=max_fwhm_l,
                         offset=offset, min_offset=min_offset, max_offset=max_offset)
        

class BalmerContinuum(model.ArithmeticModel):
    def __init__(self, name="BalCon", A=1, min_A=tinyval, max_A=1e6,
                 T=10000, min_T=5000, max_T=50000,
                 tau=1, min_tau=0.01, max_tau=2):
        self.A = model.Parameter(name, "A", A, min=min_A, hard_min=0, max=max_A)
        self.T = model.Parameter(name, "T", T, min=min_T, max=max_T, frozen=False, units="K")
        self.tau = model.Parameter(name, "tau", tau, min=min_tau, hard_min=tinyval, max=max_tau)
        p = (self.A, self.T, self.tau)
        super().__init__(name, p)

    def calc(self, pars, x, *args, **kwargs):
        lambda_BE = 3646.0  # A
        bbflux = BlackBody(temperature=pars[1] * u.K, scale=10000)
        tau = pars[2] * (x / lambda_BE) ** 3
        bb = bbflux(x * u.AA)
        result = pars[0] * bb * (1.0 - np.exp(-tau))
        ind = np.where(x > lambda_BE, True, False)
        if ind.any():
            result[ind] = 0.0
        return result.value

class BalmerLines(model.ArithmeticModel):
    def __init__(self, name="Bal_lines", balmer_csv=None, ampl=50, min_ampl=1, max_ampl=1e6,
                 offs_kms=1, min_offs_kms=-3000, max_offs_kms=3000,
                 fwhm=3000, min_fwhm=1000, max_fwhm=4000):
        if balmer_csv is None:
            balmer_csv = os.path.join(csv_lines_path, "balmer.csv")
        self.df = pd.read_csv(balmer_csv)
        self.wave = self.df.position.to_numpy()
        self.rat = self.df.int.to_numpy()
        self.ampl = model.Parameter(name, "ampl", ampl, min=min_ampl, max=max_ampl)
        self.offs_kms = model.Parameter(name, "offs_kms", offs_kms, min=min_offs_kms, max=max_offs_kms, units="km/s")
        self.fwhm = model.Parameter(name, "fwhm", fwhm, min=min_fwhm, max=max_fwhm, hard_min=tinyval, units="km/s")
        p = (self.ampl, self.offs_kms, self.fwhm)
        super().__init__(name, p)

    def calc(self, pars, x, *args, **kwargs):
        lambda_BE = 3646.0
        c = 299792.458
        ampl, offs_kms, fwhm = pars
        f = 0
        for i in range(len(self.wave)):
            pos = self.wave[i]
            rat = self.rat[i]
            offset = pos * offs_kms / c
            sigma = (pos + offset) * fwhm / (c * 2.354)
            f1 = ampl
            f2 = - (x - pos - offset) ** 2. / (2 * sigma ** 2.)
            f += rat * f1 * np.exp(f2)
        ind = np.where(x <= lambda_BE, True, False)
        if ind.any():
            f[ind] = 0.0
        return f


class Powerlaw(model.ArithmeticModel):
    def __init__(self, name='powerlaw', w_ref=5500, amp=1., index=-1.7,
                 min_w_ref=5400, max_w_ref=7000,
                 min_amp=0.01, max_amp=10000.,
                 min_index=-3, max_index=0):
        self.w_ref = model.Parameter(name, "w_ref", w_ref, min=min_w_ref, max=max_w_ref, hard_min=tinyval, frozen=False, units="Å")
        self.amp = model.Parameter(name, "amp", amp, min=min_amp, max=max_amp, hard_min=tinyval, units="Å")
        self.index = model.Parameter(name, "index", index, min=min_index, max=max_index, frozen=False)
        p = (self.w_ref, self.amp, self.index)
        super().__init__(name, p)

    def calc(self, p, x, xhi=None, **kwargs):
        arg = x / p[0]
        arg = p[1] * np.power(arg, p[2])
        return arg

class BrokenPowerlaw(model.ArithmeticModel):
    def __init__(self, name='bknpower', w_ref=5500, amp=1., index1=-1.7, index2=0,
                 min_w_ref=4000, max_w_ref=9000,
                 min_amp=0.01, max_amp=10000.,
                 min_index1=-3, max_index1=0,
                 min_index2=-1, max_index2=1):
        self.w_ref = model.Parameter(name, "w_ref", w_ref, min=min_w_ref, max=max_w_ref, hard_min=tinyval, frozen=False, units="Å")
        self.amp = model.Parameter(name, "amp", amp, min=min_amp, max=max_amp, hard_min=tinyval, units="Å")
        self.index1 = model.Parameter(name, "index1", index1, min=min_index1, max=max_index1, frozen=False)
        self.index2 = model.Parameter(name, "index2", index2, min=min_index2, max=max_index2)
        p = (self.w_ref, self.amp, self.index1, self.index2)
        super().__init__(name, p)
    
    def calc(self, p, x, xhi=None, **kwargs):
        arg = x / p[0]
        arg = p[1] * (np.where(arg > 1.0, np.power(arg, p[2] + p[3]), np.power(arg, p[2])))
        return arg


# ### --- To be better implemented later --- ###
# class Host(model.RegriddableModel1D):
#     def __init__(self, name="Host", gal_file="gal.npy", min_a=0, max_a=1e6):
#         gal = np.load(gal_file)
#         pars = []
#         for i in range(gal.shape[0]):
#             pars.append(Parameter(name, f"a{i}", 1, min=min_a, max=max_a))
#         for p in pars:
#             setattr(self, p.name, p)
#         self.gal = gal
#         super().__init__(name, pars)

#     def calc(self, pars, x, *args, **kwargs):
#         f = 0
#         for i in range(self.gal.shape[0]):
#             f += pars[i] * self.gal[i]
#         return f

# class SSP(model.RegriddableModel1D):
#     def __init__(self, name="Host", gal_file="ica.npy", min_a=0, max_a=1e6):
#         gal = np.load(gal_file)
#         pars = []
#         for i in range(gal.shape[0]):
#             pars.append(Parameter(name, f"a{i}", 1, min=min_a, max=max_a))
#         for p in pars:
#             setattr(self, p.name, p)
#         self.gal = gal
#         super().__init__(name, pars)

#     def calc(self, pars, x, *args, **kwargs):
#         f = 0
#         for i in range(self.gal.shape[0]):
#             f += pars[i] * self.gal[i]
#         return f
# ### --- To be better implemented later --- ###