import numpy as np
import pandas as pd

import astropy.units as u
import astropy.constants as const

from sherpa.models import model
from sherpa.models.parameter import Parameter, tinyval
from sherpa.models import ArithmeticModel, modelCacher1d, CompositeModel, ArithmeticFunctionModel, RegriddableModel1D
from sherpa.models.basic import clean_kwargs1d
from sherpa.utils.guess import get_position, guess_amplitude, guess_fwhm, param_apply_limits, _guess_ampl_scale
from sherpa.models import _modelfcts as _basic_modelfuncs

from sherpa.astro.models import _modelfcts as _astro_modelfuncs # Sherpa implementation uses a lorentzian function not normalized to the peak

from fantasy_agn.models import *

c = const.c.to(u.km/u.s).value # Speed of light in km/s

def set_path(path):
    global path1
    path1 = path


class GaussEmLine(RegriddableModel1D):

    def __init__(self, name='gauss1d'):
        self.ampl = model.Parameter(name, "ampl", 10, min=0, hard_min=0, max=10000)
        self.pos = model.Parameter(name, "pos", 4861, min=0, frozen=True, units="angstroms")
        self.offs_kms = model.Parameter(name, "offs_kms", 0, min=-10000, hard_min=-10000, max=10000, units="km/s")
        self.fwhm = model.Parameter(name, "fwhm", 1000, min=0, hard_min=0, max=10000, units="km/s")

        ArithmeticModel.__init__(self, name, (self.ampl, self.pos, self.offs_kms, self.fwhm))

    def get_center(self):
        center = self.pos.val + self.offs_kms.val/c * self.pos.val
        return (center, )

    def set_center(self, center, *args, **kwargs):
        offs_kms = (center - self.pos.val) / self.pos.val * c
        self.offs_kms.set(offs_kms)

    def guess(self, dep, *args, **kwargs):
        norm = guess_amplitude(dep, *args)
        param_apply_limits(norm, self.ampl, **kwargs)
        pos = get_position(dep, *args)
        offs_kms = (pos - self.pos.val) / self.pos.val * c
        param_apply_limits(offs_kms, self.offs_kms, **kwargs)
        fwhm = guess_fwhm(dep, *args)
        fwhm_kms = fwhm / pos * c
        param_apply_limits(fwhm_kms, self.fwhm, **kwargs)

    @modelCacher1d
    def calc(self, p, *args, **kwargs):
        kwargs = clean_kwargs1d(self, kwargs)
        ampl = p[0]
        pos = p[1]
        offs_kms = p[2]
        fwhm_kms = p[3]
        center = pos + offs_kms/c * pos
        fwhm = fwhm_kms/c * center
        p = (fwhm, center, ampl)
        return _basic_modelfuncs.gauss1d(p, *args, **kwargs)




def astropy_lorentzian(pars, x):
    fwhm, center, ampl = pars
    return ampl * ((fwhm / 2.0) ** 2) / ((x - center) ** 2 + (fwhm / 2.0) ** 2)

class LorentzEmLine(RegriddableModel1D):
    
    def __init__(self, name='lorentz1d'):
        self.ampl = model.Parameter(name, "ampl", 10, min=0, hard_min=0, max=10000)
        self.pos = model.Parameter(name, "pos", 4861, min=0, frozen=True, units="angstroms")
        self.offs_kms = model.Parameter(name, "offs_kms", 0, min=-10000, hard_min=-10000, max=10000, units="km/s")
        self.fwhm = model.Parameter(name, "fwhm", 1000, min=0, hard_min=0, max=10000, units="km/s")

        ArithmeticModel.__init__(self, name, (self.ampl, self.pos, self.offs_kms, self.fwhm))

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
        param_apply_limits(fwhm_kms, self.fwhm, **kwargs)

        norm = guess_amplitude(dep, *args)
        if fwhm != 10:
            aprime = norm['val'] * self.fwhm.val * np.pi / 2.
            ampl = {'val': aprime,
                    'min': aprime / _guess_ampl_scale,
                    'max': aprime * _guess_ampl_scale}
            param_apply_limits(ampl, self.ampl, **kwargs)
        else:
            param_apply_limits(norm, self.ampl, **kwargs)

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
    #     return _astro_modelfuncs.lorentz1d(p, *args, **kwargs) # Sherpa Implementation uses the function not normalized to the peak

    def calc(self, p, x, *args, **kwargs):
        ampl = p[0]
        pos = p[1]
        offs_kms = p[2]
        fwhm_kms = p[3]
        center = pos + offs_kms/c * pos
        fwhm = fwhm_kms/c * center
        p = (fwhm, center, ampl)
        return astropy_lorentzian(pars=p, x=x) # Astropy Implementation uses the function normalized to the peak

    
    
def create_line(name='line',
                pos=4861,
                ampl=5, min_ampl=0, max_ampl=500,
                fwhm= 1000, min_fwhm=5, max_fwhm=10000,
                offset=0, min_offset=-3000, max_offset=3000,
                profile='gauss'):
    
    if profile=='gauss':
        line=GaussEmLine(name=name)
    elif profile=='lorentz':
        line=LorentzEmLine(name=name)
    else:
        raise ValueError('Line profile not recognized. Use "gauss" or "lorentz"')
    
    line.pos =pos
    line.ampl=ampl
    line.ampl.min=min_ampl
    line.ampl.max=max_ampl
    line.fwhm=fwhm
    line.fwhm.min=min_fwhm
    line.fwhm.max=max_fwhm
    line.offs_kms=offset
    line.offs_kms.min=min_offset
    line.offs_kms.max=max_offset
    return line



def create_fixed_model(files=[], 
                       name='',     
                       amplitude=2,
                       fwhm=3000,
                       offset=0,
                       min_offset=-3000,
                       max_offset=3000,
                       min_amplitude=0,
                       max_amplitude=600,
                       min_fwhm=100,
                       max_fwhm=7000,
                       profile='gauss'):
    """
    The create_fixed_model function creates a model that is fixed fwhm for all lines.
    The function takes as an argument a list of csv files, which contain the information (name and position) of all lines included in the model.
    It also takes as arguments: name, amplitude, fwhm (km/s), offset (km/s) and min_offset and max_offset (km/s). 
    
    :param files=[]: Used to Pass a list of csv files to the model.
    :param name='': Used to Give the model a name.
    :param amplitude=2: Used to Set the initial value of the amplitude parameter.
    :param fwhm=3000: Used to Set the fwhm of the gaussian profile.
    :param offset=0: Used to Shift the lines to the center of each pixel.
    :param min_offset=-3000: Used to Set the minimum value of the offset.
    :param max_offset=3000: Used to Set the maximum offset of the line from its rest position.
    :param min_amplitude=0: Used to Set the minimum value of the amplitude parameter.
    :param max_amplitude=600: Used to Limit the maximum amplitude of the lines.
    :param min_fwhm=100: Used to Set a lower limit to the fwhm parameter.
    :param max_fwhm=7000: Used to Set the maximum value of the fwhm parameter.
    :param : Used to Set the initial value of the amplitude parameter.
    :return: A fixed_lines class object.
    """

    if len(files) > 0:
        F = []
        for file in files:
            F.append(pd.read_csv(path1 + file))
        df = pd.concat(F)
        df.reset_index(drop=True, inplace=True)
    else:
        print("List of csv files should be given to create model")
        
    class Fixed_Lines(model.RegriddableModel1D):
        def __init__(self, name=name):
            dft = df
            dft['name']=dft.line+'_'+dft.position.round(0).astype(int).astype(str)

            uniq = dft.name.tolist()
            pars = []

            for i in range(len(uniq)):
                pars.append(
                    Parameter(name, "amp_" + uniq[i],
                            amplitude, min=min_amplitude, max=max_amplitude, frozen=False)
                )
            for p in pars:
                setattr(self, p.name, p)
            self.offs_kms = model.Parameter(
                name, "offs_kms", offset, min=min_offset, hard_min=min_offset, max=max_offset, units='km/s'
            )
            self.fwhm = model.Parameter(
                name, "fwhm", fwhm, min=min_fwhm, hard_min=min_fwhm, max=max_fwhm, units='km/s'
            )

            pars.append(self.offs_kms)
            pars.append(self.fwhm)
            self.dft=dft

            model.RegriddableModel1D.__init__(self, name, pars)

        def fixed(self, pars, x):
            f=0
            dft=self.dft
            fwhm=pars[-1]
            offs_kms=pars[-2]
            pos = dft.position.to_numpy()

            if profile=='gauss':
                func = _basic_modelfuncs.gauss1d
            elif profile=='lorentz':
                # func = _astro_modelfuncs.lorentz1d
                func = astropy_lorentzian

            for i in range(len(pos)):
                
                offset = pos[i] * offs_kms / c
                center = pos[i] + offset
                ampl = pars[i]
                fwhm_func = fwhm/c * center
                p = (fwhm_func, center, ampl)

                f += func(p, x)
            return f
        
        def calc(self, pars, x, *args, **kwargs):
            return self.fixed(pars, x)
        
    return Fixed_Lines()





def _feii(pars, x, profile='gauss'):

    dft = pd.read_csv(path1 + "feII_model.csv")

    uniq = pd.unique(dft.ime)
    offs_kms = pars[len(uniq)]
    fwhm = pars[len(uniq) + 1]

    if profile=='gauss':
        func = _basic_modelfuncs.gauss1d
    elif profile=='lorentz':
        # func = _astro_modelfuncs.lorentz1d
        func = astropy_lorentzian
    f = 0
    index = 1  # starting index in pars
    for i in range(len(uniq)):
        df = dft[dft["ime"] == uniq[i]]

        rxc = df.wav.to_numpy()
        Int = df.Int.to_numpy()

        par = pars[i]
        for i in range(len(rxc)):

            offset = rxc[i] * offs_kms / c
            center = rxc[i] + offset
            ampl = Int[i]
            fwhm_func = fwhm / c * center
            p = (fwhm_func, center, ampl)
            f += par*func(p, x)
    return f


class FeII(model.RegriddableModel1D):
    def __init__(self, name="feii", profile='gauss'):
        dft = pd.read_csv(path1 + "feII_model.csv")
        self.profile = profile

        uniq = pd.unique(dft.ime)
        pars = []

        for i in range(len(uniq)):
            pars.append(
                Parameter(name, "amp_" + uniq[i],
                          2, min=0, max=1000, frozen=False)
            )
        for p in pars:
            setattr(self, p.name, p)
        self.offs_kms = model.Parameter(
            name, "offs_kms", 0, min=-1000, hard_min=-3000, max=1000
        )
        self.fwhm = model.Parameter(
            name, "fwhm", 2000, min=tinyval, hard_min=tinyval, max=10000
        )

        pars.append(self.offs_kms)
        pars.append(self.fwhm)

        model.RegriddableModel1D.__init__(self, name, pars)

    def calc(self, pars, x, *args, **kwargs):
        return _feii(pars, x, self.profile)
    


def create_feii_model(name='feii', fwhm=2000, min_fwhm=1000, max_fwhm=8000, offset=0, min_offset=-3000, max_offset=3000, profile='gauss'):
    """
    The create_feii_model function creates a FeII model with the specified parameters.
    
    :param name='feii': Used to Name the component.
    :param fwhm=2000: Used to Set the fwhm of the feii emission line.
    :param min_fwhm=1000: Used to Set the minimum value of the fwhm parameter.
    :param max_fwhm=8000: Used to Set the upper limit of the fwhm range.
    :param offset=0: Used to Set the central wavelength of the feii emission line.
    :param min_offset=-3000: Used to Set the minimum value of the offset parameter.
    :param max_offset=3000: Used to Set the maximum offset velocity of the feii emission line.
    :return: A feii object.
    """
    fe=FeII(name, profile)
    fe.fwhm=fwhm
    fe.fwhm.max=max_fwhm
    fe.fwhm.min=min_fwhm
    fe.offs_kms=offset
    fe.offs_kms.min= min_offset
    fe.offs_kms.max=max_offset
    return fe









def astropy_voigt(pars, x):
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

class VoigtEmLine(RegriddableModel1D):
    def __init__(self, name='voigt1d'):
        self.ampl = model.Parameter(name, "ampl", 10, min=0, hard_min=0, max=10000)
        self.pos = model.Parameter(name, "pos", 4861, min=0, frozen=True, units="angstroms")
        self.offs_kms = model.Parameter(name, "offs_kms", 0, min=-10000, hard_min=-10000, max=10000, units="km/s")
        self.fwhm_g = model.Parameter(name, "fwhm_g", 1000, min=0, hard_min=0, max=10000, units="km/s")
        self.fwhm_l = model.Parameter(name, "fwhm_l", 1000, min=0, hard_min=0, max=10000, units="km/s")

        ArithmeticModel.__init__(self, name, (self.ampl, self.pos, self.offs_kms, self.fwhm_g, self.fwhm_l))

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
        return astropy_voigt(pars=p, x=x)
    

def create_voigt_line(name='voigt',
                pos=4861,
                ampl=5, min_ampl=0, max_ampl=500,
                fwhm_g=1000, min_fwhm_g=1, max_fwhm_g=10000,
                fwhm_l=1000, min_fwhm_l=10, max_fwhm_l=10000,
                offset=0, min_offset=-3000, max_offset=3000):
    line=VoigtEmLine(name=name)
    line.pos =pos
    line.ampl=ampl
    line.ampl.min=min_ampl
    line.ampl.max=max_ampl
    line.fwhm_g=fwhm_g
    line.fwhm_g.min=min_fwhm_g
    line.fwhm_g.max=max_fwhm_g
    line.fwhm_l=fwhm_l
    line.fwhm_l.min=min_fwhm_l
    line.fwhm_l.max=max_fwhm_l
    line.offs_kms=offset
    line.offs_kms.min=min_offset
    line.offs_kms.max=max_offset
    return line


def create_voigt_fixed_model(files=[], 
                       name='',     
                       amplitude=2,
                       fwhm_g=3000,
                       fwhm_l=3000,
                       offset=0,
                       min_offset=-3000,
                       max_offset=3000,
                       min_amplitude=0,
                       max_amplitude=600,
                       min_fwhm_g=1,
                       max_fwhm_g=7000,
                       min_fwhm_l=100,
                       max_fwhm_l=7000):
    """
    The create_fixed_model function creates a model that is fixed fwhm for all lines.
    The function takes as an argument a list of csv files, which contain the information (name and position) of all lines included in the model.
    It also takes as arguments: name, amplitude, fwhm (km/s), offset (km/s) and min_offset and max_offset (km/s). 
    """

    if len(files) > 0:
        F = []
        for file in files:
            F.append(pd.read_csv(path1 + file))
        df = pd.concat(F)
        df.reset_index(drop=True, inplace=True)
    else:
        print("List of csv files should be given to create model")

    class Fixed_Lines(model.RegriddableModel1D):
        def __init__(self, name=name):
            dft = df
            dft['name']=dft.line+'_'+dft.position.round(0).astype(int).astype(str)

            uniq = dft.name.tolist()
            pars = []

            for i in range(len(uniq)):
                pars.append(
                    Parameter(name, "amp_" + uniq[i],
                            amplitude, min=min_amplitude, max=max_amplitude, frozen=False)
                )
            for p in pars:
                setattr(self, p.name, p)

            self.offs_kms = model.Parameter(name, "offs_kms", offset, min=min_offset, hard_min=min_offset, max=max_offset, units='km/s')
            self.fwhm_g = model.Parameter(name, "fwhm_g", fwhm_g, min=min_fwhm_g, hard_min=min_fwhm_g, max=max_fwhm_g, units='km/s')
            self.fwhm_l = model.Parameter(name, "fwhm_l", fwhm_l, min=min_fwhm_l, hard_min=min_fwhm_l, max=max_fwhm_l, units='km/s')

            pars.append(self.offs_kms)
            pars.append(self.fwhm_g)
            pars.append(self.fwhm_l)
            self.dft=dft

            model.RegriddableModel1D.__init__(self, name, pars)

        def fixed(self, pars, x):
            f=0
            dft=self.dft
            fwhm_l=pars[-1]
            fwhm_g=pars[-2]
            offs_kms=pars[-3]
            pos = dft.position.to_numpy()

            # func = _astro_modelfuncs.wofz
            func = astropy_voigt

            for i in range(len(pos)):
                
                offset = pos[i] * offs_kms / c
                center = pos[i] + offset
                ampl = pars[i]
                fwhm_g_func = fwhm_g/c * center
                fwhm_l_func = fwhm_l/c * center
                p = (fwhm_g_func, fwhm_l_func, center, ampl)

                f += func(p, x)
            return f
        
        def calc(self, pars, x, *args, **kwargs):
            return self.fixed(pars, x)
    
    return Fixed_Lines()


def _voigt_feii(pars, x):

    dft = pd.read_csv(path1 + "feII_model.csv")
    uniq = pd.unique(dft.ime)
    offs_kms = pars[len(uniq)]
    fwhm_g = pars[len(uniq) + 1]
    fwhm_l = pars[len(uniq) + 2]

    # func = _astro_modelfuncs.wofz
    func = astropy_voigt

    f = 0
    index = 1  # starting index in pars
    for i in range(len(uniq)):
        df = dft[dft["ime"] == uniq[i]]

        rxc = df.wav.to_numpy()
        Int = df.Int.to_numpy()

        par = pars[i]
        for i in range(len(rxc)):

            offset = rxc[i] * offs_kms / c
            center = rxc[i] + offset
            ampl = Int[i]
            fwhm_g_func = fwhm_g/c * center
            fwhm_l_func = fwhm_l/c * center
            p = (fwhm_g_func, fwhm_l_func, center, ampl)
            f += par*func(p, x)
    return f


class VoigtFeII(model.RegriddableModel1D):
    def __init__(self, name="voigt_feii"):
        dft = pd.read_csv(path1 + "feII_model.csv")
        uniq = pd.unique(dft.ime)
        pars = []
        for i in range(len(uniq)):
            pars.append(
                Parameter(name, "amp_" + uniq[i],
                          2, min=0, max=1000, frozen=False)
            )
        for p in pars:
            setattr(self, p.name, p)
        self.offs_kms = model.Parameter(
            name, "offs_kms", 0, min=-1000, hard_min=-3000, max=1000
        )
        self.fwhm_g = model.Parameter(
            name, "fwhm_g", 2000, min=tinyval, hard_min=tinyval, max=10000
        )
        self.fwhm_l = model.Parameter(
            name, "fwhm_l", 2000, min=tinyval, hard_min=tinyval, max=10000
        )
        pars.append(self.offs_kms)
        pars.append(self.fwhm_g)
        pars.append(self.fwhm_l)

        model.RegriddableModel1D.__init__(self, name, pars)

    def calc(self, pars, x, *args, **kwargs):
        return _voigt_feii(pars, x)


def create_voigt_feii_model(name='voigt_feii',
                            fwhm_g=2000, min_fwhm_g=1000, max_fwhm_g=8000,
                            fwhm_l=2000, min_fwhm_l=1000, max_fwhm_l=8000,
                            offset=0, min_offset=-3000, max_offset=3000):
    fe=VoigtFeII(name)
    fe.fwhm_g=fwhm_g
    fe.fwhm_g.max=max_fwhm_g
    fe.fwhm_g.min=min_fwhm_g
    fe.fwhm_l=fwhm_l
    fe.fwhm_l.max=max_fwhm_l
    fe.fwhm_l.min=min_fwhm_l
    fe.offs_kms=offset
    fe.offs_kms.min= min_offset
    fe.offs_kms.max=max_offset
    return fe