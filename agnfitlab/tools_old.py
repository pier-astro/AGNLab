from astropy.io import fits
from astropy import units as u
from astropy import constants as const
from itertools import product
from numpy.testing._private.utils import KnownFailureException
from sfdmap2 import sfdmap
import numpy as np
from PyAstronomy import pyasl
from copy import deepcopy
import pylab as plt
from spectres import spectres
from scipy import interpolate, linalg
from sherpa.data import Data1D
from sherpa.fit import Fit
from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch
from sherpa.stats import LeastSq, Cash, Chi2, Likelihood, Stat
from sherpa.estmethods import Confidence
import pandas as pd
import glob
import json
import concurrent.futures
from sherpa.models.parameter import Parameter, tinyval
import natsort
from sklearn import preprocessing
from RegscorePy import aic, bic
import multiprocessing.pool
import os
from tqdm.notebook import tqdm
import multiprocess as mp

script_dir = os.path.dirname(__file__)
sfdpath = os.path.join(script_dir, "sfddata")
pathEigen = os.path.join(script_dir, "eigen")
c = const.c.to(u.km/u.s).value # Speed of light in km/s

# import warnings
# warnings.filterwarnings("ignore")

def _wave_convert(lam):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param lam - Wavelength in Angstroms
    :return: conversion factor
    """
    lam = np.asarray(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
    return fact

def _vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms
    """
    return lam_vac/_wave_convert(lam_vac)

def _check_host(hst):
    if np.sum(np.where(hst < 0.0, True, False)) > 100:
        return False
    else:
        return True
    
def _fithost(hs, ags, data):
    A = np.vstack([hs, ags]).T
    W=1/data**2
    Aw = A * np.sqrt(W[:,np.newaxis])
    dataw=data * np.sqrt(W)
    k = np.linalg.lstsq(Aw, dataw, rcond=None)[0]
    return k

def _maskingEigen(x):
    """Preparing mask for eigen spectra host preparation

    Parameters
    ----------
    x : array
        Spectrum wavelength to mask in order to rebin  

    Returns
    -------
    boolean array
    """
    if min(x) > 3450:
        minx = min(x)
    else:
        minx = 3450
    if max(x) < 7000:
        maxx = max(x) - 10
    else:
        maxx = 7200 - 10
    mask = (x > minx) & (x < maxx)
    return mask

def _resam(x, wv, flx):
    """
    Args:
        x ([type]): [description]
        wv ([type]): [description]
        flx ([type]): [description]

    Returns:
        [type]: [description]
    """
    mask = _maskingEigen(x)
    y = spectres(x[mask], wv, flx)
    return y

def _host_mask(wave, host):
    line_mask = np.where( (wave < 4970.) & (wave > 4950.) |
    (wave < 5020.) & (wave > 5000.) | 
    (wave < 6590.) & (wave > 6540.) |
    (wave < 3737.) & (wave > 3717.) |
    (wave < 4872.) & (wave > 4852.) |
    (wave < 4350.) & (wave > 4330.) |
    (wave < 6750.) & (wave > 6680.) |
    (wave < 4111.) & (wave > 4091.), True, False)
    f = interpolate.interp1d(wave[~line_mask],host[~line_mask], bounds_error = False, fill_value = 0)
    return f(wave)

def _custom_mask(wave, host, mask_list):
    line_mask=np.where(wave, False, True)
    for m in mask_list:
        mask = np.where( (wave < m[1]) & (wave > m[0]), True, False)
        line_mask += mask
    f = interpolate.interp1d(wave[~line_mask],host[~line_mask], bounds_error = False, fill_value = 0)
    return f(wave)

def _prepare_host(wave, flux, fluxerr, galaxy=5, agns=10):
    """Preparing qso and galaxy eigen spectra

    Parameters
    ----------
    wave : array
        Spectrum wavelength
    flux : array
        Spectrum flux
    err : array
        Spectrum error flux
    galaxy : int, optional
        number of galaxy eigen spectra, by default 5
    agns : int, optional
        number of qso eigen spectra, by default 10

    Returns
    -------
    data cube
        rebbined spectra and derived host galaxy
    """
    if galaxy > 10 or agns > 50:
        print("Number of galaxy eigenspectra has to be less than 10 and QSO eigenspectra less than 50")
    else:
        glx = fits.open(pathEigen + "gal_eigenspec_Yip2004.fits")
        glx = glx[1].data
        gl_wave = glx["wave"].flatten()
        gl_wave=_vac_to_air(gl_wave)
        gl_flux = glx["pca"].reshape(glx["pca"].shape[1], glx["pca"].shape[2])

        qso = fits.open(pathEigen + "qso_eigenspec_Yip2004_global.fits")
        qso = qso[1].data
        qso_wave = qso["wave"].flatten()
        qso_wave=_vac_to_air(qso_wave)
        qso_flux = qso["pca"].reshape(qso["pca"].shape[1], qso["pca"].shape[2])

        mask = _maskingEigen(wave)
        wave1 = wave[mask]

        flux, fluxerr = spectres(wave1, wave, flux, spec_errs=fluxerr)
        qso_prime = []
        g_prime = []
        for i in range(galaxy):
            gix = _resam(wave, gl_wave, gl_flux[i])
            g_prime.append(gix)

        for i in range(agns):
            yi = _resam(wave, qso_wave, qso_flux[i])
            qso_prime.append(yi)
        return wave1, flux, fluxerr, g_prime, qso_prime

def _find_nearest(array):
    array = np.asarray(array)
    idx = (np.abs(array - 1)).argmin()
    return idx

def _calculate_chi(data, er, func, coef):
    diff = pow(data - func, 2.0)
    test_statistic = (diff / pow(er, 2.0)).sum()
    NDF = len(data) - len(coef)

    return test_statistic / NDF


class Spectrum():
    def __init__(self):
        self._zcorrected = False
        self._dereddened = False
        self._vac_to_air_corrected = False

    def DeRedden(self, ebv=0):
        """
        Function for dereddening  a flux vector  using the parametrization given by
        [Fitzpatrick (1999)](https://iopscience.iop.org/article/10.1086/316293).

        Parameters
        ----------
        ebv : float, optional
            Color excess E(B-V). If not given it will be automatically derived from
             Dust map data from [Schlegel, Finkbeiner and Davis (1998)](http://adsabs.harvard.edu/abs/1998ApJ...500..525S),
             by default 0
        """
        if ebv != 0:
            self.flux = pyasl.unred(self.wave, self.flux, ebv)
        else:
            m = sfdmap.SFDMap(sfdpath)
            self.flux = pyasl.unred(self.wave, self.flux, m.ebv(self.ra, self.dec))
        self._dereddened = True

    def zCorrection(self, redshift=0):
        """
        The zCorrection() function corrects the flux for redshift.
        It takes in a redshift and corrects the wavelength, flux, and error arrays by that given redshift.

        :param self: Used to Reference the class object.
        :param redshift=0: Used to Specify the redshift of the object.
        :return: The wavelength, flux and error arrays for the object at a redshift of z=0.
        """
        if redshift != 0:
            self.z = redshift
        self.wave = self.wave / (1 + self.z)
        self.flux = self.flux * (1 + self.z)
        if self.fluxerr is not None:
            self.fluxerr = self.fluxerr * (1 + self.z)
        self.fwhm = self.fwhm / (1 + self.z)
        self._zcorrected = True

    def rebin(self, factor:int=None, new_wave=None, method='mean'):
        """
        Rebin the spectrum either by an integer scaling factor or to a new wavelength grid.
        Aggregation can be 'mean' or 'median'.
        Pads with zeros outside the original range if new_wave is used.

        Parameters
        ----------
        factor : int, optional
            Integer factor to downsample the spectrum by averaging/median in bins.
        new_wave : array-like, optional
            New wavelength grid to rebin the spectrum onto. If provided, overrides factor.
        method : str, optional
            Aggregation method: 'mean' or 'median'. Default is 'mean'.
        """
        if new_wave is not None:
            # Rebin to new wavelength grid
            new_wave = np.asarray(new_wave)
            new_flux = np.zeros_like(new_wave)
            new_fluxerr = np.zeros_like(new_wave) if self.fluxerr is not None else None
            for i, w in enumerate(new_wave):
                # Find all original points within half-step of new grid point
                if i == 0:
                    dw = (new_wave[1] - new_wave[0]) / 2
                else:
                    dw = (new_wave[i] - new_wave[i-1]) / 2
                mask = (self.wave >= w - dw) & (self.wave < w + dw)
                if np.any(mask):
                    if method == 'median':
                        new_flux[i] = np.median(self.flux[mask])
                        if new_fluxerr is not None:
                            new_fluxerr[i] = np.median(self.fluxerr[mask])
                    else:
                        new_flux[i] = np.mean(self.flux[mask])
                        if new_fluxerr is not None:
                            new_fluxerr[i] = np.mean(self.fluxerr[mask])
                else:
                    new_flux[i] = np.nan
                    if new_fluxerr is not None:
                        new_fluxerr[i] = np.nan
            self.wave = new_wave
            self.flux = new_flux
            if new_fluxerr is not None:
                self.fluxerr = new_fluxerr
        elif factor is not None and factor > 1:
            # Rebin by integer factor
            n = len(self.wave)
            n_bins = n // factor
            trimmed = n_bins * factor
            wave = self.wave[:trimmed].reshape(n_bins, factor)
            flux = self.flux[:trimmed].reshape(n_bins, factor)
            if self.fluxerr is not None:
                fluxerr = self.fluxerr[:trimmed].reshape(n_bins, factor)
            else:
                fluxerr = None
            new_wave = np.mean(wave, axis=1)
            if method == 'median':
                new_flux = np.median(flux, axis=1)
                new_fluxerr = np.median(fluxerr, axis=1) if fluxerr is not None else None
            else:
                new_flux = np.mean(flux, axis=1)
                new_fluxerr = np.mean(fluxerr, axis=1) if fluxerr is not None else None
            self.wave = new_wave
            self.flux = new_flux
            if new_fluxerr is not None:
                self.fluxerr = new_fluxerr
        else:
            raise ValueError("Either scale > 1 or new_wave must be provided.")
        
    def crop(self, wbounds=None, wmask=None):
        """
        The crop function crops the spectrum to a specified wavelength range.
        
        Parameters: 
        xmin (float): The minimum wavelength of the crop region. Default is 4050 Angstroms. 
        xmax (float): The maximum wavelength of the crop region. Default is 7300 Angstroms. 
        
            Returns: None, but modifies self in place by cropping it to only include wavelengths between xmin and xmax.
        
        :param self: Used to Reference the object itself.
        :param xmin=4050: Used to Set the lower wavelength limit of the cropped spectrum.
        :param xmax=7300: Used to Select the wavelength range of interest.
        :return: A new spectrum object with the cropped wavelength range.
        """
        if wbounds is not None:
            wmin, wmax = wbounds
            wmask = (self.wave > wmin) & (self.wave < wmax)
        elif wmask is None:
            wmask = (self.wave > 4050) & (self.wave < 7300)
        else:
            raise ValueError("Either wbounds or wmask must be provided.")
        self.wave = self.wave[wmask]
        self.flux = self.flux[wmask]
        if self.fluxerr is not None:
            self.fluxerr = self.fluxerr[wmask]

    def plot_spectrum(self, ax=None):
        """
        Plot the spectrum using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        if self.fluxerr is not None:
            ax.errorbar(self.wave, self.flux, yerr=self.fluxerr, color="black", markersize=0, ls='none')
        ax.plot(self.wave, self.flux, label='Spectrum', color='black')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.legend(frameon=False)
        if ax is None:
            plt.show()
        
    def vac_to_air(self):
        """
        Convert vacuum to air wavelengths
        :param lam_vac - Wavelength in Angstroms
        :return: lam_air - Wavelength in Angstroms
        """
        self.wave = self.wave/_wave_convert(self.wave)
        self._vac_to_air_corrected = True

    def fit_host_SDSS(self, mask_host=False, custom=False, mask_list=[]):
        """
        The fit_host_sdss function fits a 5 host galaxy eigenspectra and an 10 AGNs eigenspectra to the observed spectrum.
        It takes as input:
        mask_host (bool): If True, it masks the host emission lines using _host_mask function. Default is False.
        custom (bool): If True, it masks user defined emission lines using _custom_mask function. Default is False. 
        mask_list ([float]): A list of wavelengths in angstroms that will be masked if custom=True .Default is empty list [].
        
             Returns: 
        
                self.(wave,flux,err) : The wavelength array , flux array and error array after fitting for host galaxy and AGN components respectively.
        
        :param self: Used to Access variables that belongs to the class.
        :param mask_host=False: Used to Mask the host.
        :param custom=False: Used to Mask the host.
        :param mask_list=[]: Used to Mask the data points that are not used in the fit.
        :return: The host and the agn component.
        """
        self._wave_with_host = self.wave
        self._flux_with_host = self.flux
        self._fluxerr_with_host= self.fluxerr
        a = _prepare_host(self.wave, self.flux, self.fluxerr, 5, 10)
        k = _fithost(a[3], a[4], a[1])
        host = 0
        for i in range(5):
            host += k[i] * a[3][i]
        if mask_host and custom==False:
            host= _host_mask(a[0], host)
        if mask_host and custom==True:
            host= _custom_mask(a[0], host, mask_list)
        ag_r = 0
        for i in range(10):
            ag_r += k[5 + i] * a[4][i]
        if _check_host(host) == True and _check_host(ag_r) == True:
            self.host = host
            self.flux = a[1] - host
            self.agn=ag_r
            self.wave = a[0]
            self.fluxerr = a[2]
            plt.figure()
            plt.plot(a[0], a[1])
            plt.plot(a[0], host, label="host")
            plt.plot(a[0], ag_r, label="AGN")
            plt.plot(a[0], host+ag_r, label="FIT")
        else:
            print('Host contribution is negliglable')

    def fit_host(self, mask_host=False, custom=False, mask_list=[]):
        """
        The fit_host function fits a host galaxy and an AGN to the observed spectrum.
        It takes as input:
        mask_host (bool): If True, it masks the host emission lines using _host_mask function. Default is False.
        custom (bool): If True, it masks user defined emission lines using _custom_mask function. Default is False. 
        mask_list ([float]): A list of wavelengths in angstroms that will be masked if custom=True .Default is empty list [].
        
             Returns: 
        
                self.(wave,flux,err) : The wavelength array , flux array and error array after fitting for host galaxy and AGN components respectively.
        
        :param self: Used to Access variables that belongs to the class.
        :param mask_host=False: Used to Mask the host.
        :param custom=False: Used to Mask the host.
        :param mask_list=[]: Used to Mask the data points that are not used in the fit.
        :return: The host and the agn component.
        """
        
        self.wave_old = self.wave
        self.flux_old = self.flux
        self.err_old = self.fluxerr
        lis = list(product(np.linspace(3, 10, 8), np.linspace(3, 15, 13)))
        ch = []
        HOST = []
        AGN = []
        FIT = []
        a = _prepare_host(self.wave, self.flux, self.fluxerr, 10, 15)
        for i in range(len(lis)):
            gal = int(lis[i][0])
            qso = int(lis[i][1])
            hs = a[3][:gal]
            ags = a[4][:qso]
            k = _fithost(hs, ags, a[1])
            host = 0
            for i in range(gal):
                host += k[i] * a[3][i]
            if mask_host and custom==False:
                host= _host_mask(a[0], host)
            if mask_host and custom==True:
                host= _custom_mask(a[0], host, mask_list)
            ag_r = 0
            for i in range(qso):
                ag_r += k[gal + i] * a[4][i]
            if _check_host(host) == True and _check_host(ag_r) == True:
                HOST.append(host)
                AGN.append(ag_r)
                FIT.append(host + ag_r)
                ch.append(_calculate_chi(a[1], a[2], host + ag_r, k))
        if not ch:
            print(
                "Uuups... All of the combinations returned negative host. Better luck with other AGN"
            )
            self.host=np.zeros(len(a[0]))
        else:
            idx = _find_nearest(ch)
            print("Number of galaxy components ", int(lis[idx][0]))
            print("Number of QSO components ", int(lis[idx][1]))
            print("Reduced chi square ", ch[idx])

            plt.figure()
            plt.plot(a[0], a[1])
            plt.plot(a[0], HOST[idx], label="host")
            plt.plot(a[0], AGN[idx], label="AGN")
            plt.plot(a[0], FIT[idx], label="FIT")
            self.host = HOST[idx]
            self.flux = a[1] - HOST[idx]
            self.agn=AGN[idx]
            self.wave = a[0]
            self.fluxerr = a[2]

            plt.legend()
            plt.savefig(self.name+'_host.pdf')

    def restore_host_subtraction(self):
        """
        Restore the spectra (wave, flux and error) to the one before host removal
        """
        self.wave = self._wave_with_host
        self.flux = self._flux_with_host
        self.fluxerr = self._fluxerr_with_host

    def _setup_spec4fit(self):
        self._wave_fit = self.wave
        self._flux_fit = self.flux
        self._fluxerr_fit = self.fluxerr
        # If there are nan values in the flux mask them
        if np.any(np.isnan(self.flux)):
            mask = np.isfinite(self.flux)
            self._wave_fit = self.wave[mask]
            self._flux_fit = self.flux[mask]
            if self.fluxerr is not None:
                self._fluxerr_fit = self.fluxerr[mask]

    def fit(self, model, ntrial=1, stat=Chi2(), method=LevMar()):
        """
        The fit function fits a model to the data. 
        It returns a tuple of (model, fit results).
        
        :param self: Used to Reference the class object.
        :param model: Used to Define the model that is used in the fit.
        :param ntrial=1: Used to Specify the number of times we want to repeat the fit.
        :return: The results of the fit.
        """
        self._setup_spec4fit()
        dataobj = Data1D("AGN", self._wave_fit, self._flux_fit, self._fluxerr_fit)
        gfit = Fit(dataobj, model, stat=stat, method=method)
        gres = gfit.fit()
        statistic=gres.dstatval
        if ntrial > 1:
            i=0
            while i < ntrial:
                gfit = Fit(dataobj, model, stat=stat, method=method)
                gres = gfit.fit()
                print(f"Iteration: {i+1}")
                if gres.dstatval==statistic:
                    break
                i+=1
        self.gres = gres
        self.dataobj = dataobj
        self.model = model
        return gfit
    
    def save_json(self, suffix="pars"):
        """
        The save_json function saves the parameter values in a JSON file.
        The filename is constructed from the name of the model and either 'pars' or 'samples'.
        
        :param self: Used to Refer to the object itself.
        :param suffix='pars': Used to Specify the name of the file that is saved.
        :return: A dictionary of the parameter names and values.
        """
        dicte = zip(self.gres.parnames, self.gres.parvals)
        res = dict(dicte)
        filename = self.name + "_" + suffix + ".json"
        with open(filename, "w") as fp:
            json.dump(res, fp)




    def _mc_resampling_NOparallelized(self, nsample=10, save_csv=True, filename=None, stat=Chi2(), method=LevMar()):
        if self.fluxerr is None:
            raise ValueError("Flux error is not defined. Please provide flux error for Monte Carlo sampling.")
        self._setup_spec4fit()

        dict_list = []
        for i in tqdm(range(nsample), desc="Monte Carlo Sampling"):
            # Perturb the flux based on the input error
            flux_perturbed = np.random.normal(loc=self._flux_fit, scale=self._fluxerr_fit)
            # Fit the model to the perturbed data
            d_perturbed = Data1D("AGN", self._wave_fit, flux_perturbed, self._fluxerr_fit)
            gfit_perturbed = Fit(d_perturbed, self.model, stat=stat, method=method)
            gres_perturbed = gfit_perturbed.fit()
            dicte = zip(gres_perturbed.parnames, gres_perturbed.parvals)
            res = dict(dicte)
            dict_list.append(res)
        df=pd.DataFrame(dict_list)

        if save_csv:
            if filename is None:
                filename = self.name + '_mc_pars.csv'
            df.to_csv(filename)
        self.mc_pars = df

    def _mc_resampling_parallel(self, nsample=10, save_csv=True, filename=None, stat=Chi2(), method=LevMar(), ncpu=None):
        if self.fluxerr is None:
            raise ValueError("Flux error is not defined. Please provide flux error for Monte Carlo sampling.")
        self._setup_spec4fit()

        # Prepare arguments for each sample
        ncpu = ncpu or max(1, mp.cpu_count() - 3)  # Leave some CPUs free
        base_seed = np.random.randint(0, 2**32)
        args_list = [
            (
                self._wave_fit,
                self._flux_fit,
                self._fluxerr_fit,
                self.model,
                stat,
                method,
                base_seed + i  # Unique seed per sample
            )
            for i in range(nsample)
        ]

        def _mc_worker(args):
            # from sherpa.data import Data1D
            # from sherpa.fit import Fit
            # import numpy as np
            wave, flux, err, model, stat, method, seed = args
            rng = np.random.default_rng(seed)
            flux_perturbed = rng.normal(loc=flux, scale=err)
            d_perturbed = Data1D("AGN", wave, flux_perturbed, err)
            gfit = Fit(d_perturbed, model, stat=stat, method=method)
            gres = gfit.fit()
            return dict(zip(gres.parnames, gres.parvals))

        dict_list = []
        with mp.Pool(processes=ncpu) as pool:
            for res in tqdm(pool.imap_unordered(_mc_worker, args_list), total=nsample, desc="Monte Carlo Sampling"):
                dict_list.append(res)

        df = pd.DataFrame(dict_list)
        if save_csv:
            if filename is None:
                filename = self.name + '_mc_pars.csv'
            df.to_csv(filename, index=False)
        self.mc_pars = df
        return df

    def mc_resampling(self, nsample=10, save_csv=True, filename=None, stat=Chi2(), method=LevMar(), ncpu=None):
        """
        Perform Monte Carlo resampling of the spectrum and fit the model to each sample.
        
        Parameters
        ----------
        nsample : int, optional
            Number of Monte Carlo samples to generate. Default is 10.
        save_csv : bool, optional
            Whether to save the results to a CSV file. Default is True.
        filename : str, optional
            Name of the output CSV file. If None, defaults to '<spectrum_name>_mc_pars.csv'.
        stat : sherpa.stats.Stat, optional
            Statistical method for fitting. Default is Chi2().
        method : sherpa.optmethods.OptMethod, optional
            Optimization method for fitting. Default is LevMar().
        ncpu : int, optional
            Number of CPUs to use for parallel processing. If None, uses all available CPUs minus 3.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the fitted parameters from each Monte Carlo sample.
        """
        if ncpu is None:
            ncpu = max(1, mp.cpu_count() - 3)
        if ncpu > 1:
            return self._mc_resampling_parallel(nsample, save_csv, filename, stat, method, ncpu)
        else:
            return self._mc_resampling_NOparallelized(nsample, save_csv, filename, stat, method)
        




class make_spectrum(Spectrum):
    def __init__(self, wave, flux, fluxerr=None, ra=None, dec=None, z=None, name='spectrum'):
        super().__init__()
        self.wave = wave
        self.flux = flux
        if fluxerr is None:
            self.fluxerr = np.ones_like(wave)
        else:
            if len(fluxerr) != len(wave) and len(fluxerr) != 1:
                raise ValueError("fluxerr must be either a single value or the same length as wave.")
            elif len(fluxerr) == 1:
                self.fluxerr = np.full_like(wave, fluxerr)
            else:
                self.fluxerr = fluxerr
        self.ra = ra
        self.dec = dec
        self.z = z
        self.name = name

        frac = self.wave[1] / self.wave[0]
        self.velscale = np.log(frac) * c
        dlam = (frac - 1) * self.wave
        self.fwhm = 2.355 * dlam

class read_text(Spectrum):
    def __init__(self, filename, ra=None, dec=None, z=None, name='spectrum'):
        super().__init__()
        try:
            try:
                self.wave, self.flux, self.fluxerr = np.genfromtxt(filename, unpack=True)
            except ValueError:
                self.wave, self.flux = np.genfromtxt(filename, unpack=True)
                self.fluxerr = np.ones_like(self.wave)
        except Exception as e:
            raise ValueError(f"Error reading file {filename}. Check the file format. It should contain at least two columns for wavelength and flux, and optionally a third column for flux error.") from e
        self.name = name if name is not None else filename.split(".")[0]

        frac = self.wave[1] / self.wave[0]
        self.velscale = np.log(frac) * c
        dlam = (frac - 1) * self.wave
        self.fwhm = 2.355 * dlam