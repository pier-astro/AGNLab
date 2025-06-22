from astropy import units as u
from astropy import constants as const
from sfdmap2 import sfdmap
import numpy as np
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from sherpa.data import Data1D
from sherpa.fit import Fit
from sherpa.optmethods import LevMar
from sherpa.stats import Chi2
import pandas as pd
import yaml
import os
from . import tqdm # correct tqdm version based on the environment __init__ have setup
import multiprocess as mp
import warnings

from .tools import resample_spectrum, downsample_wave, vac_to_air, get_add_comps

script_dir = os.path.dirname(__file__)
sfdpath = os.path.join(script_dir, "sfddata")
c = const.c.to(u.km/u.s).value # Speed of light in km/s
plt.rcParams['axes.xmargin'] = 0

class Spectrum():
    def __init__(self):
        self._zcorrected = False
        self._dereddened = False
        self._vac_to_air_corrected = False

        # Result of AGN-Host decomposition
        self.host = None
        self.agn = None
        
        self.name = None
        self.ra = None
        self.dec = None
        self.z = None
        self.ebv = None

    def DeRedden(self, ebv=None):
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
        if self._dereddened:
            warnings.warn("Spectrum is already dereddened. Skipping dereddening step.", UserWarning)
        else:
            self.input_wave = self.wave.copy() if not hasattr(self, 'input_wave') else self.input_wave
            self.input_flux = self.flux.copy() if not hasattr(self, 'input_flux') else self.input_flux
            self.input_fluxerr = self.fluxerr.copy() if not hasattr(self, 'input_fluxerr') else self.input_fluxerr

            if ebv is not None:
                self.ebv = ebv
            else:
                m = sfdmap.SFDMap(sfdpath)
                self.ebv = m.ebv(self.ra, self.dec)
            self.flux = pyasl.unred(self.wave, self.flux, self.ebv)
            self._dereddened = True

    def zCorrection(self, redshift=None):
        """
        The zCorrection() function corrects the flux for redshift.
        It takes in a redshift and corrects the wavelength, flux, and error arrays by that given redshift.

        :param self: Used to Reference the class object.
        :param redshift=0: Used to Specify the redshift of the object.
        :return: The wavelength, flux and error arrays for the object at a redshift of z=0.
        """
        if self._zcorrected:
            warnings.warn("Spectrum is already redshift corrected. Skipping zCorrection step.", UserWarning)
        else:
            self.input_wave = self.wave.copy() if not hasattr(self, 'input_wave') else self.input_wave
            self.input_flux = self.flux.copy() if not hasattr(self, 'input_flux') else self.input_flux
            self.input_fluxerr = self.fluxerr.copy() if not hasattr(self, 'input_fluxerr') else self.input_fluxerr
            if redshift is not None:
                self.z = redshift
            self.wave = self.wave / (1 + self.z)
            self.flux = self.flux * (1 + self.z)
            if self.fluxerr is not None:
                self.fluxerr = self.fluxerr * (1 + self.z)
            self.fwhm = self.fwhm / (1 + self.z)
            self._zcorrected = True

    def vac_to_air(self):
        """
        Convert vacuum to air wavelengths
        :param lam_vac - Wavelength in Angstroms
        :return: lam_air - Wavelength in Angstroms
        """
        if self._vac_to_air_corrected:
            warnings.warn("Spectrum is already converted to air wavelengths. Skipping vac_to_air step.", UserWarning)
        else:
            self.input_wave = self.wave.copy() if not hasattr(self, 'input_wave') else self.input_wave
            self.wave = vac_to_air(self.wave)
            self._vac_to_air_corrected = True

    def reset(self):
        """
        Reset the spectrum to its original state before any corrections.
        This will restore the original wavelength, flux, and flux error arrays.
        """
        if hasattr(self, 'input_wave'):
            self.wave = self.input_wave
            self.flux = self.input_flux
            self.fluxerr = self.input_fluxerr
            self._zcorrected = False
            self._dereddened = False
            self._vac_to_air_corrected = False
        else:
            raise ValueError("No input data found. Please ensure the spectrum has been initialized with input data.")

    def rebin(self, factor:int=None, new_wave=None, fill=np.nan, method='flux-conserving'):
        """
        Rebin the spectrum either by an integer scaling factor or to a new wavelength grid.
        Aggregation can be 'flux-conserving' (default), 'mean' or 'median'.
        """        
        if new_wave is not None:
            self.flux, self.fluxerr = resample_spectrum(self.wave, self.flux, self.fluxerr, new_wave=new_wave, fill=fill, method=method)
            self.wave = new_wave
        elif factor is not None and factor > 1:
            new_wave = downsample_wave(self.wave, factor)
            self.flux, self.fluxerr = resample_spectrum(self.wave, self.flux, self.fluxerr, new_wave=new_wave, fill=fill, method=method)
            self.wave = new_wave
        else:
            raise ValueError("Either factor > 1 or new_wave must be provided.")
        
    def crop(self, wbounds=None, wmask=None):
        """
        The crop function crops the spectrum to a specified wavelength range.
        """
        if wbounds is not None:
            wmin, wmax = wbounds
            wmask = (self.wave > wmin) & (self.wave < wmax)
        elif wmask is not None:
            wmask = wmask
        else:
            raise ValueError("Either wbounds or wmask must be provided.")
        self.wave = self.wave[wmask]
        self.flux = self.flux[wmask]
        if self.fluxerr is not None:
            self.fluxerr = self.fluxerr[wmask]
        if self.agn is not None:
            self.agn = self.agn[wmask]
        if self.host is not None:
            self.host = self.host[wmask]

    def plot_spectrum(self, ax=None):
        """
        Plot the spectrum using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, creates a new figure and axes.
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True
        if not np.all(self.fluxerr == 1):
            ax.errorbar(self.wave, self.flux, yerr=self.fluxerr, color="black", markersize=0, ls='none')
        ax.plot(self.wave, self.flux, label='Spectrum', color='black', drawstyle='steps-mid')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.legend(frameon=False)
        if created_fig:
            plt.show()
        
    def _check_spec4fit(self):
        if np.any(np.isnan(self.wave)) or np.any(np.isnan(self.flux)) or np.any(np.isnan(self.fluxerr)):
            raise ValueError("Spectrum contains NaN values in wave or flux. Please clean the data before fitting.")

    def fit(self, model, ntrial=1, stat=Chi2(), method=LevMar()):
        """
        The fit function fits a model to the data. 
        It returns a tuple of (model, fit results).
        
        :param self: Used to Reference the class object.
        :param model: Used to Define the model that is used in the fit.
        :param ntrial=1: Used to Specify the number of times we want to repeat the fit.
        :return: The results of the fit.
        """
        self._check_spec4fit()
        dataobj = Data1D("AGN", self.wave, self.flux, self.fluxerr)
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

        try:
            self.components = get_add_comps(model)
        except:
            warnings.warn("Error extracting components from the model. Try manually the tools.get_add_comp() function.", UserWarning)

        return gfit
    
    def save_pars(self, filename=None):
        """
        Saves the parameter values in a YAML file.
        The filename is constructed from the name of the model and either 'pars' or 'samples'.

        :param self: Reference to the object itself.
        :param suffix='pars': Suffix for the filename.
        :return: A dictionary of the parameter names and values.
        """
        if filename is None:
            name = self.name if self.name is not None else "spectrum"
            filename = name + "_pars.yaml"
        else:
            if not filename.endswith('.yaml'):
                filename += '.yaml'
        res = dict(zip(self.gres.parnames, self.gres.parvals))
        with open(filename, "w") as fp:
            yaml.dump(res, fp, default_flow_style=False)




    def _mc_resampling_NOparallelized(self, nsample=10, stat=Chi2(), method=LevMar()):
        if self.fluxerr is None:
            raise ValueError("Flux error is not defined. Please provide flux error for Monte Carlo sampling.")
        self._check_spec4fit()

        dict_list = []
        for i in tqdm(range(nsample), desc="Monte Carlo Sampling"):
            # Perturb the flux based on the input error
            flux_perturbed = np.random.normal(loc=self.flux, scale=self.fluxerr)
            # Fit the model to the perturbed data
            d_perturbed = Data1D("AGN", self.wave, flux_perturbed, self.fluxerr)
            gfit_perturbed = Fit(d_perturbed, self.model, stat=stat, method=method)
            gres_perturbed = gfit_perturbed.fit()
            dicte = zip(gres_perturbed.parnames, gres_perturbed.parvals)
            res = dict(dicte)
            dict_list.append(res)
        df=pd.DataFrame(dict_list)
        return df

    def _mc_resampling_parallel(self, nsample=10, stat=Chi2(), method=LevMar(), ncpu=None):
        if self.fluxerr is None:
            raise ValueError("Flux error is not defined. Please provide flux error for Monte Carlo sampling.")
        self._setup_spec4fit()

        # Prepare arguments for each sample
        ncpu = ncpu or max(1, mp.cpu_count() - 3)  # Leave some CPUs free
        base_seed = np.random.randint(0, 2**32)
        args_list = [
            (
                self.wave,
                self.flux,
                self.fluxerr,
                self.model,
                stat,
                method,
                base_seed + i  # Unique seed per sample
            )
            for i in range(nsample)
        ]

        ### --- Worker ---
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
        ### --- End Worker ---

        dict_list = []
        with mp.Pool(processes=ncpu) as pool:
            for res in tqdm(pool.imap_unordered(_mc_worker, args_list), total=nsample, desc="Monte Carlo Sampling"):
                dict_list.append(res)
        
        df = pd.DataFrame(dict_list)
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
            df = self._mc_resampling_parallel(nsample, stat, method, ncpu)
        else:
            df = self._mc_resampling_NOparallelized(nsample, stat, method)

        if save_csv:
            if filename is None and self.name is not None:
                filename = self.name + '_mc_pars.csv'
            df.to_csv(filename, index=False)
        self.mc_pars = df
        




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

class read_txt(Spectrum):
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

        self.ra = ra
        self.dec = dec
        self.z = z
        self.name = name if name is not None else filename.split(".")[0]

        frac = self.wave[1] / self.wave[0]
        self.velscale = np.log(frac) * c
        dlam = (frac - 1) * self.wave
        self.fwhm = 2.355 * dlam