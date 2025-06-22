import numpy as np
import warnings
from . import instrument

def make_bins(wavs):
    edges = np.zeros(wavs.shape[0]+1)
    widths = np.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    widths[:-1] = edges[1:-1] - edges[:-2]
    return edges, widths

def downsample_wave(wave, factor):
    """
    Create a new wavelength grid by grouping the original grid by the given factor.
    The new grid is the mean of each group of 'factor' consecutive wavelengths.
    """
    wave = np.asarray(wave)
    n = len(wave)
    n_bins = n // factor
    trimmed = n_bins * factor
    new_wave = np.mean(wave[:trimmed].reshape(n_bins, factor), axis=1)
    return new_wave

def resample_spectrum(wave, flux, fluxerr=None, new_wave=None, fill=np.nan, method='flux-conserving', verbose=True):
    """
    Resample a spectrum using either flux-conserving binning (default), mean, or median.
    Can resample to a new wavelength grid.
    """
    wave = np.asarray(wave)
    new_wave = np.asarray(new_wave)
    flux = np.asarray(flux)
    if flux.ndim == 1:
        flux = flux[None, :]  # shape (1, n_wave)
    n_spec, n_wave = flux.shape

    if fluxerr is not None:
        fluxerr = np.asarray(fluxerr)
        if fluxerr.ndim == 1:
            fluxerr = fluxerr[None, :]

    if method == 'flux-conserving':
        old_edges, old_widths = make_bins(wave)
        new_edges, new_widths = make_bins(new_wave)
        n_new = new_wave.shape[0]
        new_flux = np.full((n_spec, n_new), fill, dtype=flux.dtype)
        new_fluxerr = np.full((n_spec, n_new), fill, dtype=fluxerr.dtype) if fluxerr is not None else None

        start_indices = np.searchsorted(old_edges, new_edges[:-1], side='right') - 1
        stop_indices = np.searchsorted(old_edges, new_edges[1:], side='left')

        for j in range(n_new):
            start = start_indices[j]
            stop = stop_indices[j] - 1

            if start < 0 or stop >= n_wave:
                if verbose and (j == 0 or j == n_new-1):
                    warnings.warn(
                        "resample_spectrum: new_wave contains values outside the range in wave, output will be filled with 'fill'.",
                        category=RuntimeWarning,
                    )
                continue

            if stop == start:
                new_flux[:, j] = flux[:, start]
                if fluxerr is not None:
                    new_fluxerr[:, j] = fluxerr[:, start]
            else:
                start_factor = ((old_edges[start+1] - new_edges[j]) /
                                (old_edges[start+1] - old_edges[start]))
                end_factor = ((new_edges[j+1] - old_edges[stop]) /
                              (old_edges[stop+1] - old_edges[stop]))

                widths = old_widths[start:stop+1].copy()
                widths[0] *= start_factor
                widths[-1] *= end_factor

                denom = np.sum(widths)
                if denom == 0:
                    new_flux[:, j] = fill
                    if fluxerr is not None:
                        new_fluxerr[:, j] = fill
                    continue

                f_widths = widths * flux[:, start:stop+1]
                new_flux[:, j] = np.sum(f_widths, axis=1) / denom
                if fluxerr is not None:
                    e_wid = widths * fluxerr[:, start:stop+1]
                    new_fluxerr[:, j] = np.sqrt(np.sum(e_wid**2, axis=1)) / denom

        # Squeeze if only one spectrum
        if new_flux.shape[0] == 1:
            new_flux = new_flux[0]
            if new_fluxerr is not None:
                new_fluxerr = new_fluxerr[0]
        return (new_flux, new_fluxerr) if fluxerr is not None else new_flux

    elif method in ('mean', 'median'):
        new_flux = np.full(new_wave.shape, fill, dtype=flux.dtype)
        new_fluxerr = np.full(new_wave.shape, fill, dtype=fluxerr.dtype) if fluxerr is not None else None
        for i, w in enumerate(new_wave):
            # Find all original points within half-step of new grid point
            if i == 0:
                dw = (new_wave[1] - new_wave[0]) / 2
            else:
                dw = (new_wave[i] - new_wave[i-1]) / 2
            mask = (wave >= w - dw) & (wave < w + dw)
            if np.any(mask):
                if method == 'median':
                    new_flux[i] = np.median(flux[mask])
                    if fluxerr is not None:
                        new_fluxerr[i] = np.median(fluxerr[mask])
                else:
                    new_flux[i] = np.mean(flux[mask])
                    if fluxerr is not None:
                        new_fluxerr[i] = np.mean(fluxerr[mask])
        return (new_flux, new_fluxerr) if fluxerr is not None else new_flux

    else:
        raise ValueError("Unknown method: choose 'flux-conserving', 'mean', or 'median'.")
    


def convert_to_vacuum(wave):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param wave - Wavelength in Angstroms
    :return: conversion factor
    """
    wave = np.asarray(wave)
    sigma2 = (1e4/wave)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
    return fact

def vac_to_air(vac_wave):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms
    """
    return vac_wave/convert_to_vacuum(vac_wave)

def get_mask(x, intervals, mask_inside=True):
    """
    Get a mask of the intervals
    """
    w_masks = np.array([np.logical_and(x >= i[0], x <= i[1]) for i in intervals])
    if mask_inside:
        return np.all(w_masks == False, axis=0)
    else:
        return np.any(w_masks, axis=0)

def compute_reduced_chi2(data, data_err, model, npars):
    """
    Compute the reduced chi-squared statistic for a given model and data.

    Parameters
    ----------
    data : array-like
        Observed data (flux).
    data_err : array-like
        Uncertainties in the observed data.
    model : array-like
        Model predictions.
    npars : int
        Number of parameters in the model.

    Returns
    -------
    float
        Reduced chi-squared value.
    """
    residuals = (data - model) / data_err
    chi2 = np.sum(residuals**2)
    dof = len(data) - npars  # degrees of freedom
    return chi2 / dof if dof > 0 else np.nan  # Avoid division by zero




# ### BASIC DEFINITION
# def get_comps(model):
#     components = []
#     # Base case: if the model is a leaf node (no parts), return it in a list
#     if not hasattr(model, 'parts'):
#         return [model]
    
#     def traverse_parts(parts):
#         for part in parts:
#             if hasattr(part, 'parts'):
#                 traverse_parts(part.parts)
#             else:
#                 components.append(part)
    
#     if hasattr(model, 'parts'):
#         traverse_parts(model.parts)
    
#     return components

# def get_add_comps(model):
#     """
#     Recursively decompose a Sherpa compound model into additive components.
    
#     Parameters:
#     model (Sherpa model instance): The model to decompose.
    
#     Returns:
#     list: A list of Sherpa model instances representing each additive component.
#     """
#     # Base case: if the model is a leaf node (no parts), return it in a list
#     if not hasattr(model, 'parts'):
#         return [model]
    
#     # Split into left and right parts
#     left_part, right_part = model.parts
#     op = model.opstr  # Get the operation as a string
    
#     # Recursively get components for left and right parts
#     left_components = get_add_comps(left_part)
#     right_components = get_add_comps(right_part)
    
#     # Handle additive operations
#     if op in ('+', '-'):
#         # # For subtraction, negate the right components
#         # if op == '-':
#         #     right_components = [(-1) * comp for comp in right_components]
#         return left_components + right_components
#     else:
#         # Handle multiplicative (or other) operations by combining all pairs
#         components = []
#         for lcomp in left_components:
#             for rcomp in right_components:
#                 components.append(model.op(lcomp, rcomp))
#         return components

### RESPONSE MODEL AWARE VERSION
def get_comps(model):
    """
    Recursively decompose a Sherpa compound model into its components.
    """
    is_rsp_mod = isinstance(model, inst.ConvolvedModel)
    rsp_model = None
    if is_rsp_mod:
        rsp_model = model.response_model
        model = model.source_model
    
    components = []
    # Base case: if the model is a leaf node (no parts), return it in a list
    if not hasattr(model, 'parts'):
        return [model]
    
    def traverse_parts(parts):
        for part in parts:
            if hasattr(part, 'parts'):
                traverse_parts(part.parts)
            else:
                components.append(part)
    
    if hasattr(model, 'parts'):
        traverse_parts(model.parts)

    if is_rsp_mod:
        components = [rsp_model(comp) for comp in components]

    return components


def get_add_comps(model):
    """
    Recursively decompose a Sherpa compound model into additive components.
    
    Parameters:
    model (Sherpa model instance): The model to decompose.
    
    Returns:
    list: A list of Sherpa model instances representing each additive component.
    """
    is_rsp_mod = isinstance(model, instrument.ConvolvedModel)
    rsp_model = None
    if is_rsp_mod:
        rsp_model = model.response_model
        model = model.source_model
        
    
    # Base case: if the model is a leaf node (no parts), return it in a list
    if not hasattr(model, 'parts'):
        return [model]
    
    # Split into left and right parts
    left_part, right_part = model.parts
    op = model.opstr  # Get the operation as a string
    
    # Recursively get components for left and right parts
    left_components = get_add_comps(left_part)
    right_components = get_add_comps(right_part)
    
    # Handle additive operations
    if op in ('+', '-'):
        # # For subtraction, negate the right components
        # if op == '-':
        #     right_components = [(-1) * comp for comp in right_components]
        result = left_components + right_components
    else:
        # Handle multiplicative (or other) operations by combining all pairs
        components = []
        for lcomp in left_components:
            for rcomp in right_components:
                components.append(model.op(lcomp, rcomp))
        result = components

    # Add logic for response model if needed
    if is_rsp_mod:
        result = [rsp_model(comp) for comp in result]

    return result