from __future__ import print_function
from __future__ import absolute_import
# Licensed under an MIT open source license - see LICENSE

from .utilities import *
import numpy as np
import scipy.ndimage as nd
import scipy.optimize as op
from scipy.integrate import quad
from scipy.stats import scoreatpercentile, percentileofscore


def dist_transform(labelisofil, filclean_all):
    '''

    Recombines the cleaned skeletons from final analysis and takes the
    Euclidean Distance Transform of each. Since each filament is in an
    array defined by its own size, the offsets need to be taken into account
    when adding back into the master array.

    Parameters
    ----------
    labelisofil : list
        Contains arrays of the cleaned individual skeletons
    filclean_all : numpy.ndarray
        Master array with all final skeletons.

    Returns
    -------
    dist_transform_all : numpy.ndarray
        A Euclidean Distance Transform of all of the skeletons combined.
    dist_transform_sep : list
        Contains the Euclidean Distance Transform of each
        individual skeleton.
    '''

    dist_transform_sep = []

    for skel_arr in labelisofil:
        if np.max(skel_arr) > 1:
            skel_arr[np.where(skel_arr > 1)] = 1
        dist_transform_sep.append(
            nd.distance_transform_edt(np.logical_not(skel_arr)))

    # Distance Transform of all cleaned filaments
    dist_transform_all = nd.distance_transform_edt(
        np.logical_not(filclean_all))

    return dist_transform_all, dist_transform_sep


def cyl_model(distance, rad_profile, img_beam):
    '''

    Fits the radial profile of filament to a cylindrical model
    (see Arzoumanian et al. (2011)).

    Parameters
    ----------
    distance : list
        Distances from the skeleton.
    rad_profile : list
        Intensity values from the image.
    img_beam : float
        FWHM of the beam size.

    Returns
    -------
    fit : numpy.ndarray
        Fit values.
    fit_errors : numpy.ndarray
        Fit errors.
    model : function
        Function used to fit the profile.
    parameters : list
        Names of the parameters.
    fail_flag : bool
        Identifies a failed fit.
    '''

    p0 = (np.max(rad_profile), 0.1, 2.0)

    A_p_func = lambda u, p: (1 + u ** 2.) ** (-p / 2.)

    def model(r, *params):
        peak_dens, r_flat, p = params[0], params[1], params[2]

        A_p = quad(A_p_func, -np.inf, np.inf, args=(p))[0]

        return A_p * (peak_dens * r_flat) / (1 + r / r_flat) ** ((p - 1) / 2.)

    try:
        fit, cov = op.curve_fit(
            model, distance, rad_profile, p0=p0, maxfev=100*(len(distance)+1))
        fit_errors = np.sqrt(np.diag(cov))
    except:
        fit, cov = p0, None
        fit_errors = cov

    # Deconvolve the width with the beam size.
    # deconv = (2.35*abs(fit[1])**2.) - img_beam**2.
    # if deconv>0:
    # 	fit[1] = np.sqrt(deconv)
    # else:
    # 	fit[1] = "Neg. FWHM"
    fail_flag = False
    if cov is None or (fit_errors > fit).any():
        fail_flag = True

    parameters = [r"$\pho_c$", "r_{flat}", "p"]

    return fit, fit_errors, model, parameters, fail_flag


def gauss_model(distance, rad_profile, weights, img_beam):
    '''
    Fits a Gaussian to the radial profile of each filament.

    Parameters
    ----------
    distance : list
        Distances from the skeleton.
    rad_profile : list
        Intensity values from the image.
    weights : list
        Weights to be used for the fit.
    img_beam : float
        FWHM of the beam size.

    Returns
    -------
    fit : numpy.ndarray
        Fit values.
    fit_errors : numpy.ndarray
        Fit errors.
    gaussian : function
        Function used to fit the profile.
    parameters : list
        Names of the parameters.
    fail_flag : bool
        Identifies a failed fit.
    '''

    p0 = (np.max(rad_profile), 0.1, np.min(rad_profile))
    parameters = ["Amplitude", "Width", "Background", "FWHM"]

    def gaussian(x, *p):
        '''
        Parameters
        ----------
        x : list or numpy.ndarray
            1D array of values where the model is evaluated
        p : tuple
            Components are:
            * p[0] Amplitude
            * p[1] Width
            * p[2] Background
        '''
        return (p[0]-p[2]) * np.exp(-1 * np.power(x, 2) /
                                    (2 * np.power(p[1], 2))) + p[2]

    try:
        fit, cov = op.curve_fit(gaussian, distance, rad_profile, p0=p0,
                                maxfev=100*(len(distance)+1), sigma=weights)
        fit_errors = np.sqrt(np.diag(cov))
    except:
        print("curve_fit failed.")
        fit, fit_errors = p0, None
        return fit, fit_errors, gaussian, parameters, True

    # Because of how function is defined,
    # fit function can get stuck at negative widths
    # This doesn't change the output though.
    fit[1] = np.abs(fit[1])

    # Deconvolve the width with the beam size.
    factor = 2 * np.sqrt(2 * np.log(2))  # FWHM factor
    deconv = (factor * fit[1]) ** 2. - img_beam ** 2.
    if deconv > 0:
        fit_errors = np.append(
            fit_errors, (factor * fit[1] * fit_errors[1]) / np.sqrt(deconv))
        fit = np.append(fit, np.sqrt(deconv))
    else:  # Set to zero, can't be deconvolved
        fit = np.append(fit, 0.0)
        fit_errors = np.append(fit_errors, 0.0)

    fail_flag = False
    if fit_errors is None or fit[0] < fit[2] or (fit_errors > fit).any():
        fail_flag = True

    return fit, fit_errors, gaussian, parameters, fail_flag


def lorentzian_model(distance, rad_profile, img_beam):
    '''
    Fits a Gaussian to the Lorentzian profile to each filament.

    Parameters
    ----------
    distance : list
        Distances from the skeleton.
    rad_profile : list
        Intensity values from the image.
    img_beam : float
        FWHM of the beam size.

    Returns
    -------
    fit : numpy.ndarray
        Fit values.
    fit_errors : numpy.ndarray
        Fit errors.
    lorentzian : function
        Function used to fit the profile.
    parameters : list
        Names of the parameters.
    fail_flag : bool
        Identifies a failed fit.     '''

    p0 = (np.max(rad_profile), 0.1, np.min(rad_profile))

    def lorentzian(x, *p):
        '''
        Parameters
        ----------
        x : list or numpy.ndarray
                1D array of values where the model is evaluated
        p : tuple
            Components are:
            * p[0] Amplitude
            * p[1] FWHM Width
            * p[2] Background

        '''
        return (p[0] - p[2]) * (0.5 * p[1]) ** 2 / \
            ((0.5 * p[1]) ** 2 + x ** 2) + p[2]

    try:
        fit, cov = op.curve_fit(
            lorentzian, distance, rad_profile, p0=p0,
            maxfev=100 * (len(distance) + 1))
        fit_errors = np.sqrt(np.diag(cov))
    except:
        fit, fit_errors = p0, None

    fit = list(fit)
    # Deconvolve the width with the beam size.
    deconv = fit[1] ** 2. - img_beam ** 2.
    if deconv > 0:
        fit[1] = np.sqrt(deconv)
    else:  # Set to zero, can't be deconvolved
        fit[1] = 0.0

    fail_flag = False
    if fit_errors is None or (fit_errors > fit).any():
        fail_flag = True

    parameters = ["Amplitude", "Width", "Background"]

    return fit, fit_errors, lorentzian, parameters, fail_flag


def nonparam_width(distance, rad_profile, unbin_dist, unbin_prof,
                   img_beam, bkg_percent, peak_percent):
    '''
    Estimate the width and peak brightness of a filament non-parametrically.
    The intensity at the peak and background is estimated. The profile is then
    interpolated over in order to find the distance corresponding to these
    intensities. The width is then estimated by finding the distance where
    the intensity drops to 1/e.

    Parameters
    ----------
    distance : list
        Binned distances from the skeleton.
    rad_profile : list
        Binned intensity values from the image.
    unbin_dist : list
        Unbinned distances.
    unbin_prof : list
        Unbinned intensity values.
    img_beam : float
        FWHM of the beam size.
    bkg_percent : float
        Percentile of the data to estimate the background.
    peak_percent : float
        Percentile of the data to estimate the peak of the profile.

    Returns
    -------
    params : numpy.ndarray
        Estimated parameter values.
    param_errors : numpy.ndarray
        Estimated errors.
    fail_flag : bool
        Indicates whether the fit failed.
    '''

    fail_flag = False

    # Find the intensities at the given percentiles
    bkg_intens = scoreatpercentile(rad_profile, bkg_percent)
    peak_intens = scoreatpercentile(rad_profile, peak_percent)

    # Interpolate over the bins in distance
    interp_bins = np.linspace(0.0, np.max(distance), 10 * len(distance))
    interp_profile = np.interp(interp_bins, distance, rad_profile)

    # Find the width by looking for where the intensity drops to 1/e from the
    # peak
    target_intensity = (peak_intens - bkg_intens) / \
        (2 * np.sqrt(2 * np.log(2))) + bkg_intens
    fwhm_width = interp_bins[
        np.where(interp_profile ==
                 find_nearest(interp_profile, target_intensity))][0]

    # Estimate the width error by looking +/-5 percentile around the target
    # intensity
    target_percentile = percentileofscore(rad_profile, target_intensity)
    upper = scoreatpercentile(
        rad_profile, np.min((100, target_percentile + 5)))
    lower = scoreatpercentile(rad_profile, np.max((0, target_percentile - 5)))

    fwhm_error = np.max(unbin_dist[(unbin_prof > lower) * (unbin_prof < upper)]) -\
        np.min(unbin_dist[(unbin_prof > lower) * (unbin_prof < upper)])

    # Deconvolve the width with the beam size.
    factor = 2 * np.sqrt(2 * np.log(2))  # FWHM factor

    width = fwhm_width / factor
    width_error = fwhm_error / factor

    deconv = fwhm_width ** 2. - img_beam ** 2.
    if deconv > 0:
        fwhm_width = np.sqrt(deconv)
        fwhm_error = (factor * width * width_error) / fwhm_width
    else:  # Set to zero, can't be deconvolved
        # If you can't devolve it, set it to minimum, which is the beam-size.
        fwhm_width = 0.0
        fwhm_error = 0.0

    # Check where the "background" and "peak" are. If the peak distance is greater,
    # we are simply looking at a bad radial profile.
    bkg_dist = np.median(
        interp_bins[np.where(interp_profile == find_nearest(interp_profile, bkg_intens))])
    peak_dist = np.median(
        interp_bins[np.where(interp_profile == find_nearest(interp_profile, peak_intens))])
    bkg_error = np.std(
        unbin_prof[unbin_dist >= find_nearest(unbin_dist, bkg_dist)])
    peak_error = np.std(
        unbin_prof[unbin_dist <= find_nearest(unbin_dist, peak_dist)])
    if peak_dist > bkg_dist or fwhm_error > fwhm_width:
        fail_flag = True

    params = np.array([peak_intens, width, bkg_intens, fwhm_width])
    param_errors = np.array([peak_error, width_error, bkg_error, fwhm_error])

    return params, param_errors, fail_flag


def radial_profile(img, dist_transform_all, dist_transform_sep, offsets,
                   img_scale, bins=None, bintype="linear", weighting="number",
                   return_unbinned=True, auto_cut=True,
                   pad_to_distance=0.15, max_distance=0.3):
    '''
    Fits the radial profiles to all filaments in the image.

    Parameters
    ----------
    img : numpy.ndarray
              The original image.
    dist_transform_all : numpy.ndarray
        The distance transform of all the skeletons.
    dist_transform_sep : list
        The distance transforms of each individual skeleton.
    offsets : list
        Contains the indices where each skeleton was cut out of the original
        array.
    img_scale : float
        Pixel to physical scale conversion.
    bins : numpy.ndarray, optional
        Bins to use for the profile fitting.
    bintype : str, optional
        "linear" for linearly spaced bins; "log" for log-spaced bins.
        Default is "linear".
    weighting : str, optional
        "number" is by the number of points in each bin; "var" is the
        variance of the values in each bin. Default is "number".
    return_unbinned : bool
        If True, returns the unbinned data as well as the binned.
    auto_cut : bool, optional
        Enables the auto cutting routines.
    pad_to_distance : float, optional
        Pad the profile out to the specified distance (in pc).
        If set to 0.0, the profile will not be padded.
    max_distance : float, optional
        Cuts the profile at the specified physical distance (in pc).

    Returns
    -------
    bin_centers : numpy.ndarray
        Center of the bins used in physical units.
    radial_prof : numpy.ndarray
        Binned intensity profile.
    weights : numpy.ndarray
        Weights evaluated for each bin.
    '''

    width_value = []
    width_distance = []
    nonlocalpix = []
    x, y = np.where(np.isfinite(dist_transform_sep))
    x_full = x + offsets[0][0]  # Transform into coordinates of master image
    y_full = y + offsets[0][1]

    for i in range(len(x)):
        # Check overall distance transform to make sure pixel belongs to proper
        # filament
        if img[x_full[i], y_full[i]]!=0.0 and np.isfinite(img[x_full[i], y_full[i]]):
            if dist_transform_sep[x[i], y[i]] <= dist_transform_all[x_full[i], y_full[i]]:
                width_value.append(img[x_full[i], y_full[i]])
                width_distance.append(dist_transform_sep[x[i], y[i]])
            else:
                nonlocalpix.append([x[i], y[i], x_full[i], y_full[i]])

    if pad_to_distance>0.0 and np.max(width_distance)*img_scale < pad_to_distance:
        pad = int(
            (0.15 - np.max(width_distance) * img_scale) * img_scale ** -1)
        for pix in nonlocalpix:
            if dist_transform_sep[pix[0], pix[1]] <= dist_transform_all[pix[2], pix[3]] + pad:
                width_value.append(img[pix[2], pix[3]])
                width_distance.append(dist_transform_sep[pix[0], pix[1]])

    width_value = np.asarray(width_value)
    width_distance = np.asarray(width_distance)

    if max_distance is not None:
        width_value = width_value[width_distance <= max_distance/img_scale]
        width_distance = \
            width_distance[width_distance <= max_distance/img_scale]

    # Binning
    if bins is None:
        nbins = np.sqrt(len(width_value))
        maxbin = np.max(width_distance)
        if bintype is "log":
            # bins must start at 1 if logspaced
            bins = np.logspace(0, np.log10(maxbin), nbins + 1)
        elif bintype is "linear":
            bins = np.linspace(0, maxbin, nbins + 1)

    whichbins = np.digitize(width_distance, bins)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0
    radial_prof = np.array(
        [np.median(width_value[(whichbins == bin)]) for bin in
         range(1, int(nbins) + 1)])

    if weighting == "number":
        weights = np.array([whichbins[whichbins == bin].sum()
                            for bin in range(1, int(nbins) + 1)])
    elif weighting == "var":
        weights = np.array(
            [np.nanvar(width_value[whichbins == bin]) for bin in
             range(1, int(nbins) + 1)])
        weights[np.isnan(weights)] = 0.0  # Empty bins

    # Ignore empty bins
    radial_prof = radial_prof[weights > 0]
    bin_centers = bin_centers[weights > 0]
    weights = weights[weights > 0]

    # Put bins in the physical scale.
    bin_centers *= img_scale
    width_distance *= img_scale

    if auto_cut:
        bin_centers, radial_prof, weights = \
            _smooth_and_cut(bin_centers, radial_prof, 0.1, weights)

    if return_unbinned:
        width_distance = width_distance[np.isfinite(width_value)]
        width_value = width_value[np.isfinite(width_value)]
        return bin_centers, radial_prof, weights, width_distance, width_value
    else:
        return bin_centers, radial_prof, weights


def _smooth_and_cut(bins, values, kern_size, weights, interp_factor=10,
                    pad_cut=5):
    '''
    Smooth the radial profile and cut if it increases at increasing
    distance. Also checks for profiles with a plateau between two decreasing
    profiles and cut out the last one (as it is not the local profile).

    Parameters
    ----------
    bins : numpy.ndarray
        Bins for the profile.
    values : numpy.ndarray
        Values in each bin.
    kern_size : int or float
        If >1, is the number of bins to use in the smoothing. If <1, takes
        fraction of the data for smoothing.
    weights : numpy.ndarray
        Weights for each bin. These are only clipped to the same position as
        the rest of the profile. Otherwise, no alteration is made.
    interp_factor : int, optional
        The factor to increase the number of bins by for interpolation.
    pad_cut : int, optional
        Add additional bins after the cut is found. The smoothing often cuts
        out some bins which follow the desired profile.
    '''

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import argrelmax, argrelmin

    if kern_size < 1:
        kern_size *= values.size
        kern_size = round(kern_size)

    smooth_val = gaussian_filter1d(values, kern_size)

    smooth_bins = np.linspace(bins.min(), bins.max(), interp_factor*bins.size)

    smooth_val = np.interp(smooth_bins, bins, smooth_val)

    grad = np.gradient(smooth_val, smooth_bins[1]-smooth_bins[0])

    grad = gaussian_filter1d(grad, kern_size)

    cut = crossings_nonzero_all(grad)

    # Check for evidence of second drop-off
    new_cut = None

    # Look for local max and mins (must hold True for range of ~0.05 pc)
    loc_mins = argrelmin(grad,
                         order=int(0.05/(smooth_bins[1] - smooth_bins[0])))[0]
    loc_maxs = argrelmax(grad,
                         order=int(0.05/(smooth_bins[1] - smooth_bins[0])))[0]

    # Discard below 0.1 pc.
    loc_mins = loc_mins[smooth_bins[loc_mins] > 0.1]
    loc_maxs = loc_maxs

    if loc_mins.size > 0 and loc_maxs.size > 0:
        i = 0
        while True:
            loc_min = loc_mins[i]

            difference = loc_min - loc_maxs
            if (difference > 0).any():
                new_cut = loc_maxs[np.argmin(difference[difference > 0])]
                if smooth_bins[new_cut] > 0.1:
                    break

            i += 1

            if i == loc_mins.size:
                break

    if new_cut == 0:
        new_cut = None

    if cut.size == 0:
        if new_cut is None:
            return bins, values, weights
        else:
            cut_posn = _nearest_idx(bins, smooth_bins[new_cut])

            end_diff = bins.size - cut_posn
            if end_diff < pad_cut:
                cut_posn += end_diff
            else:
                cut_posn += pad_cut

            cut_bins = bins[:cut_posn]
            cut_vals = values[:cut_posn]
            cut_weights = weights[:cut_posn]

            return cut_bins, cut_vals, cut_weights

    else:
        if new_cut is None:
            cut_used = cut[0]
        elif new_cut >= cut[0]:
            cut_used = cut[0]
        else:
            cut_used = new_cut

        cut_posn = _nearest_idx(bins, smooth_bins[cut_used])

        end_diff = bins.size - cut_posn
        if end_diff < pad_cut:
            cut_posn += end_diff
        else:
            cut_posn += pad_cut

        cut_bins = bins[:cut_posn]
        cut_vals = values[:cut_posn]
        cut_weights = weights[:cut_posn]

        return cut_bins, cut_vals, cut_weights


def _nearest_idx(array, value):
    return (np.abs(array-value)).argmin()


def crossings_nonzero_all(data):
    assert isinstance(data, np.ndarray)
    pos = data > 0
    npos = ~pos
    return ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
