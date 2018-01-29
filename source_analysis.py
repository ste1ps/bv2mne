#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import numpy as np


import mne
from mne.source_space import SourceSpaces
from mne.time_frequency.csd import CrossSpectralDensity, csd_epochs
from mne.source_estimate import SourceEstimate
from mne.beamformer import dics_source_power
from mne.beamformer._dics import dics_source_power_epochs
from mne import make_forward_solution
from mne.connectivity.spectral import (_epoch_spectral_connectivity,
                                       spectral_connectivity)

from data import create_param_dict

from data import read_serialize


def forward_model(subject, raw, fname_trans, src, subjects_dir, force_fixed=False, surf_ori=False, name='single-shell'):
    """construct forward model

    Parameters
    ----------
    subject : str
        The name of subject
    raw : instance of rawBTI
        functionnal data
    fname_trans : str
        The filename of transformation matrix
    src : instance of SourceSpaces | list
        Sources of each interest hemisphere
    subjects_dir : str
        The subjects directory
    force_fixed: Boolean
        Force fixed source orientation mode
    name : str
        Use to save output
       

    Returns
    -------
    fwd : instance of Forward
    -------
    Author : Alexandre Fabre
    """
    # files to save step
    fname_bem_model = subjects_dir + '{0}/bem/{0}-{1}-bem.fif'.format(subject, name)
    fname_bem_sol = subjects_dir + '{0}/bem/{0}-{1}-bem-sol.fif'.format(subject, name)
    fname_fwd = subjects_dir + '{0}/fwd/{0}-{1}-fwd.fif'.format(subject, name)

    # Make bem model: single-shell model. Depends on anatomy only.
    model = mne.make_bem_model(subject, conductivity=[0.3], subjects_dir=subjects_dir)
    mne.write_bem_surfaces(fname_bem_model, model)

    # Make bem solution. Depends on anatomy only.
    bem_sol = mne.make_bem_solution(model)
    mne.write_bem_solution(fname_bem_sol, bem_sol)

    # bem_sol=mne.read_bem_solution(fname_bem_sol)

    if len(src) == 2:
            # gather sources each the two hemispheres
            lh_src, rh_src = src
            src = lh_src + rh_src

    # Compute forward operator, commonly referred to as the gain or leadfield matrix.
    fwd = make_forward_solution(raw.info, fname_trans, src, bem_sol, fname_fwd, mindist=0.0, overwrite=True)

    # Set orientation of the source
    if force_fixed:
        # Force fixed
        fwd = mne.read_forward_solution(fname_fwd, force_fixed=True)
    elif surf_ori:
        # Surface normal
        fwd = mne.read_forward_solution(fname_fwd, surf_ori=True)
    else:
        # Free like a bird
        fwd = mne.read_forward_solution(fname_fwd)

    return fwd


def get_epochs_dics(epochs, fwd, tmin=None, tmax=None, tstep=0.005, win_lengths=0.2, mode='multitaper',
                    fmin=0, fmax= np.inf, fsum=True, n_fft=None, mt_bandwidth=None,
                    mt_adaptive=False, mt_low_bias=True, projs=None, verbose=None,
                    reg=0.01, label=None, pick_ori=None, on_epochs=True,
                    avg_tapers=True):
    """construct forward model

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs
    fwd : instance of Forward
        The solution of forward problem
    tmin : float | None
        Minimum time instant to consider. If None start at first sample
    tmax : float | None
        Maximum time instant to consider. If None end at last sample
    tstep : float | None 
        Time window step. If None, it's as large as the time interval (tmin - tmax)
    win_lengths: float | None
        The window size. If None, it's as large as the time interval (tmin - tmax)
    mode : 'multitaper' | 'fourier'
        Spectrum estimation mode
    fmin: float
        Minimum frequency of interest
    fmax : float
        Maximum frequency of interest
    fsum : bool
        Sum CSD values for the frequencies of interest. Summing is performed
        instead of averaging so that accumulated power is comparable to power
        in the time domain. If True, a single CSD matrix will be returned. If
        False, the output will be a list of CSD matrices.
    n_fft : int | None
        Length of the FFT. If None the exact number of samples between tmin and
        tmax will be used.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    projs : list of Projection | None
        List of projectors to use in CSD calculation, or None to indicate that
        the projectors from the epochs should be inherited.
    reg : float
        The regularization for the cross-spectral density.
    label : Label | None
        Restricts the solution to a given label.
    pick_ori : None | 'normal'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    on_epochs : bool
        Average on epoch
    on_tapers : bool
        Averge on tapers in csd computing

    Returns
    -------
    stc : list of SourceEstimate
    -------
    Author : Alexandre Fabre
    """
    

    # Default values
    if tmin is None:
        tmin = epochs.times[0]
    if tmax is None:
        tmax = epochs.times[-1] - win_lengths

    # Multiplying by 1e3 to avoid numerical issues
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))

    # Init power and time
    power = []
    time  = np.zeros(n_time_steps)

    print('Computing cross-spectral density from epochs...')
    for i_time in range(n_time_steps):

        win_tmin = tmin + i_time * tstep
        win_tmax = win_tmin + win_lengths
        time[i_time] = win_tmin + win_lengths/2

        avg_csds = None
        
        print('   From {0}s to {1}s'.format(win_tmin, win_tmax))

        # Compute cross-spectral density csd matrix (nChans, nChans, nTapers, nTrials)
        csds = csd_epochs(epochs, mode=mode, fmin=fmin, fmax=fmax, tmin=win_tmin,
                          tmax=win_tmax, fsum=fsum, n_fft=n_fft,
                          mt_bandwidth=mt_bandwidth, mt_adaptive= mt_adaptive,
                          mt_low_bias=mt_low_bias, projs=projs, verbose=verbose,
                          avg_tapers=avg_tapers, on_epochs=on_epochs)

        if len(csds[0].data.shape)>2:
            # Compute average csd to compute beamforming DICS filter
            avg_csds = csd_epochs(epochs, mode='multitaper', fmin=fmin, fmax=fmax, tmin=win_tmin,
                                  tmax=win_tmax, fsum=fsum, n_fft=n_fft, mt_bandwidth=mt_bandwidth,
                                  mt_adaptive= mt_adaptive,  mt_low_bias=mt_low_bias, projs=projs,
                                  verbose=verbose, avg_tapers=True, on_epochs=True)

        # Perform DICS
        power_time, vertno = dics_source_power_epochs(epochs.info, fwd, csds, avg_csds)

        # Append time slices
        power.append(power_time)

    return power, time



def source2atlas(data, baseline, atlas):
    '''
    Transform source estimates to atalas-based
    i) log transform
    ii) takes zscore wrt baseline
    iii) average across sources within the same area
    '''

    # Dimensions
    n_time_points, n_src, n_trials = np.array(data).shape

    # Take z-score of event related data with respect to baseline activity
    z_value = z_score(data, baseline)

    # Extract power time courses and sort them for each parcel (area in MarsAtlas)
    power_sources = area_activity(z_value, atlas)

    # Take average time course for each parcel across sources within an area (n_epochs, n_areas, n_times)
    dims = np.array(power_sources).shape
    power_atlas = np.zeros((n_trials, np.prod(dims), n_time_points))
    narea = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            power_singlearea = power_sources[i][j]
            power_singlearea = np.mean(power_singlearea, axis=0)
            power_atlas[:, narea, :] = power_singlearea
            narea += 1

    # Get names and lobes of areas
    names = []
    lobes = []
    for hemi in ['lh', 'rh']:
        for i in range(len(atlas[hemi])):
            names.append(atlas['lh'][i].name + '_' + hemi)
            lobes.append(atlas['lh'][i].lobe + '_' + hemi)

    # Transform to list
    names = np.array(names).T
    names = names.tolist()
    lobes = np.array(lobes).T
    lobes = lobes.tolist()

    return power_atlas, names, lobes



def z_score(data, baseline):
    """ z-score of source power wrt baseline period (noise)

    Parameters
    ----------
    data : instance of SourceEstimate | array
        The studied data
    baseline : instance of SourceEstimate | array
        The baseline

    Returns
    -------
    z_value : array
        The data transformed to z-value

    -------
    Author : Andrea Brovelli and Alexandre Fabre
    """
    # Dimensions
    n_time_points, n_src, n_trials = np.array(data).shape

    # Take log to make data approximately Gaussian
    data_log = np.log(data)
    baseline_log = np.log(baseline)

    # Mean and std of baseline
    mean = baseline_log.mean(axis=0)
    std = baseline_log.std(axis=0)

    # Take z-score wrt baseline
    z_value = np.zeros((n_src, n_trials, n_time_points))
    for i in range(n_time_points):
        # Compute z-score
        value = (data_log[i] - mean) / std
        # Store
        z_value[..., i] = value

    return z_value



def area_activity(data, obj):
    """ Mean activity for each atlas area

    Parameters
    ----------
    data : array
        Data activities
    obj : (list | dict) of array | (list | dict) of object with index_pack_src attribute
        Allows to select sources to build regions

    Returns
    -------
    fwd : instance of Forward
    -------
    Author : Andrea Brovelli and Alexandre Fabre
    """
 
    obj_dict = create_param_dict(obj)
    for key in obj_dict:
        if hasattr(obj_dict[key], 'index_pack_src'):
            obj[key] = list(map( lambda x : x.index_pack_src, obj_dict[key]))
        else:
            obj[key] = obj_dict[key]

    # src number across each hemisphere
    nb_src = [0, 0]
    i = -1
    for hemi in ['lh', 'rh']:
        i += 1
        for v in obj[hemi]:
            if v.index_pack_src is not None:
                nb_src[i] += len(v.index_pack_src)


    # Organize in hemispheres
    regions = []
    start = 0
    i = -1
    for hemi in ['lh', 'rh']:
        i += 1
        values = data[start:start+nb_src[i]]
        start += nb_src[i]
        regions.append([])
        for region in obj[hemi]:
            if region.index_pack_src is not None:
                # Power values
                select_sources = values[region.index_pack_src]
                regions[i].append(select_sources)

    return regions



if __name__=="__main__":
    pass
