#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import numpy as np

import os

import mne
from mne.source_space import SourceSpaces
# from mne.time_frequency.csd import csd_epochs
from csd import csd_epochs
from mne.source_estimate import SourceEstimate
# from mne.beamformer import dics_source_power_bis
from _dics import dics_source_power_bis
from mne import make_forward_solution
from mne.connectivity.spectral import (_epoch_spectral_connectivity,
                                       spectral_connectivity)

from data import create_param_dict

from preprocessing import preprocessing

from data import read_serialize


def forward_model(subject, raw, trans_fname, src, subjects_dir, force_fixed=True, name= 'surf'):
    """construct forward model

    Parameters
    ----------
    subject : str
        The name of subject
    raw : instance of rawBTI
        functionnal data
    trans_fname : str
        The filename of transformation matrix
    src : instance of SourceSpaces | list
        Sources of each interest hemisphere
    subjects_dir : str
        The subjects directory
    force_fiwed: Boolean
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
    fname_bem_model = '{0}/bem/{0}-{1}-bem.fif'.format(subject, name)
    fname_bem_sol = '{0}/bem/{0}-{1}-bem-sol.fif'.format(subject, name)
    fname_fwd = '{0}/fwd/{0}-{1}-fwd.fif'.format(subject, name)

    #make bem model
    model = mne.make_bem_model(subject, subjects_dir='.')
    mne.write_bem_surfaces(fname_bem_model, model)

    #make bem solution
    bem_sol = mne.make_bem_solution(model)
    mne.write_bem_solution(fname_bem_sol, bem_sol)

    # bem_sol=mne.read_bem_solution(fname_bem_sol)

    if len(src) == 2:
            # gather sources each the two hemispheres
            lh_src, rh_src = src
            src= lh_src + rh_src
    
    fwd = make_forward_solution(raw.info, trans_fname, src, bem_sol, fname=fname_fwd,
                                mindist=0.0, overwrite=True)
    if force_fixed:
        # avoid the code rewriting
        fwd = mne.read_forward_solution(fname_fwd, force_fixed=True)
    
    return fwd


def get_epochs_dics(epochs, fwd, tmin=None, tmax=None, tstep=None, win_lengths=None, mode='multitaper',
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
    

    if tmin is None:
        tmin = epochs.times[0]+step/2+0.01
    if tmax is None:
        tmax = epochs.times[-1]-step/2-0.01
    if tstep is None:
        tstep = tmin - tmax
    if win_lengths is None:
        win_lengths = tmin - tmax

    # Multiplying by 1e3 to avoid numerical issues
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))
    
    stc=[]
    for i_time in range(n_time_steps):

        win_tmin = tmin + i_time * tstep
        win_tmax = win_tmin + win_lengths

        avg_csds = None
        
        print('window : {0} to {1}'.format(win_tmin, win_tmax))

        #compute csds
        csds = csd_epochs(epochs, mode=mode, fmin=fmin, fmax=fmax, tmin=win_tmin,
                          tmax=win_tmax, fsum=fsum, n_fft=n_fft,
                          mt_bandwidth=mt_bandwidth, mt_adaptive= mt_adaptive,
                          mt_low_bias=mt_low_bias, projs=projs, verbose=verbose,
                          avg_tapers=avg_tapers, on_epochs=on_epochs)

        if len(csds[0].data.shape)>2:
            # compute average csds to compute matrix filter of DICS
            avg_csds = csd_epochs(epochs, mode='multitaper', fmin=fmin, fmax=fmax, tmin=win_tmin,
                                  tmax=win_tmax, fsum=fsum, n_fft=n_fft, mt_bandwidth=mt_bandwidth,
                                  mt_adaptive= mt_adaptive,  mt_low_bias=mt_low_bias, projs=projs,
                                  verbose=verbose, avg_tapers=True, on_epochs=True)
            
        #compute DICS
        cour_stc = dics_source_power_bis(epochs.info, fwd, csds, avg_csds).copy()
        stc.append(cour_stc)

    return stc


def z_score(data, noise):
    """construct forward model

    Parameters
    ----------
    data : instance of SourceEstimate | array
        The studied data
    noise : instance of SourceEstimate | array
        The baseline

    Returns
    -------
    z_value : array
        The data z-value on baseline

    -------
    Author : Alexandre Fabre
    """
    for stc in [data, noise]:
        if isinstance(stc, SourceEstimate):
            stc = np.array(list(map(lambda x : x.data, stc)))
        else:
            try:
                stc = np.array(stc)
            except TypeError:
                print('stc must be SourceEstimate | list | array-like')

    # compute mean and std on baseline
    mean = noise.mean(axis=0)
    std = noise.std(axis=0)
    
    n_time_points, n_src, n_trials= data.shape

    z_value = np.zeros((n_src, n_trials, n_time_points))
    
    for i in range(n_time_points):
        cour_data = data[i]
        
        # compute z-score on data
        value = (cour_data-mean)/std
        z_value[..., i] = value

    return z_value

    

def area_activity(data, obj):
    """construct forward model

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
    Author : Alexandre Fabre
    """
 
    obj_dict = create_param_dict(obj)
    for key in obj_dict:
        if hasattr(obj_dict[key], 'index_pack_src'):
            obj[key] = list(map( lambda x : x.index_pack_src, obj_dict[key]))
        else:
            obj[key] = obj_dict[key]
            
    intervalle_min = 0
    remains = 0

    # src number across each hemisphere
    nb_src = [0,0]
    for i, hemi in  enumerate(obj):
        for v in obj[hemi]:
            if v.index_pack_src is not None:
                nb_src[i] += len(v.index_pack_src)

    regions = []
    start = 0
    for i, hemi in enumerate(obj):
        values = data[start:start+nb_src[i]]
        start += nb_src[i]
        regions.append([])
        for region in obj[hemi]:
            if region.index_pack_src is not None:
                select_sources = values[region.index_pack_src]
                regions[i].append(select_sources)
                
    return regions



if __name__=="__main__":
    pass
