#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>
#         Andrea Brovelli

from brain import get_brain
from preprocessing import preprocessing
from source_analysis import (forward_model,
                             get_epochs_dics,
                             area_activity,
                             z_score)
import changepath
import mne
from mne.viz import circular_layout, plot_connectivity_circle
import numpy as np
import matplotlib.pyplot as plt
from data import (read_serialize,
                  Master,
                  serialize,
                  create_trans)


# from connectivity import (linear_corr,
#                           covGC_time)

def test(subject='subject_04', session='1'):
    """
    Pipeline for importing ...
    :param subject:
    :param session:
    :return:
    """

    figure = None

    # Project 's directory
    subjects_dir = '/hpc/comco/brovelli.a/db_mne/meg_te/'
    # Functional data
    pdf_name = subjects_dir + '{0}/raw/{1}/c,rfDC'.format(subject, session)
    config_name = subjects_dir + '{0}/raw/{1}/config'.format(subject, session)
    head_shape_name = subjects_dir + '{0}/raw/{1}/hs_file'.format(subject, session)
    # White matter meshes
    fname_surf_L = subjects_dir + '{0}/surf/{0}_Lwhite.gii'.format(subject)
    fname_surf_R = subjects_dir + '{0}/surf/{0}_Rwhite.gii'.format(subject)
    # MarsAtlas surface parcellation from Brainvisa
    fname_tex_L = subjects_dir + '{0}/tex/{0}_Lwhite_parcels_marsAtlas.gii'.format(subject)
    fname_tex_R = subjects_dir + '{0}/tex/{0}_Rwhite_parcels_marsAtlas.gii'.format(subject)
    # MarsAtas files
    fname_atlas = subjects_dir + 'label/MarsAtlas_BV_2015.txt'  # Labelling xls file
    fname_color = subjects_dir + 'label/MarsAtlas.ima'          # Color palette
    # MarsAtlas volume parcellation from Brainvisa
    fname_vol = subjects_dir + '{0}/vol/{0}_parcellation.nii.gz'.format(subject)
    name_lobe_vol = ['Subcortical']
    # File to align coordinate frames meg2mri computed using mne analyze (interactive gui)
    trans_fname = subjects_dir + '{0}/trans/{0}_meg2mri-trans.fif'.format(subject)
    # Referential file list (standard files to be added)
    file_trans_ref = subjects_dir + 'referential/referential.txt'
    ref = subjects_dir + '{0}/ref/{0}-trans.trm'.format(subject)

    # Create .trm file transformation from BrainVisa to FreeSurfer needed for brain.py function
    create_trans(subject, file_trans_ref, ref)

    # Create the brain object with all structures (surface and volume) and MarsAtlas
    brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      0, fname_vol, name_lobe_vol, ref, fname_atlas, fname_color)

    # Place sources onto cortical surface and subcortical volumes
    src = brain.get_sources(space=5, distance='euclidean')

    # Load raw MEG 4D BTI data and do basic preprocessing
    epochs_s, epochs_a, epochs_r, raw = preprocessing(subject, session, pdf_name, config_name, head_shape_name)

    # Initialise
    # bases = Master(areas=brain)
    # serialize(bases, 'serialize/{0}-brain.pickle'.format(subject))

    # brain = read_serialize('serialize/S4-brain.pickle').areas

    # epochs_stim= mne.read_epochs('{0}//preprocessing/{0}-stim-epo.fif'.format(subject))
    # epochs_act= mne.read_epochs('{0}//preprocessing/{0}-act-epo.fif'.format(subject))
    # raw= mne.io.read_raw_fif('{0}//preprocessing/{0}-raw.fif'.format(subject))

    # Forward model for cortical sources
    # fwd = forward_model(subject, raw, trans_fname, src[1], subjects_dir=subjects_dir, force_fixed=False, name='singleshell-cx')
    # Forward model for subcortical sources
    fwd = forward_model(subject, raw, trans_fname, src[0], subjects_dir, force_fixed=True, name='singleshell-sc')

    # fwd= mne.read_forward_solution('S4/fwd/S4-fwd-surf.fif', force_fixed=True)

    # TF parameters
    # High-gamma activity (HGA) parameters
    fmin = 88
    fmax = 92
    mt_bandwidth = 60
    # Time parameters
    win_lengths = 0.15
    tstep = 0.01
    fs = 1 / tstep  # Sampling rate of HGA
    lp = 50  # Low pass filter of HGA
    t_data = [-1.5, 1.5]
    t_noise = [-0.5, -0.1]

    activity = []
    for epochs, time in zip([epochs_a, epochs_s], [t_data, t_noise]):
        tmin, tmax = time

        stc = get_epochs_dics(epochs, fwd, tmin=tmin, tmax=tmax, tstep=tstep,
                              win_lengths=win_lengths, mode='multitaper',
                              fmin=fmin, fmax=fmax, fsum=False, mt_bandwidth=mt_bandwidth,
                              mt_adaptive=False, on_epochs=False, avg_tapers=False)

        data = list(map(lambda x: x.data, stc))
        log = np.log(data)
        proc_data = mne.filter.low_pass_filter(log, Fs=fs, Fp=lp)
        activity.append(proc_data)

    data, noise = activity

    # save and get data
    bases = Master(areas=data)
    serialize(bases, 'serialize/{0}-activity-data.pickle'.format(subject))

    bases = Master(areas=noise)
    serialize(bases, 'serialize/{0}-activity-noise.pickle'.format(subject))

    data = read_serialize('serialize/{0}-activity-data.pickle'.format('S4')).areas

    noise = read_serialize('serialize/{0}-activity-noise.pickle'.format('S4')).areas

    z_value = z_score(data, noise)

    # n_hemisphere, n_regions, n_time_points, n_trials
    power_area = area_activity(z_value, brain.surfaces)

    trials_map = []

    for i, hemi in enumerate(power_area):
        trials_map.append([])
        for region in hemi:
            trials_map[i].append(np.mean(region, axis=0))

    tmin, tmax = t_data

    # window size : 200 ms 
    corr = linear_corr(trials_map[0], 1, 140, 1, 20)

    # center on action, trial --> number 6
    f_corr = corr[..., -1, 5]

    # get area names
    lh_labels = [surface.name for surface in brain.surfaces['lh']]

    # get color names
    node_colors = [surface.color for surface in brain.surfaces['lh']]

    plot_connectivity_circle(f_corr, lh_labels, n_lines=10, vmin=0.5,
                             node_colors=node_colors,
                             facecolor='blue', textcolor='black')

    #    # too long
    #    trials_mean=np.mean(z_value, axis=1)
    #    GC, pairs= covGC_time(trials_mean, 3, 2, 2)

    # left hemisphere
    parcel = power_area[0]

    n_time_steps = len(data)
    tmin, tmax = t_data
    n_trials = len(data[-1])

    # areas --> visual and motor cortex
    parcel = [parcel[0], parcel[22]]

    # display power
    for (i, region) in enumerate(parcel):
        region = np.mean(region, axis=0)
        n_trials, n_times = region.shape
        plt.imshow(region, cmap=None, aspect='auto', extent=[tmin, tmax, n_trials, 0])
        plt.colorbar()
        # plt.title('parcel %d'% (i+1))
        plt.show()
        mean = region.mean(axis=0)
        max_plot = tmin + n_times * tstep
        x = np.arange(tmin, max_plot, tstep)
        plt.plot(x, mean)
        plt.show()


if __name__ == '__main__':
    test()
