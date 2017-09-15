#!/usr/bin/env python

# Author: Andrea Brovelli <andrea.brovelli@univ-amu.fr>
#         Ruggero Basanisi <ruggero.basanisi@gmail.com>
#         Alexandre Fabre <alexandre.fabre22@gmail.com>


import sys
# Remove competing MNE versions
sys.path.remove('/home/brovelli.a/.local/lib/python2.7/site-packages/mne-0.11.0-py2.7.egg')
sys.path.remove('/hpc/soft/lib/python2.7/site-packages/mne-0.12.0-py2.7.egg')
sys.path.remove('/hpc/comco/anaconda2/lib/python2.7/site-packages/mne-0.14.dev0-py2.7.egg')
sys.path.append('/home/brovelli.a/PycharmProjects/mne-python')
# Change paths
sys.path.remove('/home/brovelli.a/.local/lib/python2.7/site-packages')
sys.path.remove('/hpc/soft/lib/python2.7/dist-packages')
sys.path.remove('/hpc/soft/lib/python2.7/site-packages')

import mne
from brain import get_brain
from source_analysis import (forward_model,
                             get_epochs_dics,
                             source2atlas)
import pickle
# from mne.viz import circular_layout, plot_connectivity_circle
import numpy as np
from data import (read_serialize,
                  Master,
                  serialize,
                  create_trans)

# from connectivity import (linear_corr,
#                           covGC_time)

# ----------------------------------------------------------------------------------------------------------------------
# Subject directoy settings
# ----------------------------------------------------------------------------------------------------------------------
Subjects_Dir = '/hpc/comco/basanisi.r/Databases/db_mne/meg_te/'
Subject = 'subject_13'
Session = '3'

#Subject = sys.argv[1]
#Session = sys.argv[2]

print(Subject)
print('Session = ' + str(Session))

# ----------------------------------------------------------------------------------------------------------------------


def create_source_model(subjects_dir=Subjects_Dir, subject=Subject):
    '''
    Pipeline for
    i) importing BrainVISA white meshes  for positions and MarsAtlas textures for areas
    ii) create transformation file from BV to head coordinates
    iii) create brain object and source space src with cortical and subcortical dipoles
    iv) possibility to visualize BEM meshes, sources, cortical meshes, etc
    '''

    # -------------------------------------------------------------------------------------------------------------------
    # Anatomical data
    # -------------------------------------------------------------------------------------------------------------------
    # White matter meshes
    fname_surf_L = subjects_dir + '{0}/surf/{0}_Lwhite.gii'.format(subject)
    fname_surf_R = subjects_dir + '{0}/surf/{0}_Rwhite.gii'.format(subject)
    # MarsAtlas surface parcellation from Brainvisa
    fname_tex_L = subjects_dir + '{0}/tex/{0}_Lwhite_parcels_marsAtlas.gii'.format(subject)
    fname_tex_R = subjects_dir + '{0}/tex/{0}_Rwhite_parcels_marsAtlas.gii'.format(subject)
    # MarsAtas files
    fname_atlas = subjects_dir + 'label/MarsAtlas_BV_2015.xls'  # Labelling xls file
    fname_color = subjects_dir + 'label/MarsAtlas.ima'  # Color palette
    # MarsAtlas volume parcellation from Brainvisa
    fname_vol = subjects_dir + '{0}/vol/{0}_parcellation.nii.gz'.format(subject)
    name_lobe_vol = ['Subcortical']
    # Referential file list (standard files to be added)
    fname_trans_ref = subjects_dir + 'referential/referential.txt'
    fname_trans_out = subjects_dir + '{0}/ref/{0}-trans.trm'.format(subject)
    # Brain object file
    fname_brain = subjects_dir + '{0}/src/{0}-brain.pickle'.format(subject)
    # Source space file
    fname_src = subjects_dir + '{0}/src/{0}-src.pickle'.format(subject)

    # -------------------------------------------------------------------------------------------------------------------
    # Setting up the source space from BrainVISA results
    # -------------------------------------------------------------------------------------------------------------------
    # http://martinos.org/mne/stable/manual/cookbook.html#source-localization
    # Create .trm file transformation from BrainVisa to FreeSurfer needed for brain.py function for surface only
    create_trans(subject, fname_trans_ref, fname_trans_out)

    # Create the brain object with all structures (surface and volume) and MarsAtlas from BrainVISA
    brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      0, fname_vol, name_lobe_vol, fname_trans_out, fname_atlas, fname_color)

    # Create source space and put dipoles on the white matter surface and on a grid in subcortical volumes
    src = brain.get_sources(space=5, distance='euclidean')
    # src = brain.get_sources(space=5, distance='dijkstra')

    # Save brian object to file
    serialize(brain, fname_brain)

    # Save source space to file
    serialize(src, fname_src)

    print('[done]')

def compute_singletrial_source_power(subjects_dir=Subjects_Dir, subject=Subject, session=Session, event='action'):
    '''
    Pipeline for the calculation of single trials estimates of power at the source level using MarsAtlas
    '''

    # -------------------------------------------------------------------------------------------------------------------
    # Anatomical data
    # -------------------------------------------------------------------------------------------------------------------
    # File to align coordinate frames meg2mri computed using mne analyze (interactive gui)
    fname_trans = subjects_dir + '{0}/trans/{0}-trans.fif'.format(subject)
    # Brain object file
    fname_brain = subjects_dir + '{0}/src/{0}-brain.pickle'.format(subject)
    # Source space files
    fname_src = subjects_dir + '{0}/src/{0}-src.pickle'.format(subject)
    # Load source space
    pkl_file = open(fname_brain, 'rb')
    brain = pickle.load(pkl_file)
    pkl_file.close()
    # Load source space
    pkl_file = open(fname_src, 'rb')
    src = pickle.load(pkl_file)
    pkl_file.close()

    # -------------------------------------------------------------------------------------------------------------------
    # Functional data
    # -------------------------------------------------------------------------------------------------------------------
    # Epoched event-of-interest data
    fname_event = subjects_dir + '{0}/prep/{1}/{0}_{2}-epo.fif'.format(subject, session, event)
    epochs_event = mne.read_epochs(fname_event)
    # Epoched baseline data
    fname_baseline = subjects_dir + '{0}/prep/{1}/{0}_baseline-epo.fif'.format(subject, session)
    epochs_baseline = mne.read_epochs(fname_baseline)
    # Output filename for source power analysis at the atals level
    fname_power = subjects_dir + '{0}/hga/{1}/{0}_{2}_hga-epo.fif'.format(subject, session, event)

    # -------------------------------------------------------------------------------------------------------------------
    # Functional parameters
    # -------------------------------------------------------------------------------------------------------------------
    # High-gamma activity (HGA) parameters
    fmin = 88
    fmax = 92
    mt_bandwidth = 80
    # Time parameters
    win_lengths = 0.2
    tstep = 0.005
    # Sampling rate of power estimates
    sfreq = 1 / tstep
    # Initial time points of multitaper window
    t_data = [-1.0, 0.8]
    t_bline = [-1.0, -0.2]

    # -------------------------------------------------------------------------------------------------------------------
    # Computing the single-shell forward solution using raw data for each session
    # -------------------------------------------------------------------------------------------------------------------

    for c_s in range(2):
        if c_s == 0:
            # Forward model for cortical sources (fix the source orientation normal to cortical surface)
            fwd = forward_model(subject, epochs_event, fname_trans, src[0], subjects_dir, force_fixed=True,
                                name='singleshell-cortex')
        elif c_s == 1:
            # Forward model for subcortical sources (no fixed orientation)
            fwd = forward_model(subject, epochs_event, fname_trans, src[1], subjects_dir, force_fixed=False,
                                name='singleshell-subcort')

        # Event-related source time course stc of single-trial power estimates (all sources)
        power_event = get_epochs_dics(epochs_event, fwd, tmin=t_data[0], tmax=t_data[1], tstep=tstep,
                                      win_lengths=win_lengths, mode='multitaper',
                                      fmin=fmin, fmax=fmax, fsum=False, mt_bandwidth=mt_bandwidth,
                                      mt_adaptive=False, on_epochs=False, avg_tapers=False, pick_ori=None)

        # Baseline source time course stc of single-trial power estimates (all sources)
        power_baseline = get_epochs_dics(epochs_baseline, fwd, tmin=t_bline[0], tmax=t_bline[1], tstep=tstep,
                                         win_lengths=win_lengths, mode='multitaper',
                                         fmin=fmin, fmax=fmax, fsum=False, mt_bandwidth=mt_bandwidth,
                                         mt_adaptive=False, on_epochs=False, avg_tapers=False, pick_ori=None)

        if c_s == 0:
            # Single-trial power timecourse at atlas level (MarsAtlas) for cortical regions
            power_atlas_cor, area_names, area_lobes = source2atlas(power_event, power_baseline, brain.surfaces)
        elif c_s == 1:
            # Single-trial power timecourse at atlas level (MarsAtlas) for subcortical regions
            power_atlas_sub, area_names, area_lobes = source2atlas(power_event, power_baseline, brain.volumes)

        # Create Epoch class to store power_atlas (can add montage later for MarsAtlas positions)
        info = mne.create_info(ch_names=area_names, ch_types='seeg', sfreq=sfreq)

        # Trials were cut from tmin
        tmin = t_data[0] + win_lengths / 2

        if c_s == 0:
            # Single-trial power at area level (cortical)
            power_atlas_cor = mne.EpochsArray(power_atlas_cor, info, epochs_event.events, tmin, epochs_event.event_id)
        elif c_s == 1:
            # Single-trial power at area level (subcortical)
            power_atlas_sub = mne.EpochsArray(power_atlas_sub, info, epochs_event.events, tmin, epochs_event.event_id)

    # Append of both cortical and subcortical EpochsArray
    power_atlas = power_atlas_cor.add_channels([power_atlas_sub])
    # Save data
    power_atlas.save(fname_power)


def check_results(subjects_dir=Subjects_Dir, subject=Subject, session=Session, event='action'):

    # -------------------------------------------------------------------------------------------------------------------
    # Functional data
    # -------------------------------------------------------------------------------------------------------------------
    # Filename for single-trial HGA
    fname_hga = subjects_dir + '{0}/hga/{1}/{0}_{2}_hga-epo.fif'.format(subject, session, event)
    hga = mne.read_epochs(fname_hga)

    # Plotting HGA from all areas of cortex left hemi
    hga.average().plot_image(picks=np.arange(0, 41, 1), units='z-score', scalings=1, titles='HGA - lh', cmap='interactive')
    plt.yticks(np.linspace(0, 40, 41), hga.ch_names[0:41], rotation='horizontal', fontsize=8)
    plt.grid()

    # Plotting HGA from all areas of cortex right hemi
    hga.average().plot_image(picks=np.arange(41, 82, 1), units='z-score', scalings=1, titles='HGA - lr', cmap='interactive')
    plt.yticks(np.linspace(0, 40, 41), hga.ch_names[41:82], rotation='horizontal', fontsize=8)
    plt.grid()

    # Single area Mdl
    hga.plot_image(picks=22, units='z-score', scalings=1, cmap='interactive')

    brain.set_index('volume')

    brain.show_sources(src[1], hemi='lh', lobe=['Subcortical'], name=['Thal'], sphere_color=(0.7, 0.7, 0.7))
    brain.show_sources(src[1], hemi='lh', lobe=['Subcortical'], name=['Thal'], opacity=0.1)

    brain.set_index(index='surface')
    brain.show_sources(src[0], hemi='lh', lobe=['Frontal'], name=['Insula'], sphere_color=(0.7, 0.7, 0.7), opacity=1)

    # !!!!!
    # Add half time window to timepoints
    # !!!!!

    # Low-pass filter to reduce noise
    # proc_data = mne.filter.low_pass_filter(power_area, Fs=fs, Fp=lp)

    # # Initialise
    # bases = Master(areas=brain)
    # serialize(bases, subjects_dir + '{0}/serialize/{0}-brain.pickle'.format(subject))
    # brain = read_serialize(subjects_dir + '{0}/serialize/{0}-brain.pickle'.format(subject)).areas

    # trials_map = []
    # for i, hemi in enumerate(power_area):
    #     trials_map.append([])
    #     for region in hemi:
    #         trials_map[i].append(np.mean(region, axis=0))
    #
    tmin, tmax = t_data
    #
    # # window size : 200 ms
    # corr = linear_corr(trials_map[0], 1, 140, 1, 20)
    #
    # # center on action, trial --> number 6
    # f_corr = corr[..., -1, 5]
    #
    # # get area names
    # lh_labels = [surface.name for surface in brain.surfaces['lh']]
    #
    # # get color names
    # node_colors = [surface.color for surface in brain.surfaces['lh']]
    #
    # plot_connectivity_circle(f_corr, lh_labels, n_lines=10, vmin=0.5,
    #                          node_colors=node_colors,
    #                          facecolor='blue', textcolor='black')
    #
    # #    # too long
    # #    trials_mean=np.mean(z_value, axis=1)
    # #    GC, pairs= covGC_time(trials_mean, 3, 2, 2)
    #
    # # left hemisphere
    # parcel = power_area[0]
    #
    tmin, tmax = t_data
    Mdl = power_atlas[0][22]
    Mdl = (np.mean(Mdl, axis=0))
    plt.figure()
    plt.imshow(Mdl, cmap=None, interpolation='none', aspect='auto', extent=[tmin + 0.1, tmax + 0.1, 54, 0],
               clim=(-4, 4))
    plt.xlim(-1, 0.7)
    plt.colorbar()

    X = np.zeros((len(data), 6094, 54))
    for i in range(len(data)):
        X[i, :] = data[i]

    L = np.zeros((len(data_log), 6094, 54))
    for i in range(len(data_log)):
        L[i, :] = data_log[i]

    plt.figure()
    plt.imshow(L[:, 1000, :].T, cmap=None, interpolation='none', aspect='auto', extent=[tmin + 0.2, tmax + 0.2, 54, 0])

    tmin, tmax = t_data
    plt.figure()
    plt.imshow(X[:, 1000, :].T, cmap=None, interpolation='none', aspect='auto', extent=[tmin + 0.2, tmax + 0.2, 54, 0])
    plt.xlim(-1, 0.75)
    plt.colorbar()

    N = np.zeros((len(noise), 6094, 54))
    for i in range(len(noise)):
        N[i, :] = noise[i]

    t_data = [-1.1, 0.7]
    tmin, tmax = t_data
    t = np.linspace(tmin, tmax, 36)
    plt.figure()
    plt.plot(t, X[:, 1000, :])

    t_noise = [-1.6, -1.2]
    tmin, tmax = t_noise
    t = np.linspace(tmin, tmax, 7)
    plt.figure()
    plt.plot(t, N[:, 1000, :])

    plt.imshow(N[:, 1000, :].T, cmap=None, interpolation='none', aspect='auto', extent=[tmin + 0.1, tmax + 0.1, 54, 0])
    plt.colorbar()

    plt.xlim(-1, 0.7)
    plt.figure()
    plt.plot(N[:, 1000, :])

    plt.figure()
    plt.imshow(X[:, 1000, :].T, cmap=None, interpolation='none', aspect='auto', extent=[tmin + 0.1, tmax + 0.1, 57, 0])
    plt.xlim(-1, 0.7)

    Mdl = np.reshape(Mdl, (400 * 57, 1))
    fMdl = mne.filter.low_pass_filter(Mdl, Fs=fs, Fp=25)
    fMdl = np.reshape(fMdl, (400, 57)).T
    plt.imshow(Mdl, cmap=None, interpolation='none', aspect='auto', extent=[tmin, tmax, 57, 0])

    Mdl = power_area[0][22]
    m = np.mean(np.mean(Mdl, axis=0), axis=0)
    t = np.linspace(tmin + 0.1, tmax + 0.1, len(m))
    plt.figure()
    plt.plot(t, m)
    plt.xlim(-1, 0.7)

    Mdl = power_area[0][22]
    m = np.mean(np.mean(Mdl, axis=0), axis=0)
    t = np.linspace(tmin, tmax, len(m)) + 0.1
    plt.figure()
    plt.plot(t, m)
    plt.xlim(-1, 0.7)

    Vcm = power_area[0][0]
    m = np.mean(np.mean(Vcm, axis=1), axis=0)
    plt.plot(m)

    tmin, tmax = [-1.0, 1.0]
    m = np.mean(Mdl, axis=0)
    n_trials, n_time_steps = m.shape
    plt.imshow(m, cmap=None, interpolation='none', aspect='auto', extent=[tmin, tmax, n_trials, 0])

    n_time_steps = len(data)
    # tmin, tmax = t_data
    # n_trials = len(data[-1])
    #
    # # areas --> visual and motor cortex
    # parcel = [parcel[0], parcel[22]]
    #
    # # display power
    # for (i, region) in enumerate(parcel):
    #     region = np.mean(region, axis=0)
    #     n_trials, n_times = region.shape
    #     plt.imshow(region, cmap=None, interpolation='none', aspect='auto', extent=[tmin, tmax, n_trials, 0])
    #     plt.colorbar()
    #     # plt.title('parcel %d'% (i+1))
    #     plt.show()
    #     mean = region.mean(axis=0)
    #     max_plot = tmin + n_times * tstep
    #     x = np.arange(tmin, max_plot, tstep)
    #     plt.plot(x, mean)
    #     plt.show()


if __name__ == '__main__':
    # do_preprocessing()
    # create_source_model()
    compute_singletrial_source_power()
    # check_results()