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

def do_preprocessing(subject='subject_04', session='1'):
    """

    :param subject:
    :param session:
    :return:
    """
    #-------------------------------------------------------------------------------------------------------------------
    # Project's directory
    #-------------------------------------------------------------------------------------------------------------------
    subjects_dir = '/hpc/comco/brovelli.a/db_mne/meg_te/'

    #-------------------------------------------------------------------------------------------------------------------
    # Functional data
    #-------------------------------------------------------------------------------------------------------------------
    fname_bti = subjects_dir + '{0}/raw/{1}/c,rfDC'.format(subject, session)
    fname_config = subjects_dir + '{0}/raw/{1}/config'.format(subject, session)
    fname_hs = subjects_dir + '{0}/raw/{1}/hs_file'.format(subject, session)

    # -------------------------------------------------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------------------------------------------------
    # Load raw MEG 4D BTI data and do basic preprocessing
    epochs_s, epochs_a, epochs_r, raw = preprocessing(subject, session, fname_bti, fname_config, fname_hs)


def create_source_model(subject='subject_04', session='1'):
    """
    Pipeline for importing ...
    :param subject:
    :param session:
    :return:
    """

    figure = None
    #-------------------------------------------------------------------------------------------------------------------
    # Project's directory
    #-------------------------------------------------------------------------------------------------------------------
    subjects_dir = '/hpc/comco/brovelli.a/db_mne/meg_te/'

    #-------------------------------------------------------------------------------------------------------------------
    # Functional data
    #-------------------------------------------------------------------------------------------------------------------
    # Raw fif fname
    fname_raw = subjects_dir + '{0}/raw/{1}/{0}_raw.fif'.format(subject, session)

    #-------------------------------------------------------------------------------------------------------------------
    # Anatomical data
    #-------------------------------------------------------------------------------------------------------------------
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
    fname_trans = subjects_dir + '{0}/trans/{0}_meg2mri-trans.fif'.format(subject)
    # Referential file list (standard files to be added)
    fname_trans_ref = subjects_dir + 'referential/referential.txt'
    fname_trans_out = subjects_dir + '{0}/ref/{0}-trans.trm'.format(subject)

    #-------------------------------------------------------------------------------------------------------------------
    # Setting up the source space from BrainVISA results
    #-------------------------------------------------------------------------------------------------------------------
    # http://martinos.org/mne/stable/manual/cookbook.html#source-localization
    # Create .trm file transformation from BrainVisa to FreeSurfer needed for brain.py function
    create_trans(subject, fname_trans_ref, fname_trans_out)

    # Create the brain object with all structures (surface and volume) and MarsAtlas from BrainVISA
    brain = get_brain(subject, fname_surf_L, fname_surf_R, fname_tex_L, fname_tex_R,
                      0, fname_vol, name_lobe_vol, fname_trans_out, fname_atlas, fname_color)

    # Create source space and put dipoles on the white matter surface and on a grid in subcortical volumes
    src = brain.get_sources(space=5, distance='euclidean')

    #-------------------------------------------------------------------------------------------------------------------
    # Computing the single-shell forward solution using the boundary-element model (BEM) meshes from Freesurfer
    #-------------------------------------------------------------------------------------------------------------------
    # Load *.fif raw data to collect information for forward model in raw.info
    raw = mne.io.read_raw_fif(fname_raw)
    # Forward model for cortical sources (fixed orientation)
    fwd = forward_model(subject, raw, fname_trans, src[0], subjects_dir, force_fixed=True, name='singleshell-cortex')
    # Forward model for subcortical sources (no fixed orientation)
    fwd = forward_model(subject, raw, fname_trans, src[1], subjects_dir, force_fixed=False, name='singleshell-subcort')


    # Visualize BEM surfaces
    mne.viz.plot_bem(subject, subjects_dir, brain_surfaces='white', orientation='coronal')

    # Visualize the coregistration
    info = mne.io.read_info(fname_raw)
    mne.viz.plot_trans(info, fname_trans, subject=subject, dig=True, meg_sensors=True, head='outer_skin', subjects_dir=subjects_dir)
    mne.viz.plot_trans(info, fname_trans, subject=subject, dig=[], meg_sensors=[], head='outer_skin', brain='white', subjects_dir=subjects_dir)

    # Visualize sources ontop
    cx_lh = src[0][0]
    cx_lh = mne.SourceSpaces(src[0][0])
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, brain_surfaces='white', slices=np.linspace(100,150,10), orientation='coronal')
    mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, brain_surfaces='white', slices=np.linspace(110,150,10), orientation='coronal')

    # Cortical sources left hemi
    cx_lh = src[0][0]
    # Source coordinates, faces and normals
    coords = np.array(cx_lh[0]['rr'])
    x1, y1, z1 = coords.T
    x, y, z = coords[cx_lh[0]['inuse'].astype(bool)].T
    faces = cx_lh[0]['tris']
    normals = cx_lh[0]['nn']
    # Create mesh
    mesh = mlab.pipeline.triangular_mesh_source(x1, y1, z1, faces)
    mesh.data.point_data.normals = normals
    # Plot
    mlab.figure(1, bgcolor=(0, 0, 0))
    mlab.pipeline.surface(mesh, color=3 * (0.7,))
    mlab.points3d(x, y, z, color=(1, 1, 0), scale_factor=0.002)


    # Subcortical sources left hemi
    sub_lh = src[1][0]
    # Source coordinates, faces and normals
    coords = np.array(sub_lh[0]['rr'])
    x, y, z = coords[sub_lh[0]['inuse'].astype(bool)].T
    mlab.points3d(x, y, z, color=(1, 1, 0), scale_factor=0.001)


    # Initialise
    # bases = Master(areas=brain)
    # serialize(bases, 'serialize/{0}-brain.pickle'.format(subject))

    # brain = read_serialize('serialize/S4-brain.pickle').areas

    # epochs_stim= mne.read_epochs('{0}//preprocessing/{0}-stim-epo.fif'.format(subject))
    # epochs_act= mne.read_epochs('{0}//preprocessing/{0}-act-epo.fif'.format(subject))
    # raw= mne.io.read_raw_fif('{0}//preprocessing/{0}-raw.fif'.format(subject))


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
    # do_preprocessing()
    create_source_model()
