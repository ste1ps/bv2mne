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
import matplotlib.pyplot as plt
from scipy.io import loadmat
# from mne.viz import circular_layout, plot_connectivity_circle
# from connectivity import (linear_corr,
#                           covGC_time)

# ----------------------------------------------------------------------------------------------------------------------
# Subject directoy settings
# ----------------------------------------------------------------------------------------------------------------------
Subjects_Dir = '/hpc/comco/basanisi.r/Databases/db_mne/meg_te/'
Subject = 'subject_01'
Session = '2'
Event   = 'act'

print('Subject = ' + str(Subject))
print('Session = ' + str(Session))
print('Event   = ' + str(Event))
print('-------------------------')

def create_marsatlas_montage(subjects_dir=Subjects_Dir):

    # ------------------------------------------------------------------------------------------------------------------
    # MarsAtlas Montage and 2D Layout
    # ------------------------------------------------------------------------------------------------------------------
    # Import matlab data
    fname_atlas = subjects_dir + 'marsatlas/MarsAtlas.mat'

    # Labels for each area
    mat = loadmat(fname_atlas)
    names = mat['names']
    area_names = [str(''.join(letter)) for letter_array in names for letter in letter_array]

    # 3D positions
    pos = mat['pos_3D']
    pos = pos[:,[1,0,2]]
    pos[:,0] = -pos[:,0]

    # Create dictionary
    dig_ch_pos = dict(zip(area_names, pos))

    # Create montage
    mon = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)

    # Create measurement info
    info = mne.create_info(area_names, 1000., 'seeg', montage=mon)

    # Create Layout
    xy = mat['pos_2D']
    layout = mne.channels.generate_2d_layout(xy, ch_names=area_names)

    return mon, names, layout



def plot_results(subjects_dir=Subjects_Dir, subject=Subject, session=Session, event=Event):


    # ------------------------------------------------------------------------------------------------------------------
    # Set montage and layout
    # ------------------------------------------------------------------------------------------------------------------
    # Load montage and layout for MarsAtlas
    mon, names, lay = create_marsatlas_montage(subjects_dir=Subjects_Dir)

    # ------------------------------------------------------------------------------------------------------------------
    # Plot HGA
    # ------------------------------------------------------------------------------------------------------------------
    # Filename for single-trial HGA
    fname_hga = subjects_dir + '{0}/hga/{1}/{0}_{2}_hga-epo.fif'.format(subject, session, event)
    hga = mne.read_epochs(fname_hga)

    # Set montage
    hga.set_montage(mon)

    # Check montage
    hga.plot_sensors()
    hga.average().plot_topo()

    # ------------------------------------------------------------------------------------------------------------------
    # Test stats
    # ------------------------------------------------------------------------------------------------------------------

    # Merge events
    hga.events.merge_events(events, [1, 2], 12, replace_events=True)


    # Plot mean HGA
    hga.average().plot_topo(layout=lay)

    hga_avg = hga.average()
    mne.viz.plot_evoked_topo(hga_avg, layout=lay, title='Average HGA')


    # Plotting HGA from all areas of cortex left hemi
    hga.average().plot_image(picks=np.arange(0, 41, 1), units='z-score', scalings=1, titles='HGA - lh', cmap='jet')
    plt.yticks(np.linspace(0, 40, 41), hga.ch_names[0:41], rotation='horizontal', fontsize=8)
    plt.grid()

    # Plotting HGA from all areas of cortex right hemi
    hga.average().plot_image(picks=np.arange(41, 82, 1), units='z-score', scalings=1, titles='HGA - lr', cmap='jet')
    plt.yticks(np.linspace(0, 40, 41), hga.ch_names[41:82], rotation='horizontal', fontsize=8)
    plt.grid()

    # Test
    hga.plot_sensors()

    # Single area Mdl
    hga.plot_image(picks=22, units='z-score', scalings=1, cmap='jet')

    brain.set_index('volume')

    brain.show_sources(src[1], hemi='lh', lobe=['Subcortical'], name=['Thal'], sphere_color=(0.7, 0.7, 0.7))
    brain.show_sources(src[1], hemi='lh', lobe=['Subcortical'], name=['Thal'], opacity=0.1)

    brain.set_index(index='surface')
    brain.show_sources(src[0], hemi='lh', lobe=['Frontal'], name=['Insula'], sphere_color=(0.7, 0.7, 0.7), opacity=1)


    # mne.viz.plot_sensors(info, show_names=True)

    # mne.viz.plot_layout(layout)


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
    plot_results()