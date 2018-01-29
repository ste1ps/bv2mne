#!/usr/bin/env python

# Author: Andrea Brovelli <andrea.brovelli@univ-amu.fr>
#         Ruggero Basanisi <ruggero.basanisi@gmail.com>
#         Alexandre Fabre <alexandre.fabre22@gmail.com>

# ----------------------------------------------------------------------------------------------------------------------
# Import packages
# ----------------------------------------------------------------------------------------------------------------------
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
import numpy as np
from GUI_bad_selector import GUI_plot
from data import serialize

# ----------------------------------------------------------------------------------------------------------------------
# Subject directoy settings
# ----------------------------------------------------------------------------------------------------------------------
# Names of input raw MEG data
Subjects_Dir_Raw = '/envau/work/comco/brovelli.a/Data/Neurophy/MEG_TE/'
Subject_Raw, Subject = 'S13', 'subject_13'
# Names of output MEG data (dir and fname)
Subjects_Dir = '/hpc/comco/basanisi.r/Databases/db_mne/meg_te/'
Sessions = [ '3', '4', '5', '6']

def do_preprocessing(subjects_dir_raw=Subjects_Dir_Raw, subjects_dir=Subjects_Dir, subject_raw=Subject_Raw, subject=Subject, Sessions=Sessions):
    '''
    Pipeline for the preprocessing:
    i) import raw data
    ii) artefact rejection
    iii) detection of events and epoching
    '''

    for session in Sessions:

        print('Subject = ' + Subject)
        print('Session = ' + str(session))

        # --------------------------------------------------------------------------------------------------------------
        # Raw MEG data
        # --------------------------------------------------------------------------------------------------------------
        fname_bti = subjects_dir_raw + '{0}/{1}/c,rfDC'.format(subject_raw, session)
        fname_config = subjects_dir_raw + '{0}/{1}/config'.format(subject_raw, session)
        fname_hs = subjects_dir_raw + '{0}/{1}/hs_file'.format(subject_raw, session)

        # --------------------------------------------------------------------------------------------------------------
        # Preprocessing data
        # --------------------------------------------------------------------------------------------------------------
        preprocessing_meg_te(subjects_dir, subject, session, fname_bti, fname_config, fname_hs)

    return


def preprocessing_meg_te(subjects_dir, subject, session, pdf_name, config_name, head_shape_name,
                         response_chan_name='RESPONSE', trigger_chan_name='TRIGGER'):
    # ------------------------------------------------------------------------------------------------------------------
    # Preprocessing steps for trial-and-error learning task (arbitrary visuomotor learning)
    # ------------------------------------------------------------------------------------------------------------------
    # Step 1. Load raw data and create events
    # ------------------------------------------------------------------------------------------------------------------
    # Load raw data
    raw = mne.io.read_raw_bti(pdf_name, config_name, head_shape_name,
                              rename_channels=False, sort_by_ch_name=True,
                              preload=True)
    # Mark bad channels
    raw.info['bads'] = ['A151', 'A125']

    # Band pass filter
    raw.filter(1, 250)

    # Nothc filter
    raw.notch_filter(np.arange(50, 151, 50), phase='zero-double')

    # Define codes for Stim, Action and Reward
    S_id = [522, 532, 542] # Stim 10, 20, 30 + 512 of photodyode code
    A_id = [128, 256, 512, 1024, 2048] # Thumb, index, middle, ring, little fingers
    R_id = [712, 722] # R: Incorrect and Correct + 512 of photodyode code (exclude late trials)

    # Find triggers (photodiode) and motor resposnes
    triggers = mne.find_events(raw, stim_channel=trigger_chan_name, min_duration=0.01, shortest_event=2)
    resps = mne.find_events(raw, stim_channel=response_chan_name)

    # Find stimuli, resps and rewards
    stims = triggers[np.logical_or.reduce([triggers[:,-1] == _id for _id in S_id])]
    resps = resps[np.logical_or.reduce([resps[:,-1] == _id for _id in A_id])]
    rews = triggers[np.logical_or.reduce([triggers[:,-1] == _id for _id in R_id])]

   # Concatanate events and sort timestamps
    events = np.concatenate((stims, resps, rews), axis=0)
    ind_sar = np.argsort(events[:,0], axis=0)
    events = events[ind_sar]

    # Find events for correct or incorrect rewards R_id
    r_id = np.array([i for i in range(len(events[:, 2])) if events[i, 2] in R_id])

    # Check that rewards are preceded by actions A_id
    a_id = np.array([i for i in range(len(r_id)) if events[r_id[i]-1, 2] in A_id])

    # Check that actions are preceeded by stimuli S_id
    s_id = np.array([i for i in range(len(a_id)) if events[r_id[a_id[i]]-2, 2] in S_id])

    # Reward index
    r_id = r_id[a_id[s_id]]

    # Create events with correctly executed trials
    e_id = np.concatenate((r_id-2, r_id-1, r_id), axis=0)
    events = events[np.sort(e_id),:]

    # ------------------------------------------------------------------------------------------------------------------
    # Step 2. Label events as representative learning trials
    # ------------------------------------------------------------------------------------------------------------------
    # Create SAR
    s = (events[0:len(events):3, 2]-512)/10 - 1
    a = np.log2(events[1:len(events):3, 2]/128).astype(int)
    r = (events[2:len(events):3, 2]-712)/10

    # Create Labels for representative learning trials
    sar = np.zeros((len(s), 5))
    Q = np.zeros((3, 5))    # Association matrix of tried responses
    R = np.zeros(5)         # Association matrix of rewards
    L = np.zeros(len(s))    # Action-based Labels
    M = np.zeros(len(s))    # Reward-based Labels
    for i in range(len(s)):

        # Create action-based labels for learning
        Q[s[i], a[i]] = 1

        # Number of tried responses nTR
        nTR = np.sum(Q, axis=1)

        # Search phase: if the association has not been found, the label is the number of tried reponses
        if R[s[i]] == 0:
            L[i] = nTR[s[i]]

        # Consolidation phase: if the association is found, the label is the number of correct responses + 5
        if R[s[i]] > 0:
            L[i] = R[s[i]] + 5

        # Update Reward matrix
        R[s[i]] = r[i] + R[s[i]]

        # Create reward-based labels for learning
        # Search phase: if the association has not been found, the label is the number of tried reponses
        if R[s[i]] == 0:
            M[i] = nTR[s[i]]

        # Consolidation phase: if the association is found, the label is the number of correct rewards + 4
        if R[s[i]] > 0:
            M[i] = R[s[i]] + 4

        # Crate SAR + Learning Labels
        sar[i, :] = [s[i]+1, a[i]+1, r[i], L[i], M[i]]

    # Create dictionary task_events
    timestamps = np.reshape(events[:, 0], (len(s), 3))
    task_events = {'stim_time':timestamps[:,0], 'act_time':timestamps[:,2], 'rew_time':timestamps[:,2], 'stim': s, 'act': a, 'rew': r, 'learn_label_a': L.astype(int), 'learn_label_r': M.astype(int)}

    # Print SAR matrix to check consistency
    print(sar)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 3. Artifact rejection
    # ------------------------------------------------------------------------------------------------------------------
    # MEG data in high-gamma range
    meg = raw.pick_types(meg=True)

    # High-gamma activity for artifact rejection
    hga = meg.filter(60, 120)

    # Unique index of events that are present in the task (performed actions only)
    A_idx = list(np.unique(events[np.logical_or.reduce([events[:,2] == _id for _id in A_id]), 2]))

    # Unique index of events that are present in the task (observed outomces only)
    R_idx = list(np.unique(events[np.logical_or.reduce([events[:,2] == _id for _id in R_id]), 2]))

    # Compute RMS of high-gamma meg activity
    epochs_a = mne.Epochs(hga, events, event_id=A_idx, tmin=-1.5, tmax=2.5)
    epoch_data = epochs_a.get_data() # n_epochs * n_channels * datas
    rms = np.sqrt(np.mean(np.square(epoch_data), axis=2)) # RMS

    # GUI for artifact rejection. Use interactive navigation to zoom in and out to select bad trials and channel
    artifact_rejection = GUI_plot(rms, subjects_dir, subject)

    # ------------------------------------------------------------------------------------------------------------------
    # Step 4. Create Epochs, remove Artifacted trials and channels and Save results
    # ------------------------------------------------------------------------------------------------------------------
    # Create epochs for baseline
    epochs_b = mne.Epochs(meg, events, event_id=S_id, tmin=-1.5, tmax=0.5)

    # Create epochs on stimulus onset
    epochs_s = mne.Epochs(meg, events, event_id=S_id, tmin=-1, tmax=2)

    # Create epochs on action
    epochs_a = mne.Epochs(meg, events, event_id=A_idx, tmin=-2, tmax=1.5)

    # Create epochs on reward onset
    epochs_r = mne.Epochs(meg, events, event_id=R_idx, tmin=-1, tmax=2)

    # Update thirds column of events according to learning labels
    epochs_b.events[:,2] = task_events['learn_label_a']
    epochs_s.events[:,2] = task_events['learn_label_a']
    epochs_a.events[:,2] = task_events['learn_label_a']
    epochs_r.events[:,2] = task_events['learn_label_r']

    # # Remove artifacted trials
    # bad_trials = artifact_rejection[1]
    # if bad_trials is not None:
    #     # MEG data
    #     epochs_b.drop(indices=bad_trials)
    #     epochs_s.drop(indices=bad_trials)
    #     epochs_a.drop(indices=bad_trials)
    #     epochs_r.drop(indices=bad_trials)

    # Remove artifacted channels
    bad_channels = artifact_rejection[0]
    if bad_channels is not None:
        for ar in bad_channels:
            bad_chans = [epochs_a.ch_names[ar]]
            epochs_b.info['bads'] = bad_chans
            epochs_s.info['bads'] = bad_chans
            epochs_a.info['bads'] = bad_chans
            epochs_r.info['bads'] = bad_chans

    # Update event_id dictionary according to learning labels
    u_id = np.unique(epochs_a.events[:,2])
    s_id = [str(u_id[i]) for i in range(len(u_id))]
    event_id_a = dict(zip(s_id, u_id))
    u_id = np.unique(epochs_r.events[:,2])
    s_id = [str(u_id[i]) for i in range(len(u_id))]
    event_id_r = dict(zip(s_id, u_id))

    # Change even_id in EpocCreate Epochs,hs
    epochs_b.event_id = event_id_a
    epochs_s.event_id = event_id_a
    epochs_a.event_id = event_id_a
    epochs_r.event_id = event_id_r

    print(epochs_a.events)

    # Save epochs
    epochs_b.save(subjects_dir+'{0}/prep/{1}/{0}_bline-epo.fif'.format(subject, session))
    epochs_s.save(subjects_dir+'{0}/prep/{1}/{0}_stim-epo.fif'.format(subject, session))
    epochs_a.save(subjects_dir+'{0}/prep/{1}/{0}_act-epo.fif'.format(subject, session))
    epochs_r.save(subjects_dir+'{0}/prep/{1}/{0}_rew-epo.fif'.format(subject, session))

    # Save all meg events on fif file
    fname_events = subjects_dir + '{0}/prep/{1}/{0}_meg-eve.fif'.format(subject, session)
    mne.write_events(fname_events, events)

    # Update task_events with information of bad MEG trials
    # Remove artifacted trials later
    bad_trials = artifact_rejection[1]
    good_trials = np.setdiff1d(range(0,len(a)), bad_trials)
    task_events.update({'good_trials': good_trials, 'bad_trials': bad_trials})

    # Save all task events on fif file
    fname_events = subjects_dir + '{0}/prep/{1}/{0}_task_events.pkl'.format(subject, session)
    serialize(task_events, fname_events)

    return

def events2sar(events_id):
    # Index to switch from sar to events and vice versa. events at line 8 corresponds to sar[row, column]
    row = np.floor(events_id/3)
    column = events_id - row*3 - 1

    return row, column


if __name__ == '__main__':
    do_preprocessing()