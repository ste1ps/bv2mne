#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import mne
import numpy as np
import matplotlib.pyplot as plt
from GUI_bad_selector import GUI_plot
from scipy.stats.mstats import zscore


def preprocessing_meg_te(subjects_dir, subject, session, pdf_name, config_name, head_shape_name,
                         response_chan_name='RESPONSE', trigger_chan_name='TRIGGER',
                         data_id=None, noise_id=None):

    # Load raw data
    raw = mne.io.read_raw_bti(pdf_name, config_name, head_shape_name,
                              rename_channels=False, sort_by_ch_name=True,
                              preload=True)
    # Mark bad channels
    raw.info['bads'] = ['A151', 'A125']

    # Band pass filter
    raw.filter(1, 250)

    # Nothc filter
    raw.notch_filter(np.arange(50, 151, 50))

    # Save raw data
    raw.save(subjects_dir + '{0}/raw/{1}/{0}_raw.fif'.format(subject, session), overwrite=True)

    # Remode power noise
    # raw.notch_filter(np.arange(50, 251, 50), phase='zero-double')

    # Event data ("trigger" channels)
    events = mne.find_events(raw, stim_channel=trigger_chan_name)

    # Save event file for mne_analyze
    mne.write_events(subjects_dir + '{0}/prep/{1}/{0}-eve.fif'.format(subject, session), events)

    # Responses
    resps = mne.find_events(raw, stim_channel=response_chan_name)

    # Define codes for Stim, Action and Reward
    S_id = [522, 532, 542] # Stim 1, 2, 3 + 512 of photodyode code
    A_id = [128, 256, 512, 1024, 2048] # Thumb, index, middle, ring, little fingers
    R_id = [712, 722, 732] # R: Incorrect, Correct, Late + 512 of photodyode code

    # Get stimuli events
    stims = events[np.logical_or.reduce([events[:,-1] == _id for _id in S_id])]

    # Get response events
    resps = resps[np.logical_or.reduce([resps[:,-1] == _id for _id in A_id])]

    # Get reward events
    rews = events[np.logical_or.reduce([events[:,-1] == _id for _id in R_id])]

    # Check size
    S_size = stims.shape[0]
    A_size = resps.shape[0]
    if A_size > S_size:
        raise ValueError('there are more action than stimuli events')

    # Take only correct and incorrect trials (remove late trials)
    c = rews[:,2]<732
    stims = stims[c]
    resps = resps[c]
    rews = rews[c]

    # MEG data
    meg = raw.pick_types(meg=True)

    # Create epochs for baseline
    epochs_b = mne.Epochs(meg, stims, tmin=-1.5, tmax=0.5)

    # Create epochs on stimulus onset
    epochs_s = mne.Epochs(meg, stims, tmin=-1.0, tmax=1.5)

    # Create epochs on action
    epochs_a = mne.Epochs(meg, resps, tmin=-2.0, tmax=2.0)

    # Create epochs on reward onset
    epochs_r = mne.Epochs(meg, rews, tmin=-0.5, tmax=1.5)
    #
    # # Resample epochs
    # epochs_b = epochs_b.resample(1000)
    # epochs_s = epochs_s.resample(1000)
    # epochs_a = epochs_a.resample(1000)
    # epochs_r = epochs_r.resample(1000)

    # Filter in the high-gamma range
    hga = meg.filter(60, 120)
    hga_epochs_a = mne.Epochs(hga, resps, tmin=-1.5, tmax=1.5)
    # n_epochs * n_channels * datas
    epoch_datas = hga_epochs_a.get_data()
    # RMS
    rms = np.sqrt(np.mean(np.square(epoch_datas), axis=2))
    # USe interactive navigation to zoom in and out to select bad trials and channele
    fig = plt.figure()
    plt.imshow(rms, interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xlabel('Channels')
    plt.ylabel('Trials')
    plt.title('RMS')
    plt.rc('grid', linestyle="-", linewidth=5, color='black')
    plt.grid(which='both')

    artifact_rejection = GUI_plot(rms, subjects_dir, subject)

    # Channels covariances
    # cov = mne.compute_covariance(epochs_a, method='empirical')
    # var = np.diagonal(cov.data)

    # bads = np.where(var>1.5*10-24)[0]

    # Drop epochs with large RMS from Fieldtrip
    # bad_trials=np.array([5, 17, 20, 33, 59])-1
    bad_trials = artifact_rejection[1]
    if bad_trials is not None:
        epochs_b.drop(indices=bad_trials)
        epochs_s.drop(indices=bad_trials)
        epochs_a.drop(indices=bad_trials)
    # epochs_r.drop(indices=bad_trials)
    # Drop channels with large RMS
    # bad_chans=[epochs_a.ch_names[28]]
    # bad_chans = []
    bad_channels = artifact_rejection[0]
    if bad_channels is not None:
        for ar in bad_channels:
            bad_chans = [epochs_a.ch_names[ar]]
            print type(bad_chans)
            epochs_b.info['bads'] = bad_chans
            epochs_s.info['bads'] = bad_chans
            epochs_a.info['bads'] = bad_chans
            # epochs_r.info['bads']=bad_chans

    # save
    epochs_b.save(subjects_dir+'{0}/prep/{1}/{0}_baseline-epo.fif'.format(subject, session))
    epochs_a.save(subjects_dir+'{0}/prep/{1}/{0}_action-epo.fif'.format(subject, session))

    return

if __name__=="__main__":
    pass



