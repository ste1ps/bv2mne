#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import mne
import numpy as np

from scipy.stats.mstats import zscore


def preprocessing(subject, session, pdf_name, config_name, head_shape_name,
                  response_chan_name='RESPONSE', trigger_chan_name='TRIGGER',
                  data_id=None, noise_id=None):

    # Project 's directory
    subjects_dir = '/hpc/comco/brovelli.a/db_mne/meg_te/'

    # Load raw data
    raw = mne.io.read_raw_bti(pdf_name, config_name, head_shape_name,
                              rename_channels=False, sort_by_ch_name=True,
                              preload=True)
    # Mark bad channels
    raw.info['bads'] = ['A151', 'A125']

    # Save raw data
    raw.save(subjects_dir + '{0}/raw/{1}/{0}_raw.fif'.format(subject, session), overwrite=True)

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
        raise ValueError('there are  more action than stimuli events')

    # Take only correct and incorrect trials (remove late trials)
    c = rews[:,2]<732
    stims = stims[c]
    resps = resps[c]
    rews = rews[c]

    # MEG data
    meg = raw.pick_types(meg=True)

    # Create epochs on stimulus onset
    epochs_s = mne.Epochs(meg, stims, tmin=-0.7, tmax=0.2)

    # Create epochs on motor response
    epochs_a = mne.Epochs(meg, resps, tmin=-1.7, tmax=1.7)

    # Create epochs on reward onset
    epochs_r = mne.Epochs(meg, rews, tmin=-1.7, tmax=1.7)

    # n_epochs * n_channels * datas
    epoch_datas = epochs_a.get_data()


#    _zscore = zscore(epoch_datas)
#    _zscore_abs = np.abs(_zscore)

#    # channels covariances
#    cov = mne.compute_covariance(epochs_act, method='empirical')


#    var = np.diagonal(cov.data)
#
#    bads = np.where(var>1.5*10-24)[0]

#    # channels were excluded from further analyses, epochs must be preload
#    epochs_act.drop_channels(ch_names=ch_names, copy=False)

#    # remove bas channels in covariance matrix
#    np.delete(cov.data, bads, axis=0)
#    np.delete(cov.data, bads, axis=1)

    # save
    epochs_s.save(subjects_dir+'{0}/prep/{1}/{0}_stim-epo.fif'.format(subject, session))
    epochs_a.save(subjects_dir+'{0}/prep/{1}/{0}_resp-epo.fif'.format(subject, session))
    epochs_r.save(subjects_dir+'{0}/prep/{1}/{0}_rew-epo.fif'.format(subject, session))

    return epochs_s, epochs_a, epochs_r, raw

if __name__=="__main__":
    pass



