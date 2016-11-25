#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

import mne

import numpy as np

from scipy.stats.mstats import zscore

    

def preprocessing(subject, pdf_name, config_name, head_shape_name,
                  data_event= 'RESPONSE', noise_event='TRIGGER',
                  data_id= None, noise_id= None):

    raw= mne.io.read_raw_bti(pdf_name,config_name, head_shape_name,
                             rename_channels=False, sort_by_ch_name=True,
                             preload=True)


    # Mark bad channels
    raw.info['bads'] = ['A151', 'A125']

    # Keep only MEG data
    meg = raw.pick_types(meg=True, copy=True)

    # Triggers
    events = mne.find_events(raw, stim_channel= noise_event)

    # Responses
    resps = mne.find_events(raw, stim_channel= data_event)

    # Define S-A codes
    S_id = [522, 532, 542]
    A_id = dict(A1=128, A2=256, A3=512, A4=1024, A5=2048)

    # get stimuli events
    events = events[np.logical_or.reduce([events[:,-1] == _id for _id in S_id])]

    S_size = events.shape[0]
    A_size = resps.shape[0]

    if A_size > S_size:
        raise ValueError('there are  more action than stimuli events')

    i=0
    while i < S_size and i < A_size:
        if i < S_size-1:
            # print(resps[i,0] - events[i,0])
            if resps[i,0] > events[i+1,0]:
                events = np.delete(events, i, axis=0)
                S_size -= 1
#  to exlude response greater that trigger more than 1000 ms after stimulus              
#            elif resps[i,0] - events[i,0] > 1000:
#                events = np.delete(events, i, axis=0)
#                resps = np.delete(resps, i, axis=0)
#                A_size -= 1
#                S_size -= 1
            else:
                i +=1
        else:
            i += 1
            
    if S_size != A_size:
        events = events[:A_size]
    

    # MEG and SEEG signals were first down-sampled to 1 kHz,
    # low-pass filtered to 250 Hz
    raw._data = mne.filter.low_pass_filter(raw._data, Fs=1000, Fp=250,
                                           picks=None, method = 'fft')

    # MEG data
    meg = raw.pick_types(meg=True, copy=True)

    # epochs aligned on finger movement (i.e., button press),
    # also performed on stimulus onset

    # create epochs
    epochs_stim = mne.Epochs(meg, events, event_id=S_id, tmin=-0.7,
                             tmax=0.2, baseline=(-0.7, 0.2),
                             preload=True)
    
    epochs_act = mne.Epochs(meg, resps, event_id=A_id, tmin=-1.7,
                            tmax=1.7, baseline=(None, None),
                            preload=True)

    # n_epochs * n_channels * datas
    epoch_datas =epochs_act.get_data()


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
    epochs_stim.save('{0}//preprocessing/{0}-stim-epo.fif'.format(subject))
    epochs_act.save('{0}//preprocessing/{0}-act-epo.fif'.format(subject))
    raw.save('{0}//preprocessing/{0}-raw.fif'.format(subject), overwrite=True)

    return epochs_act, epochs_stim, raw

if __name__=="__main__":
    pass



