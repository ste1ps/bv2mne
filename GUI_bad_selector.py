# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:48:32 2017

@author: Ruggero Basanisi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from matplotlib.widgets import RadioButtons

def GUI_plot(RMS, subjects_dir, subject):

    # subjects_dir = '/hpc/comco/basanisi.r/Databases/db_mne/meg_te/'
    # subject = 'subject_04'

    # RMS = np.array(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                 [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    #                 [5, 4, 3, 2, 1, 10, 9, 8, 7, 6],
    #                 [6, 7, 8, 9, 10, 1, 2, 3, 4, 5]), dtype=float)

    bad_channels = []
    bad_trials = []

    tot_trials = RMS.shape[0]
    tot_channels = RMS.shape[1]
    channels_average = ((np.sum(RMS, axis=1)) / (tot_channels))  * 10 ** 14
    trials_average = ((np.sum(RMS, axis=0)) / (tot_trials))  * 10 ** 14
    channels_dev_std = np.std(RMS, axis=1)  * 10 ** 14
    trials_dev_std = np.std(RMS, axis=0)  * 10 ** 14
    RMS_zscore = zscore(RMS)
    ch_av_zscore = ((np.sum(RMS_zscore, axis=1)) / (tot_channels))  * 10 ** 14
    tr_av_zscore = ((np.sum(RMS_zscore, axis=0)) / (tot_trials))  * 10 ** 14

    plt.close('all')
    fig = plt.figure()

    ax1 = plt.subplot2grid((4, 6), (2, 0), colspan=4, rowspan=3)
    ax1.set_title('RMS')
    ax1.set_xlabel('Channels')
    ax1.set_ylabel('Trials')

    ax2 = plt.subplot2grid((4, 6), (0, 1), colspan=3, rowspan=2)
    ax2.set_title('Average RMS for Trial')
    ax2.set_xlabel('Trials')
    ax2.set_ylabel('Channels RMS Average *10^14')
    ax2.set_xlim([-1, tot_trials])

    ax3 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=4)
    ax3.set_title('Average of Trials for Channel')
    ax3.set_xlabel('Channels Average in Trials *10^14')
    ax3.set_ylabel('Channels')
    ax3.set_ylim([-1, tot_channels])

    ax4 = plt.subplot2grid((4, 6), (0, 0))

    line1 = ax1.imshow(RMS, interpolation='none', aspect='auto')
    line2, = ax2.plot(range(tot_trials), channels_average, 'o', picker=5)
    line3, = ax3.plot(trials_average, range(tot_channels), 'o', picker=5)
    button = RadioButtons(ax4, ('Average', 'Dev Std', 'Z Score'))

    plt.colorbar(line1, ax=ax1)
    plt.tight_layout()

    def FuncType(label):
        FuncDict = {'Average': (channels_average, trials_average), 'Dev Std': (channels_dev_std, trials_dev_std),
                    'Z Score': (ch_av_zscore, tr_av_zscore)}
        ydata = FuncDict[label]
        line2.set_ydata(ydata[0])
        line3.set_xdata(ydata[1])
        if label == 'Average':
            ax2.set_title('Average RMS for Trial')
            ax2.set_ylabel('Channels RMS Average *10^14')
            ax2.set_ylim([channels_average.min() - 1, channels_average.max() + 1])
            ax3.set_title('Average of Trials for Channel')
            ax3.set_xlabel('Channels Average in Trials *10^14')
            ax3.set_xlim([trials_average.min() - 1, trials_average.max() + 1])
        elif label == 'Dev Std':
            ax2.set_title('Dev. Std. for Trial')
            ax2.set_ylabel('Channels Dev. Std. *10^14')
            ax2.set_ylim([channels_dev_std.min() - 1, channels_dev_std.max() + 1])
            ax3.set_title('Dev. Std. of Trials for Channel')
            ax3.set_xlabel('Channels Dev. Std. in Trials *10^14')
            ax3.set_xlim([trials_dev_std.min() - 1, trials_dev_std.max() + 1])
        elif label == 'Z Score':
            ax2.set_title('Average Z-score RMS for Trial')
            ax2.set_ylabel('Channels Average Z-score *10^14')
            ax2.set_ylim([ch_av_zscore.min() - 1, ch_av_zscore.max() + 1])
            ax3.set_title('Average Z-score of Trials for Channel')
            ax3.set_xlabel('Channels Average Z-score in Trials *10^14')
            ax3.set_xlim([tr_av_zscore.min() - 1, tr_av_zscore.max() + 1])
        plt.draw()

    button.on_clicked(FuncType)

    def onpick(event):
        if ics == 1:
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            if xdata[ind] in bad_trials:
                bad_trials.remove(xdata[ind])
                print 'Trial', xdata[ind], 'removed from rejection list, correspondent value: ', ydata[ind]
                thisline.set_color('b')
            else:
                bad_trials.append(xdata[ind])
                print 'Trial', xdata[ind], 'added to rejection list, correspondent value: ', ydata[ind]
                # thisline._edgecolors[event.ind, :] = (1, 0, 0, 1)
        elif ics == 2:
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            if ydata[ind] in bad_channels:
                bad_channels.remove(ydata[ind])
                print 'Channel', ydata[ind], 'removed from rejection list, correspondent value: ', xdata[ind]
                thisline.set_color('b')
            else:
                bad_channels.append(ydata[ind])
                print 'Channel', ydata[ind], 'added to rejection list, correspondent value: ', xdata[ind]
                thisline.set_color('r')
                # print('onpick points:', points)
        else:
            return

    def sub_click(event_check):
        global ics
        if event_check.inaxes in [ax2]:
            ics = 1
        elif event_check.inaxes in [ax3]:
            ics = 2
        else:
            return

    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('button_press_event', sub_click)
    plt.show()

    channels_ar = open(subjects_dir+'{0}/channels_ar.txt'.format(subject), 'w')
    for car in bad_channels:
        channels_ar.write("%s \n" % car)
    channels_ar.close()
    trials_ar = open(subjects_dir + '{0}/trials_ar.txt'.format(subject), 'w')
    for tar in bad_trials:
        trials_ar.write("%s \n" % tar)
    trials_ar.close()

    if len(bad_trials) == 0 and len(bad_channels) == 0:
        artifact_rejection = (None, None)
    elif len(bad_trials) == 0:
        artifact_rejection = (np.concatenate(bad_channels), None)
    elif len(bad_channels) == 0:
        artifact_rejection = (None, np.concatenate(bad_trials))
    else:
        artifact_rejection = (np.concatenate((bad_channels)), np.concatenate((bad_trials)))
    return artifact_rejection