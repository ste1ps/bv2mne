#!/usr/bin/env python

# Author: Alexandre Fabre <alexandre.fabre22@gmail.com>

from numpy import (cov,
                   log,
                   corrcoef)
                   
from numpy.matlib import repmat
from numpy.linalg import det
import numpy as np

import math as ma

def covGC_time(X, dt, lag, t0):
    """
        [GC, pairs] = covGC_time(X, dt, lag, t0)

        Computes single-trials covariance-based Granger Causality for gaussian variables

        X   = data arranged as sources x timesamples
        dt  = duration of the time window for covariance correlation in samples
        lag = number of samples for the lag within each trial
        t0  = zero time in samples

        GC  = Granger Causality arranged as (number of pairs) x (3 directionalities (pair(:,1)->pair(:,2), pair(:,2)->pair(:,1), instantaneous))
        pairs = indices of sources arranged as number of pairs x 2

        -------------------- Total Granger interdependence ----------------------
        Total Granger interdependence:
        TGI = GC(x,y)
        TGI = sum(GC,2):
        TGI = GC(x->y) + GC(y->x) + GC(x.y)
        TGI = GC(x->y) + GC(y->x) + GC(x.y) = Hycy + Hxcx - Hxxcyy
        This quantity can be defined as the Increment of Total
        Interdependence and it can be calculated from the different of two
        mutual informations as follows

        ----- Relations between Mutual Informarion and conditional entropies ----
        % I(X_i+1,X_i|Y_i+1,Y_i) = H(X_i+1) + H(Y_i+1) - H(X_i+1,Y_i+1)
        Ixxyy   = log(det_xi1) + log(det_yi1) - log(det_xyi1);
        % I(X_i|Y_i) = H(X_i) + H(Y_i) - H(X_i, Y_i)
        Ixy     = log(det_xi) + log(det_yi) - log(det_yxi);
        ITI(np) = Ixxyy - Ixy;

        Reference
        Brovelli A, Chicharro D, Badier JM, Wang H, Jirsa V (2015)

    Copyright of Andrea Brovelli (Jan 2015) - Matlab version -

    """

    X= np.array(X)

    # Data parameters. Size = sources x time points
    nSo, nTi = X.shape

    # Select a single window according to index t0
    ind_t = np.arange(t0-dt+1, t0+1)[:, None]

    # Create indices for all lags
    ind_t = repmat(ind_t, 1, lag+1) - repmat( np.arange(0, lag+1), dt, 1)

    # Pairs between sources
    pairs_x, pairs_y = np.where(np.tril(np.ones(nSo),-1) == 1)

    pairs= np.append(pairs_x[:,None], pairs_y[:,None], axis=1)
    
    nPairs = pairs.shape[0]

    # Init
    GC    = np.zeros((nPairs,3))
    count = 1

    # Normalisation coefficient for gaussian entropy
    C = ma.log(2*ma.pi*ma.exp(1))

    # Loop over number of pairs
    for npair in range(nPairs):

        # Extract data for a given pair of sources
        x = np.squeeze(X[pairs[npair,0],ind_t])
        y = np.squeeze(X[pairs[npair,1],ind_t])
        # Reshape to trials x dt x lags
        x = np.reshape(x, (dt, lag+1))
        y = np.reshape(y, (dt, lag+1))

        # ---------------------------------------------------------------------
        # Conditional Entropies
        # ---------------------------------------------------------------------
        # Hycy: H(Y_i+1|Y_i) = H(Y_i+1) - H(Y_i)
        det_yi1  = det(cov(y))
        det_yi   = det(cov(y[:,1:]))
        #print(det_yi)
        Hycy     = log(det_yi1) - log(det_yi)
        # Hycx: H(Y_i+1|X_i,Y_i) = H(Y_i+1,X_i,Y_i) - H(X_i,Y_i)
        det_yxi1 = det(cov(np.append( y, x[:,1:], axis=1)))
        det_yxi  = det(cov(np.append( y[:,1:], x[:,1:], axis=1)))
        Hycx     = log(det_yxi1) - log(det_yxi)
        # Hxcx: H(X_i+1|X_i) = H(X_i+1) - H(X_i)
        det_xi1  = det(cov(x))
        det_xi   = det(cov(x[:,1:]))
        Hxcx     = log(det_xi1) - log(det_xi)
        # Hxcy: H(X_i+1|X_i,Y_i) = H(X_i+1,X_i,Y_i) - H(X_i,Y_i)
        det_xyi1 = det(cov(np.append( x, y[:,1:], axis=1)))
        Hxcy     = log(det_xyi1) - log(det_yxi)
        # Hxxcyy: H(X_i+1,Y_i+1|X_i,Y_i) = H(X_i+1,Y_i+1,X_i,Y_i) - H(X_i,Y_i)
        det_xyi1 = det(cov(np.append( x, y, axis=1)))
        Hxxcyy   = log(det_xyi1) - log(det_yxi)

        # ---------------------------------------------------------------------
        # Causality measures
        # ---------------------------------------------------------------------
        # GC(pairs[:,0]->pairs[:,1])
        GC[npair,0] = Hycy - Hycx
        # GC(pairs[:,1]->pairs[:,2])
        GC[npair,1] = Hxcx - Hxcy
        # GC[x.y]
        GC[npair,2] = Hycx + Hxcy - Hxxcyy

    return GC, pairs

def linear_corr(data, tmin, tmax, tstep, win_lengths):
    """ matrix correlation --> regions """

    data=np.array(data)
    n_regions, n_trials, n_times= data.shape

    
    n_time_steps = int(((tmax - tmin) * 1e3) // (tstep * 1e3))
    
    #regions * regions * time point * trials
    corr= np.zeros((n_regions, n_regions, n_time_steps, n_trials))
    
    for i_time in range(n_time_steps):

        win_tmin = tmin + i_time * tstep
        win_tmax = win_tmin + win_lengths

        win_data= data[...,win_tmin:win_tmax]

        for i_trial in range(n_trials):
            win_trials= win_data[...,i_trial,:]
            corr[..., i_time, i_trial]= corrcoef(win_trials)

    return corr

if __name__ == '__main__':
    pass



