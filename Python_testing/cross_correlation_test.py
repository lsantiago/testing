#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:10:41 2020

@author: annamaria


Code to test cross-correlation with Python
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import obspy
import math
from scipy import signal
#from obspy.signal.cross_correlation import correlate, xcorr

#%% First case - phase shift

# =============================================================================
# # setting parameters 
# =============================================================================
npts = 500                     #number of points
x = np.linspace(0, 50, npts)    # setting x axis
A = 5                           # amplitude

# =============================================================================
# # defining signals
# =============================================================================

signal_1 = A * np.sin(x/2)
signal_2 = A * np.sin(x/2+math.pi/2)


# =============================================================================
# # cross-correlation
# =============================================================================

cc = np.correlate(signal_1, signal_2, mode = 'full')

# defining the x axis for correlation plot
lags = np.arange(-npts + 1, npts)
#normalizing the amplitude of the cross-correlation
ccor = cc / (npts * signal_1.std() * signal_2.std())

# max argument of the cross corr
maxlag = lags[np.argmax(ccor)]
maxlag_time = x[abs(maxlag)]

# =============================================================================
# # plots
# =============================================================================

fig, axs = plt.subplots(2,1, figsize = (15,7))

# plotting the signls
ax = axs[0]
ax.plot(x, signal_1, label = 'signal 1')
ax.plot(x, signal_2, label = 'singal 2')
ax.set_ylabel('Amplitude')
ax.legend()


# plotting the corss-correlation
ax = axs[1]
ax.plot(lags,ccor)
ax.plot(maxlag, ccor[np.argmax(ccor)],'ro', label = 'max correlation')
ax.set_xlabel('lag of signal 1 relative to signal 2')
ax.set_ylabel('cross-correlation')
ax.set_title(f'max correlation is at lag {maxlag}')
ax.legend()

fig.suptitle('Case #1 - phase shift')


#%% Autocorrelation of random noise

# =============================================================================
# # defining signal
# =============================================================================

signal_3 = np.random.randn(npts)

# =============================================================================
# # auto-correlation
# =============================================================================
ac = np.correlate(signal_3, signal_3, mode = 'full')

# defining the x axis for correlation plot
lags = np.arange(-npts + 1, npts)
#normalizing the amplitude of the cross-correlation
acor = ac / (npts * signal_3.std() * signal_3.std())

# max argument of the cross corr
maxlag = lags[np.argmax(acor)]
maxlag_time = x[abs(maxlag)]



fig, axs = plt.subplots(2,1, figsize = (15,7))
ax = axs[0]
ax.plot(x,signal_3)
ax.set_ylabel('Amplitude')
ax.set_title('Random noise signal')

ax = axs[1]
ax.plot(lags,acor)
ax.plot(lags[np.argmax(acor)], ccor[np.argmax(ccor)],'ro', label = 'max correlation')
ax.set_xlabel('lag of signal 1 relative to signal 2')
ax.set_ylabel('cross-correlation')
ax.set_title(f'max correlation is at lag {maxlag}')
ax.legend()

fig.suptitle('Case #2 - auto-correlation of the random noise')

    


#%% Two recordings from seismic array

# importing signals
data = np.load("/Users/annamaria/PhD/Data/Matera/20200214_Line1_Matera/preprocessed/pre_Line1_25_10_2019_10_51_00_000.npy")


# taking just to first signals to analyze
row_1 = data[:,0]
row_2 = data[:,19]


# numebr of data points
dpt = len(row_1)
# timestep
dt  = 1./250

time = np.linspace(0, dpt, dpt )*dt

# plotting all of the signals to visualize them
fig, ax = plt.subplots(figsize = (15,5))
for i in range(0,20):
    ax.plot(time, data[:,i]+i*0.001)





# =============================================================================
# # cross-correlation
# =============================================================================

cc3 = np.correlate(row_1-row_1.mean(), row_2-row_2.mean(), mode = 'full')

# defining the x axis for correlation plot
lags = np.arange(-row_2.size + 1, row_1.size)*dt
#normalizing the amplitude of the cross-correlation
ccor3 = cc3 / (dpt * row_1.std() * row_2.std())

# max argument of the cross corr
maxlag3 = lags[np.argmax(ccor3)]
#maxlag_time3 = time[abs(maxlag3)]

# =============================================================================
# # plots
# =============================================================================

# setting some plotting parameters for zoom plots

dstack = 1 # plotting time limit
Fs = 250
cor_len = len(ccor3)
ccor3_d = np.concatenate((ccor3[(math.floor(cor_len/2)+1):],ccor3[:(math.floor(cor_len/2))]))
cor_len2 = len(ccor3_d)

ccor3_dt = ccor3[(math.floor(cor_len2/2+1)-int(dstack*Fs)):(math.floor(cor_len2/2+1)+int(dstack*Fs))]



t = np.arange(-dstack,dstack,1./Fs)
maxlag4 = t[np.argmax(ccor3_dt)]

# plotting the autocorrelation od two rows

fig, axs = plt.subplots(3,1, figsize = (15,10))
ax = axs[0]
ax.plot(time, row_1, label = 'row1')
ax.plot(time, row_2+0.001, label = 'row2')
ax.set_ylabel('Amplitude')
ax.set_title('Random noise signal')

ax = axs[1]
ax.plot(lags,ccor3)
ax.plot(lags[np.argmax(ccor3)], ccor3[np.argmax(ccor3)],'ro', label = 'max correlation')
ax.set_xlabel('lag of signal 1 relative to signal 2')
ax.set_ylabel('cross-correlation')
ax.set_title(f'max correlation is at lag {maxlag3:.5f} s')
ax.legend()

ax = axs[2]
ax.plot(t,ccor3_dt)
#ax.plot(lags[np.argmax(ccor3)], ccor3[np.argmax(ccor3)],'ro', label = 'max correlation')
ax.set_xlabel('lag of signal 1 relative to signal 2')
ax.set_ylabel('cross-correlation')
ax.set_title(f'ZOOM - max correlation is at lag {maxlag3:.5f} s')
ax.legend()

fig.suptitle('Case #3 - cross-correlation of two recordings Matera')

