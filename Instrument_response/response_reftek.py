#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:30:30 2021

@author: annamaria

Code to test the instrument response removal form Reftek sensor using my 
deconvolution funtion. 
+ check if the deconvolution was performed correctly.
+ options to check the prefiltering or post-filtering of the data
"""
#%% Import libraries
import obspy as ob
from obspy import read, signal
import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.invsim import paz_to_freq_resp, invert_spectrum, cosine_taper
from obspy.signal.util import _npts2nfft
import time

#%% Importing set parameters
from recordings_parameters.parameters import paz
from myfunctions import butter_highpass, butter_bandpass
#%% Reading the data
data = read('/Users/annamaria/PhD/Data/Matera/20201105_building_data/EQ/Reftek.9EDA.EHE.2019.298.04.31.10')
data.plot()

# signal pre-processing
data[0].data -= data[0].data.mean()
data[0].data *= cosine_taper(data[0].stats.npts , 0.1)
data.plot()


#%% Instument response  Reftek 
tic = time.time()

poles = paz["Reftek"]["poles"]  # The poles of the transfer function
zeros = paz["Reftek"]["zeros"]   # The zeros of the transfer function
scale_fac = paz["Reftek"]["gain"]         # Gain factor
sensitivity = paz["Reftek"]["sensitivity"]

t_samp = data[0].stats.delta     # Sampling interval in seconds
ndat = data[0].stats.npts        # Number of FFT points of signal which needs correction
nfft = _npts2nfft(ndat) 

h, f = paz_to_freq_resp(poles, zeros, scale_fac, t_samp, nfft, freq=True)
phase = 2 * np.pi + np.unwrap(np.angle(h))

toc = time.time() - tic
print(f' calculating instrument response = {toc} s')

# reftek sensor response plot
fig, ax = plt.subplots(1,2)

ax[0].loglog(f, abs(h))
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('Amplitude')

ax[1].semilogx(f, phase)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Phase [radian]')
ax[1].set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax[1].set_yticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax[1].set_ylim(-0.2, 2 * np.pi + 0.2)

fig.suptitle('Frequency Response of Reftek sensor')
fig.subplots_adjust(wspace=0.3)

#%% FFT and defining some parameters

samp_rate = data[0].stats.sampling_rate
paz_file =  paz["Reftek"]

print("Are we doing pre-filtering? (y/n)")
prefiltering = input()
if prefiltering == 'y':
    #data filtering
    data[0].filter("highpass", freq = 0.5)

fft_raw = np.fft.rfft(data[0].data, n=nfft)
time_vector = np.arange(ndat) * t_samp
freqaxis = np.fft.rfftfreq(nfft, d = 1./samp_rate)

#%% my deconvolution function
tic = time.time()

WL = 1
abs_h=abs(h)
h_wl = np.zeros(len(h),dtype = "complex")
# calculating the response with the waterlevel
WL2 = max(abs_h) * WL / 100

for i in range(len(h_wl)):
    if abs_h[i] == 0:
        h_wl[i] = WL2
    elif abs_h[i] < WL2: #and np.isclose(S2a[i],0):
        h_wl[i] = WL2 * h[i]/abs_h[i]
    elif abs_h[i] >= WL2:
        h_wl[i] = h[i]
fft_instr_rr3 = fft_raw / h_wl
fft_instr_rr3 /=sensitivity
signal_rr3 = np.fft.irfft(fft_instr_rr3)[0:ndat]

print("Are we doing post - filtering? (y/n)")
filtering = input()
if filtering == 'y':
    filtered_signal_rr3 = butter_bandpass(np.fft.irfft(fft_instr_rr3), 0.5, 20, samp_rate)[0:ndat]
    filtered_fft = np.fft.rfft(filtered_signal_rr3, n=nfft)
    
    fig, ax = plt.subplots()
    ax.plot(signal_rr3, label = 'response removal')
    ax.plot(filtered_signal_rr3, label = 'filtered')
    ax.plot(data[0].data, label = 'raw')
    ax.legend()
    
    fft_instr_rr3 = filtered_fft
else:
    fig, ax = plt.subplots()
    ax.plot(signal_rr3, label = 'response removal')
    ax.plot(data[0].data, label = 'raw')
    ax.legend()

toc = time.time() - tic
print(f'Response removal using my deconvolution function  = {toc} s')

# normalized signals
norm_raw = abs(fft_raw)/max(abs(fft_raw))
norm_ic = abs(fft_instr_rr3)/max(abs(fft_instr_rr3))

#%% Checking the deconvolution

rm_check_ws = fft_raw / fft_instr_rr3
rm_check = fft_raw / fft_instr_rr3 * sensitivity

fig, ax = plt.subplots(1,2, figsize = (15,5))
ax[0].loglog(freqaxis, abs(rm_check))
ax[0].set_title("spectral ratio without sensitivity removal")

ax[1].loglog(freqaxis, abs(rm_check_ws))
ax[1].set_title("spectral ratio with  sensitivity removal")

fig.suptitle(f'Checking the deconvolution process with waterlevel = {WL} %')

#%% plot - comparison the fourier spectra's

fig, axs = plt.subplots(4,1, figsize = (10,13))

ax = axs[0]
ax.plot(signal_rr3)
ax.set_xlabel('Time')
ax.set_ylabel('Velocity [m/s]')
ax.set_title("signal after response removal")

ax = axs[1]
ax.loglog(freqaxis, abs(fft_raw))
ax.set_title("Spectra of the raw signal")
ax.set_xlabel('Frequency [Hz]')

ax = axs[2]
ax.loglog(freqaxis, abs(fft_instr_rr3))
ax.set_xlabel('Frequency [Hz]')
ax.set_title(f'Spectra after response removal with waterlevel = {WL} %')

ax = axs[3]
ax.loglog(freqaxis, norm_raw , label = 'raw signal')
ax.loglog(freqaxis, norm_ic , label = 'insturment correction')
ax.set_xlabel('Frequency [Hz]')
ax.legend()
ax.set_title(f'Normalized comparison of the fft before and after response removal, waterlevel = {WL} %')

fig.suptitle("Reftek", fontsize = 20)
fig.tight_layout()