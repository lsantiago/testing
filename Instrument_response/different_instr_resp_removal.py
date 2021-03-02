#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:52:39 2021

@author: annamaria

Testing different types of instrument response removal methods
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
from myfunctions import butter_bandpass
from recordings_parameters.parameters import paz
#%% Reading the data
path = '/Users/annamaria/PhD/Data/Matera/20201105_building_data/EQ/'

""" Files to choose:
    -> Lunitek.GE004.EHE.2019.298.04.31.10
    -> Reftek.9EDA.EHE.2019.298.04.31.10
    -> CUBE.c0AAU.p2.2019.298.04.31.10
"""
example_files = ['Lunitek.GE004.EHE.2019.298.04.31.10', 'Reftek.9EDA.EHE.2019.298.04.31.10', 'CUBE.c0AAU.p2.2019.298.04.31.10']
example = 2 # choose a number from 0 to 2
file = example_files[example]

data = read(path + file)
data.plot()

sensor = file.split('.')[0]

# signal pre-processing
data[0].data -= data[0].data.mean()
data[0].data *= cosine_taper(data[0].stats.npts , 0.1)
data.plot()

#%% Instument response  
tic = time.time()

poles = paz[sensor]["poles"]  # The poles of the transfer function
zeros = paz[sensor]["zeros"]   # The zeros of the transfer function
scale_fac = paz[sensor]["gain"]         # Gain factor
sensitivity = paz[sensor]["sensitivity"]

t_samp = data[0].stats.delta     # Sampling interval in seconds
ndat = data[0].stats.npts        # Number of FFT points of signal which needs correction
nfft = _npts2nfft(ndat) 

h, f = paz_to_freq_resp(poles, zeros, scale_fac, t_samp, nfft, freq=True)

if sensor == "Lunitek":
    phase = np.pi + np.unwrap(np.angle(h))
else:
    phase = 2* np.pi + np.unwrap(np.angle(h))

toc = time.time() - tic
print(f' calculating instrument response = {toc} s')

#%% Define some parameters

samp_rate = data[0].stats.sampling_rate
paz_file =  paz[sensor]

fft_raw = np.fft.rfft(data[0].data, n=nfft)
time_vector = np.arange(ndat) * t_samp
freqaxis = np.fft.rfftfreq(nfft, d = 1./samp_rate)

#%% Approach #1 - Obspy simulate seismometer 
tic = time.time()
wl = 40

signal_rr1 = ob.signal.invsim.simulate_seismometer(
        data[0].data, samp_rate, paz_remove=paz_file, paz_simulate=None,
        remove_sensitivity=True, simulate_sensitivity=False, water_level=wl,
        zero_mean=False, taper=False, pre_filt=None,
        seedresp=None, nfft_pow2=False, pitsasim=False, sacsim=False,
        shsim=False)

fft_instr_rr1 = np.fft.rfft(signal_rr1, n=nfft)

toc = time.time() - tic
print(f'Approach #1 - Obspy simulate seismometer  = {toc} s')

#%% Approach #2 - Obspy manual response removal
tic = time.time()

response = h.copy()
invert_spectrum(response, wl)
fft_instr_rr2 = fft_raw * response
fft_instr_rr2 /= sensitivity

signal_rr2 = np.fft.irfft(fft_instr_rr2)[0:ndat]

toc = time.time() - tic
print(f'Approach #2 - Obspy manual response removal  = {toc} s')
#%% Approach #3 - Using my deconvolution function
tic = time.time()

wl_deconvo = 1
abs_h=abs(h)
h_wl = np.zeros(len(h),dtype = "complex")
# calculating the response with the waterlevel
WL2 = max(abs_h) * wl_deconvo / 100

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

toc = time.time() - tic
print(f'Approach #3 - Using my deconvolution function  = {toc} s')

#%% Filtering of the data

# setting filtering parameters
fmin = 0.1
fmax = 30
fs = samp_rate

# filtered signal in time domain
filtered_sig1 = butter_bandpass(signal_rr1, fmin, fmax, fs)
filtered_sig2 = butter_bandpass(signal_rr2, fmin, fmax, fs)
filtered_sig3 = butter_bandpass(signal_rr3, fmin, fmax, fs)

filt_fft_sig1 = np.fft.rfft(filtered_sig1, n=nfft)
filt_fft_sig2 = np.fft.rfft(filtered_sig2, n=nfft)
filt_fft_sig3 = np.fft.rfft(filtered_sig3, n=nfft)

#%% Figures
# sensor response
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

fig.suptitle(f'Frequency Response of {sensor} sensor')
fig.subplots_adjust(wspace=0.3)

# Approach #1 - Obspy simulate seismometer 
fig, axs = plt.subplots(4,1,figsize=(15,15))

ax = axs[0]
ax.plot(time_vector, signal_rr1, label = 'no filter', color = 'blue')
ax.plot(time_vector, filtered_sig1, label = "filtered", color = 'green')
ax.set_xlabel("Time [s]", fontsize = 14)
ax.set_ylabel("Velocity [m/s]", fontsize = 14)
ax.set_title("Signal after instrument response", fontsize = 16)
ax.legend()

ax = axs[1]
ax.loglog(freqaxis,abs(fft_instr_rr1), label = 'no filter', color = 'blue')
ax.loglog(freqaxis,abs(filt_fft_sig1), label = "filtered", color = 'green')
ax.set_xlabel("Frequency [Hz]", fontsize = 14)
ax.set_ylabel("Amplitude", fontsize = 14)
ax.set_title("Signal after instrument response", fontsize = 16)
ax.legend()

ax = axs[2]
ax.plot(time_vector, data[0].data/max(data[0].data), label = "raw signal", color = 'orange')
ax.plot(time_vector, signal_rr1/max(signal_rr1), label = "instrument correction", color = 'blue')
ax.set_xlabel("Time", fontsize = 14)
ax.set_ylabel("Normalized amplitude", fontsize = 14)
ax.legend()
ax.set_title("Normalized signal in time domain", fontsize = 16)

ax = axs[3]
ax.loglog(freqaxis,abs(fft_raw)/max(abs(fft_raw)), label = "raw signal", color = 'orange')
ax.loglog(freqaxis,abs(fft_instr_rr1)/max(abs(fft_instr_rr1)), label = "instrument correction", color = 'blue')
ax.set_xlabel("Frequency [Hz]", fontsize = 14)
ax.set_ylabel("Normalized amplitude", fontsize = 14)
ax.set_title("Normalized signals in frequency domain", fontsize = 16)
ax.legend()

fig.suptitle(f"Approach #1 - Obspy simulate_seismometer(). Waterlevel = {wl} dB", fontsize = 16)
fig.tight_layout()

# Approach #2 - Obspy manual response removal
fig, axs = plt.subplots(4,1,figsize=(15,15))

ax = axs[0]
ax.plot(time_vector, signal_rr2, label = 'no filter', color = 'blue')
ax.plot(time_vector, filtered_sig2, label = 'filtered', color = 'green')
ax.set_ylabel("Velocity [m/s]", fontsize = 14)
ax.set_xlabel("Time [s]", fontsize = 14)
ax.set_title("Signal after instrument response", fontsize = 16)
ax.legend()

ax = axs[1]
ax.loglog(freqaxis,abs(fft_instr_rr2), label = 'no filter', color = 'blue')
ax.loglog(freqaxis,abs(filt_fft_sig2), label = "filtered", color = 'green')
ax.set_xlabel("Frequency [Hz]", fontsize = 14)
ax.set_ylabel("Amplitude", fontsize = 14)
ax.set_title("Signal after instrument response", fontsize = 16)
ax.legend()

ax = axs[2]
ax.plot(time_vector, data[0].data/max(data[0].data), label = "raw signal", color = 'orange')
ax.plot(time_vector, signal_rr2/max(signal_rr2), label = "instrument correction", color = 'blue')
ax.set_xlabel("Time", fontsize = 14)
ax.set_ylabel("Normalized amplitude", fontsize = 14)
ax.legend()
ax.set_title("Normalized signal in time domain", fontsize = 16)

ax = axs[3]
ax.loglog(freqaxis,abs(fft_raw)/max(abs(fft_raw)), label = "raw signal", color = 'orange')
ax.loglog(freqaxis,abs(fft_instr_rr2)/max(abs(fft_instr_rr2)), label = "instrument correction", color = 'blue')
ax.set_xlabel("Frequency [Hz]", fontsize = 14)
ax.set_ylabel("Normalized amplitude", fontsize = 14)
ax.set_title("Normalized signals in frequency domain", fontsize = 16)
ax.legend()

fig.suptitle(f"Approach #2 - manual removal with waterlevel (Obspy). Waterlevel = {wl} dB", fontsize = 16)
fig.tight_layout()

# Approach #3 - Using my deconvolution function
fig, axs = plt.subplots(4,1,figsize=(15,15))

ax = axs[0]
ax.plot(time_vector, signal_rr3, label = 'no filter', color = 'blue')
ax.plot(time_vector, filtered_sig3, label = "filtered", color = 'green')
ax.set_xlabel("Time [s]", fontsize = 14)
ax.set_ylabel("Velocity [m/s]", fontsize = 14)
ax.set_title("Signal after instrument response", fontsize = 16)
ax.legend()

ax = axs[1]
ax.loglog(freqaxis,abs(fft_instr_rr3), label = 'no filter', color = 'blue')
ax.loglog(freqaxis,abs(filt_fft_sig3), label = "filtered", color = 'green')
ax.set_xlabel("Frequency [Hz]", fontsize = 14)
ax.set_ylabel("Amplitude", fontsize = 14)
ax.set_title("Signal after instrument response", fontsize = 16)
ax.legend()

ax = axs[2]
ax.plot(time_vector, data[0].data/max(data[0].data), label = "raw signal", color = 'orange')
ax.plot(time_vector, signal_rr3/max(signal_rr3), label = "instrument correction", color = 'blue')
ax.set_xlabel("Time", fontsize = 14)
ax.set_ylabel("Normalized amplitude", fontsize = 14)
ax.legend()
ax.set_title("Normalized signal in time domain", fontsize = 16)

ax = axs[3]
ax.loglog(freqaxis,abs(fft_raw)/max(abs(fft_raw)), label = "raw signal", color = 'orange')
ax.loglog(freqaxis,abs(fft_instr_rr3)/max(abs(fft_instr_rr3)), label = "instrument correction", color = 'blue')
ax.set_xlabel("Frequency [Hz]", fontsize = 14)
ax.set_ylabel("Normalized amplitude", fontsize = 14)
ax.set_title("Normalized signals in frequency domain", fontsize = 16)
ax.legend()

fig.suptitle(f"Approach #3 - using deconvo() function (manual). Waterlevel = {wl_deconvo} %", fontsize = 16)
fig.tight_layout()

# Comparision of all three approaches
fig, axs = plt.subplots(3,1, figsize = (15,10))

ax = axs[0]
ax.plot(time_vector, signal_rr1, label = "approach #1")
ax.plot(time_vector, signal_rr2, label = "approach #2")
ax.plot(time_vector, signal_rr3, label = "approach #3", linestyle = ":")
ax.set_ylabel("Velocity [m/s]", fontsize = 14)
ax.set_xlabel("Time [s]", fontsize = 14)
ax.set_title("Signals after instrument response", fontsize = 16)
ax.legend()

ax = axs[1]
ax.plot(time_vector[15000:30000], signal_rr1[15000:30000], label = "approach #1", linestyle = ":", linewidth = 3)
ax.plot(time_vector[15000:30000], signal_rr2[15000:30000], label = "approach #2", linestyle = "--")
ax.plot(time_vector[15000:30000], signal_rr3[15000:30000], label = "approach #3", linestyle = ":")
ax.set_ylabel("Velocity [m/s]", fontsize = 14)
ax.set_xlabel("Time [s]", fontsize = 14)
ax.set_title("Signals after instrument response - zoom", fontsize = 16)
ax.legend()

ax = axs[2]
ax.loglog(freqaxis, abs(fft_instr_rr1), label = "approach #1")
ax.loglog(freqaxis, abs(fft_instr_rr2), label = "approach #2")
ax.loglog(freqaxis, abs(fft_instr_rr3), label = "approach #3")
ax.set_xlabel("Frequency [Hz]", fontsize = 14)
ax.set_ylabel("Amplitude", fontsize = 14)
ax.set_title("Signal after instrument response", fontsize = 16)
ax.legend()

fig.suptitle("Comparision of the results using different approaches", fontsize = 16)
fig.tight_layout()



