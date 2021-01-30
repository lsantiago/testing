#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:11:58 2020

@author: annamaria

Testing the signal processing tools for building noise data
"""


#%%
# Import libraries

import numpy as np
from obspy import read, read_inventory
import matplotlib.pyplot as plt
import scipy

#%% Reading the file

st = read('/Users/annamaria/PhD/Data/Matera/20201105_building_data/Reftek/921B/921B.EHZ.2019.296.07.07.17')
print(st[0].stats)

#%% Trend and mean removal 





# detrending 
detrend_st = read('/Users/annamaria/PhD/Data/Matera/20201105_building_data/Reftek/921B/921B.EHZ.2019.296.07.07.17')
detrend_st = detrend_st.detrend(type='simple')

# mean removal
mean_rm_st = detrend_st.copy()
mean_rm_st = mean_rm_st.detrend(type='demean')

# Plotting the figures
plt.figure(figsize=(20,5))
plt.plot(st[0].data[4000:5000],lw = 4, label = "raw")
plt.plot(detrend_st[0].data[4000:5000],lw = 2, ls="--",label = "detrended")
plt.plot(mean_rm_st[0].data[4000:5000],ls = "--", lw = 2, label = "detrended + mean removed")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()



#%% filtering

c_freq = 4

# creating the hight pass filter
highpass_filter_st = mean_rm_st.copy()
highpass_filter_st = highpass_filter_st.filter('highpass', freq=c_freq, corners=1, zerophase=True)

# creating the low pass filter
lowpass_filter_st = mean_rm_st.copy()
lowpass_filter_st = lowpass_filter_st.filter('lowpass', freq=c_freq, corners=1, zerophase=True)

fmin, fmax = 1, 4
# creating the band pass filter
bandpass_filter_st = mean_rm_st.copy()
bandpass_filter_st = bandpass_filter_st.filter('bandpass', freqmin = fmin, freqmax = fmax , corners=1, zerophase=True)



# Figure of the filtered signal
fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(15,14))

ax1.plot(mean_rm_st[0].data[:4000], label = "detrended + mean removed")
ax1.plot(highpass_filter_st[0].data[:4000], label = "highpass filter")
ax1.set_title(f"highpass filter - corner frequency {c_freq} Hz")

ax2.plot(mean_rm_st[0].data[:4000], label = "detrended + mean removed")
ax2.plot(lowpass_filter_st[0].data[:4000], label = "lowpass filter")
ax2.set_title(f"lowpass filter - corner frequency {c_freq} Hz")

ax3.plot(mean_rm_st[0].data[:4000], label = "detrended + mean removed")
ax3.plot(bandpass_filter_st[0].data[:4000], label = "bandpass filter")
ax3.set_title(f"bandpass filter - corner frequencies {fmin} - {fmax} Hz ")


#%% Response removal

from obspy.signal.invsim import paz_to_freq_resp

#poles = [-2.007840e+01 + 1.916930e+01j, -2.007840e+01 - 1.916930e+01j]#, -1.083 + 0.0j]
zeros = [(0.0 + 0.0j), (0.0 + 0.0j),( 0.0 + 0.0j)]
scale_fac = 2.517300e+08#8.944000e+01
poles = [(-4.210000e+00 + 4.660000e+00j), (-4.210000e+00 - 4.660000e+00j),
	(-2.105000e+00 + 0.000000e+00j)]

h, f = paz_to_freq_resp(poles, zeros, scale_fac, 0.004, 3000, freq=True)
 

response_st = mean_rm_st.copy()
plt.figure()
plt.subplot(121)
plt.loglog(f, abs(h))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')

plt.subplot(122)
phase = 2 * np.pi + np.unwrap(np.angle(h))
plt.semilogx(f, phase)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radian]')
# ticks and tick labels at multiples of pi
plt.yticks(
    [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
    ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
plt.ylim(-0.2, 2 * np.pi + 0.2)
# title, centered above both subplots
plt.suptitle('Frequency Response of LE-3D/1s Seismometer')
# make more room in between subplots for the ylabel of right plot
plt.subplots_adjust(wspace=0.3)
plt.show()


#%%
from obspy.signal.invsim import simulate_seismometer
reftek = ob.read('/Users/annamaria/PhD/Data/Matera/20201105_building_data/Reftek/9EAA/9EAA.EHE.2019.296.08.33.38')
response = {'gain': scale_fac,
            'poles': poles,
            'sensitivity': 2.519500e+08,
            'zeros': zeros}
df = reftek[0].stats.sampling_rate

data = reftek.copy()

data[0].data = simulate_seismometer(data[0].data, df, paz_remove=response, simulate_sensitivity=False, water_level=60.0)


fig, ax = plt.subplots()

ax.plot(reftek[0].data[:4000])
ax.plot(data[0].data[:4000])


plt.plot(data[0].data[int(13895500/2):])
