#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:34:17 2021

@author: annamaria

Code for cross-correlation and deconvolution test
% data used - recording from Matera - Line 1 geophones
% recording - 25_10_2019_10_51_00_000
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import obspy
import math
from scipy import signal
import time
from scipy.fft import fft, ifft
from myfunctions import deconvo, rm_damaged_signal, interfer_deconvo

#%% opening the files and extracing the data from the .txt file

# load the file
file = open("/Users/annamaria/PhD/Data/Matera/20200214_Line1_Matera/151_25_10_2019_10_51_00_000.txt_line1",'r')
data_all= np.loadtxt(file)

# some parameters setting
Fs = 250
dt = 1./Fs
nb_sen = 20
n = 45056

# changing the vector into data matrix (rows - data points, columns - sensors)
data_all = data_all.reshape((nb_sen,n)).T

# time vector
time = np.arange(n) * dt

#%% trend and mean removal

## mean removal
data_zero_mean = data_all - data_all.mean(axis=0)

## trend removal
# defining trend coefficients
trend_coeff = np.polyfit(time, data_zero_mean, 1)

# loop for defining trend for each sensor
sensor_trend = np.empty((n,nb_sen), dtype = 'float')
for sensor in range(nb_sen):
    sensor_trend[:,sensor] = trend_coeff[0,sensor] * time + trend_coeff[1,sensor]

data_detrend = data_zero_mean - sensor_trend


## pre-processed data
data = data_detrend.copy()

#%% removing damaged signals

treshold = 0.26 
data = rm_damaged_signal(data,treshold)

#%% Plotting the signals

fig, ax = plt.subplots(figsize = (15,15))

for sensor in range(nb_sen):
    ax.plot(data[:,sensor]+0.0005*sensor, linewidth = 0.3)
plt.tight_layout()

#%% Cross-correlation


## for one recording 
cc = np.correlate(data[:,0], data[:,2], mode = 'full')
lags = np.linspace(-(n-1),n-1,n*2-1) # lags axis

max_lag_cc = lags[np.argmax(cc)]
max_lag_time_cc = max_lag_cc * dt



## for all recordings
cc_matrix = np.empty((len(lags),nb_sen), dtype = 'float')
max_lag_cc_vector = np.empty((nb_sen), dtype = 'float')
max_lag_cc_time_vector = np.empty((nb_sen), dtype = 'float')

for sensor in range(nb_sen):
    cc_matrix[:,sensor] = np.correlate(data[:,0], data[:,sensor], mode = 'full')
    max_lag_cc_vector[sensor] = lags[np.argmax(cc_matrix[:,sensor])]
    max_lag_cc_time_vector[sensor] = max_lag_cc_vector[sensor] * dt


## plots
#  one time series
fig, ax = plt.subplots()
ax.plot(lags,cc)
ax.plot(max_lag_cc, max(cc), 'ro')
ax.set_xlabel('lags')
ax.set_title(f'Max lag time is {max_lag_time_cc} s')

# all recordings
fig, ax = plt.subplots(figsize = (15,15))
for sensor in range(nb_sen):
    ax.plot(lags*dt,cc_matrix[:,sensor]+0.00001*sensor)
    ax.plot(max_lag_cc_vector[sensor]* dt , max(cc_matrix[:,sensor])+0.00001*sensor, 'ro')
    ax.set_xlabel('lags')
ax.set_xlim([-0.5,0.5])
    

#%% Deconvolution

## setting necessary parameters for deconvolution
wl = 1  # water level in %
r = 3   # resampling for better peak peaking
dstack = 0.5   # length of the singal for the visualisation of the deconvolution [in  sec]
 
#-----------------------------------------------------------------------------

## for one signal

# signal deconvolution
sig_deco, t = interfer_deconvo(data[:,0], data[:,19], wl, r, dstack, Fs, t_ax = True)

# max delay values
max_lag_d = t[np.argmax(sig_deco_r)]
max_lag_time_d = max_lag_d * dt


## for all recordings
sig_deco_matrix = np.empty((Fs * r+1,nb_sen))   #empty matrix for all recordings deconvolution

for sensor in range(nb_sen):
    sig_deco_temp = interfer_deconvo(data[:,0], data[:,sensor], wl, r, dstack, Fs)
    sig_deco_matrix[:,sensor] = sig_deco_temp


## plots
# one signal
fig, ax = plt.subplots(figsize = (15,5))
ax.plot(t,sig_deco_r)
ax.plot(max_lag_d, max(sig_deco_r), 'ro')
ax.set_title(f'Max delay is {max_lag_time_d} s')

# all recordings
fig, ax = plt.subplots(figsize = (15,15))
for sensor in range(nb_sen):
    ax.plot(t,sig_deco_matrix[:,sensor]+0.1*sensor)
    ax.set_xlabel('time delay [s]')


