#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:32:37 2021

@author: annamaria

Code to test stacking of the signals from 1 line of recordings in:
    - cross correlation
    - deconvolution
Data used: Matera data - Line 1 of vertical geophones recordings
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import obspy
import os
import fnmatch
from myfunctions import deconvo, rm_damaged_signal, interfer_deconvo, xcorr
import time
#%% Importing signals

# defining the path for the data to read
path = '/Users/annamaria/PhD/Data/Matera/20200429_Matera_ottobre2019_dati_passivi_campo_sportivo_ASCII/final_preprocessed/line1/'

# creating the sorted list of files 
file_list = sorted(os.listdir(path))

# creating the list containing only data files
file_list = [s for s in file_list if fnmatch.fnmatch(s, '*line1*')]


#%% some parameters setting
Fs = 250
dt = 1./Fs
nb_sen = 20
n = 45056

## setting necessary parameters for deconvolution
wl = 1  # water level in %
r = 2   # resampling for better peak peaking
dstack = 0.5   # length of the singal for the visualisation of the deconvolution [in  sec]

#%% Deconvolution 

tic = time.time()

count = 1
sig_deco_matrix = np.empty((Fs * r+1,nb_sen))   #empty matrix for all recordings deconvolution

for file in file_list:
    data = np.load(path+file)
    print(file, f'the iteration number is {count}')
    
    for sensor in range(nb_sen):
        sig_deco_temp = interfer_deconvo(data[:, 0], data[:, sensor], wl, r, dstack, Fs)
        sig_deco_matrix[:, sensor] = sig_deco_temp
    
    if count == 1:
        deconvo_stacked = sig_deco_matrix
    else:
        deconvo_stacked += sig_deco_matrix
    
    count += 1

toc = time.time() - tic
print(f'Deconvolution loop takes: {toc} s')


t = np.linspace(-dstack, dstack, deconvo_stacked.shape[0])
max_lag_deconvo_vector = t[np.argmax(deconvo_stacked,axis =0)]


#%% Deconvolution plot
fig, ax = plt.subplots(figsize = (15,15))
labels = []
offset = 0.3

for sensor in range(nb_sen):
    y = sig_deco_matrix[:,sensor] + offset * sensor
    
    ax.plot(t, y, linewidth = 2)
    ax.set_xlabel('time delay [s]')
    ax.fill_between(t, offset * sensor, y, where = (y > 0 + offset * sensor))
    ax.plot(max_lag_deconvo_vector[sensor], max(y), 'ro')
    labels.append(str(sensor + 1))
    
ax.vlines(0, ymin = -0.5, ymax = offset * sensor + 0.5, color = 'red')
ax.set_yticks(np.linspace(0,nb_sen-1,num=nb_sen)* offset)
ax.set_yticklabels(labels)
ax.set_ylabel('sensor nb', fontsize = 20)
ax.set_xlabel('time [s]', fontsize = 20)
ax.set_xlim([-0.5, 0.5])
ax.set_title('Deconvolution', fontsize = 20)   
    

    
#%% Cross-correlation
lags = np.linspace(-(n-1),n-1,n*2-1) # lags axis
cc_matrix = np.empty((len(lags),nb_sen), dtype = 'float')

count1 = 1

tic = time.time()

for file in file_list:
   data = np.load(path+file)
   print(file, f'the iteration number is {count1}')   
   for sensor in range(nb_sen): 
       cc_temp = np.correlate(data[:,0], data[:,sensor], mode = 'full')
       cc_matrix[:,sensor] = cc_temp
   
   if count1 == 1:
       cc_stacked = cc_matrix
   else:
       cc_stacked += cc_matrix
   count1 += 1

toc = time.time() - tic
print(f'Cross-correlation loop takes: {toc} s')


#max_lag_cc_vector = np.empty((nb_sen), dtype = 'float')
#max_lag_cc_time_vector = np.empty((nb_sen), dtype = 'float')

max_lag_cc_vector = lags[np.argmax(cc_stacked,axis =0)]
max_lag_cc_time_vector = max_lag_cc_vector * dt
    
    
#%% all recordings
fig, ax = plt.subplots(figsize = (15,15))

labels = []
offset = 0.0002
for sensor in range(nb_sen):
    y = cc_stacked[:,sensor] * (1 + 0.5 * sensor) + offset * sensor
    
    ax.plot(lags * dt, y, linewidth = 2)
    ax.fill_between(lags * dt, offset * sensor, y, where = (y > 0 + offset * sensor)) 
    ax.plot(max_lag_cc_vector[sensor] * dt , max(y), 'ro')
   
    labels.append(str(sensor + 1))
    
ax.vlines(0, ymin = -0.0001, ymax = offset * sensor + 0.0001, color = 'red')

ax.set_yticks(np.linspace(0,nb_sen-1,num=nb_sen)* offset)
ax.set_yticklabels(labels)
ax.set_ylabel('sensor nb', fontsize = 20)
ax.set_xlabel('time [s]', fontsize = 20)

ax.set_xlim([-1,1])  
ax.set_title('Cross-correlation - 1st approach (all data lag)', fontsize = 20)    
    
    
#%% cross-correlation using xcorr
#lags = np.linspace(-(n-1),n-1,n*2-1) # lags axis
cc= np.empty((501,nb_sen), dtype = 'float')

count2 = 1
max_lag = 250
tic = time.time()


for file in file_list:
   data = np.load(path+file)
   print(file, f'the iteration number is {count2}')   
   for sensor in range(nb_sen): 
       c = xcorr(data[:,0], data[:,sensor], max_lag)
       cc[:,sensor] = c
   if count2 == 1:
       cc_stacked2 = cc
   else:
       cc_stacked2 += cc
   count2 += 1

toc = time.time() - tic
print(f'Cross-correlation loop using xcorr takes: {toc} s')

lags_cc = np.linspace(-max_lag, max_lag, 2*max_lag+1)

max_lag_cc_vector2 = lags_cc[np.argmax(cc_stacked2,axis =0)]
max_lag_cc_time_vector2 = max_lag_cc_vector2 * dt

#%% all recordings
fig, ax = plt.subplots(figsize = (15,15))

labels = []
offset = 0.0002
for sensor in range(nb_sen):
    y = cc_stacked2[:,sensor] * (1 + 0.5 * sensor) + offset * sensor
    
    ax.plot(lags_cc * dt, y, linewidth = 2)
    ax.fill_between(lags_cc * dt, offset * sensor, y, where = (y > 0 + offset * sensor)) 
    ax.plot(max_lag_cc_vector2[sensor] * dt , max(y), 'ro')
   
    labels.append(str(sensor + 1))
    
ax.vlines(0, ymin = -0.0001, ymax = offset * sensor + 0.0001, color = 'red')

ax.set_yticks(np.linspace(0,nb_sen-1,num=nb_sen)* offset)
ax.set_yticklabels(labels)
ax.set_ylabel('sensor nb', fontsize = 20)
ax.set_xlabel('time [s]', fontsize = 20)
 
ax.set_title('Cross-correlation - 2nd approach (using max-lag)', fontsize = 20)  


#%%  lag figure


fig, ax = plt.subplots(figsize = (10,7))

ax.plot(max_lag_deconvo_vector, 'go', label = 'Deconvolution', markersize = 12)
ax.plot(max_lag_cc_vector * dt, 'b^', label = 'CC 1st approach', markersize = 12)
ax.plot(max_lag_cc_vector2 * dt, 'r*', label = 'CC 2nd approach', markersize = 12)

ax.set_xticks(np.linspace(0,nb_sen-1,num=nb_sen))
ax.set_xticklabels(labels)
ax.set_ylabel("time lag [s]", fontsize = 20)
ax.set_xlabel("senor number", fontsize = 20)
ax.legend()
ax.set_ylim([-0.1,0.05])
ax.set_title('Comparison of lag times', fontsize = 20)





