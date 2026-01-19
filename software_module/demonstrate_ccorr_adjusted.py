#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:08:59 2024

@author: eric
"""
import numpy as np
from scipy.signal import hilbert
from astropy.stats import circmean
import time



n_ch = 64   # Number of channels
n_samp = 1000 # Number of samples

# Step 1: Generate two random signals of size 1000x64
signal_1 = np.random.randn(n_samp, n_ch)
signal_2 = np.random.randn(n_samp, n_ch)

# Step 2: Concatenate the signals along the second axis to get a 1000x128 signal
combined_signal = np.concatenate((signal_1, signal_2), axis=1)

# Step 3: Apply Hilbert transform on the concatenated signal (per channel)
complex_signal = hilbert(combined_signal, axis=0)  # Shape is now 1000x128

transpose_axes = (1, 0)
# Start timing the code execution
start_time = time.time()

# Step 4: Calculate the phase angle of the complex signal
angle = np.angle(complex_signal).T

r1 = np.einsum('lm,mk->lk', np.exp(1j * angle), 1/np.exp(1j * angle.transpose(transpose_axes))) #abminus
r2 = np.einsum('lm,mk->lk', np.exp(1j * angle), np.exp(1j * angle.transpose(transpose_axes))) #abplus
mu = ((np.angle(r2)+ np.angle(r1))/2).reshape(2 * n_ch, 2 * n_ch, 1) #epoch x freq x 2*n_ch x 2*n_ch x 1


angle1 = np.tile(np.expand_dims(angle, axis=1), (1, 2*n_ch, 1)) 
den = 2*np.sqrt(np.sum(np.sin(angle1 - mu)**2,axis =2) * np.sum(np.sin(angle1 - mu)**2,axis =2).transpose((1,0)))
 
con = (np.abs(r1)-np.abs(r2))/den

ccorr = np.abs(con)

# Stop timing the code execution
end_time = time.time()

# Print the time taken to run the code
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

# con now contains the result of the einsum operation applied on the Hilbert-transformed signal