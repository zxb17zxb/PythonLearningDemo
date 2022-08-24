#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
output_filename = "HRV_Test.csv"
plt.rc('font', size=8)

# Download data
data = nk.data("bio_resting_8min_100hz")
# Clean signal and find peaks
ecg_cleaned = nk.ecg_clean(data["ECG"], sampling_rate=100)
# Find peaks
peaks, info = nk.ecg_peaks(data["ECG"], sampling_rate=100)


# Compute HRV indices
hrv_indices = nk.hrv(peaks, sampling_rate=100, show=True)

hrv_indices.to_csv(output_filename)
