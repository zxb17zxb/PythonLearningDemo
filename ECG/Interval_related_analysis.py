#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Load NeuroKit and other useful packages
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
# This "decorative" cell should be hidden from the docs once this is implemented:
# https://github.com/microsoft/vscode-jupyter/issues/1182
plt.rcParams['figure.figsize'] = [15, 5]  # Bigger images
plt.rcParams['font.size']= 14
# Get data
data = nk.data("bio_resting_5min_100hz")
# Process ecg
ecg_signals, info = nk.ecg_process(data["ECG"], sampling_rate=100)

plot = nk.ecg_plot(ecg_signals[:3000], sampling_rate=100)
plot.show()
