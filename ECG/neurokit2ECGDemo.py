#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# Load the package
import neurokit2 as nk

# Download the example dataset
data = nk.data("bio_resting_5min_100hz")

# Process the data
df, info = nk.bio_process(ecg=data["ECG"],
                          rsp=data["RSP"],
                          sampling_rate=100)
# Extract features
results = nk.bio_analyze(df, sampling_rate=100)

# Show subset of results
res = results[["ECG_Rate_Mean","HRV_RMSSD","RSP_Rate_Mean","RSA_P2T_Mean"]]

print(res)


