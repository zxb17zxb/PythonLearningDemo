#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.graph_objs import *
from scipy import stats
import scipy as sp

ECG_Features = pd.read_csv("E:\\1-毕业课题项目\\拿来练手的EEG项目\\Dreamer\\Biosignal-Emotions-BHS-2020-master\\Biosignal-Emotions-BHS-2020-master\\DREAMER_features_ecg.csv")
del ECG_Features['Unnamed: 0']
# Number of features : Nf = 30
Nf=ECG_Features.shape[1]-1

# print(Nf)
# print(ECG_Features.head())


