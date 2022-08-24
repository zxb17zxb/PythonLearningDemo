#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
a = np.linspace(1,20,20).reshape(4,5)
print(a)
print(np.sum(a))
print(np.sum(a,axis = (0,2)))
