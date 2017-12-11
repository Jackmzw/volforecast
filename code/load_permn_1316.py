#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:14:02 2017

@author: Jack
"""

import numpy as np
import pandas as pd
import urllib.request
from itertools import chain
import time

sp500 = np.loadtxt('../data/sp500.txt',dtype="int32")
sp500_list = np.unique(sp500[:,0])
permn = np.loadtxt('../data/permn_1316.txt',dtype="int32")
S = list(set(sp500_list).union(set(permn)))
permn = np.array(S)
N = len(permn)
date = [None]*N
vol = [None]*N

start = time.time()
for i in range(N):
    url = 'http://dachxiu.chicagobooth.edu/portal.php?api=vol&permno='+str(permn[i])
    with urllib.request.urlopen(url) as f: data = f.read().decode('ascii')
    rows = data.split("\n")[:-1]
    date[i] = [int(row.split(",")[0]) for row in rows]
    vol[i] = [np.sqrt(float(row.split(",")[1])) for row in rows]
    
end = time.time()
ela = end-start
print(ela)

uni_date = np.sort(list(set(chain.from_iterable(date[:N]))))
Vol = np.zeros((N,len(uni_date)))

for i in range(N):
    Vol[i,np.searchsorted(uni_date,date[i])] = vol[i]

date = [pd.to_datetime(str(d),format='%Y%m%d') for d in uni_date]
Vol = pd.DataFrame(Vol.T, index = date, columns=permn)
Vol[Vol==0] = np.nan

Vol.to_csv('../data/Vol_1316.csv')

