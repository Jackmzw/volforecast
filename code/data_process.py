#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:47:11 2017

@author: Jack
"""

import numpy as np
import pandas as pd
import urllib.request
from itertools import chain
import time

sp500 = np.loadtxt('./sp500.txt',dtype="int64")
sp500_list = np.unique(sp500[:,0])
N = len(sp500_list)
date = [None]*N
vol = [None]*N
rv5 = [None]*N
rv15 = [None]*N

start = time.time()
for i in range(N):
    url1 = 'http://dachxiu.chicagobooth.edu/portal.php?api=vol&permno='+str(sp500_list[i])
    url2 = 'http://dachxiu.chicagobooth.edu/portal.php?api=rv&permno='+str(sp500_list[i])
    with urllib.request.urlopen(url1) as f: data = f.read().decode('ascii')
    rows = data.split("\n")[:-1]
    date[i] = [int(row.split(",")[0]) for row in rows]
    vol[i] = [np.sqrt(float(row.split(",")[1])) for row in rows]
    with urllib.request.urlopen(url2) as f: data = f.read().decode('ascii')
    rows = data.split("\n")[:-1]
    #date[i] = [int(row.split(",")[0]) for row in rows]
    rv5[i] = [float(row.split(",")[1]) for row in rows]
    rv15[i] = [float(row.split(",")[2]) for row in rows]   
end = time.time()
ela = end-start
print(ela)

uni_date = np.sort(list(set(chain.from_iterable(date[:N]))))
Vol = np.zeros((N,len(uni_date)))
RV5 = np.zeros((N,len(uni_date)))
RV15 = np.zeros((N,len(uni_date)))

for i in range(N):
    Vol[i,np.searchsorted(uni_date,date[i])] = vol[i]
    RV5[i,np.searchsorted(uni_date,date[i])] = rv5[i]
    RV15[i,np.searchsorted(uni_date,date[i])] = rv15[i]

date = [pd.to_datetime(str(d),format='%Y%m%d') for d in uni_date]
Vol = pd.DataFrame(Vol.T, index = date, columns=sp500_list)
Vol[Vol==0] = np.nan
RV5 = pd.DataFrame(RV5.T, index = date, columns=sp500_list)
RV5[RV5==0] = np.nan
RV15 = pd.DataFrame(RV15.T, index = date, columns=sp500_list)
RV15[RV15==0] = np.nan

Vol.to_csv('./Vol.csv')
RV5.to_csv('./RV5.csv')
RV15.to_csv('./RV15.csv')



