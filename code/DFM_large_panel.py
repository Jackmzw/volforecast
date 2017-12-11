#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:39:26 2017

@author: Jack
"""

import DFM as dfm
import HAR as har
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def get_permn_sp500(day,sp500):
    day = int(day)
    sp500_permno = sp500[(sp500[:,1]<=day)&(sp500[:,2]>=day),0]
    sp500_permno = [str(v) for v in sp500_permno]
    return sp500_permno

def catch2(df,col,day,windows=252):
    tt = (df.loc[:day,col]).tail(windows+1)
    tt = tt.dropna(axis = 1)
    tt = np.log(tt)
    train = tt.iloc[:-1,:]
    test = tt.iloc[-1,:]
    return train, test    

Vol_1316 = pd.read_csv('../data/Vol_1316.csv',index_col=0,parse_dates=True)
sp500 = np.loadtxt('../data/sp500.txt',dtype="int64")

start = '20150102'
end = '20150105'
#end = '20161230'
date = Vol_1316[start:end].index
date_str = date.strftime('%Y%m%d')

MSE_DFM = pd.DataFrame(index = date,columns=['DFM','ICp1','ICp2'])
MSE_HAR = pd.DataFrame(index = date,columns=['HAR'])

for day in date_str:

    sp500_list = get_permn_sp500(day,sp500)
    train, test1 = catch2(Vol_1316, sp500_list, day)
    train_all, test2 = catch2(Vol_1316,Vol_1316.columns, day)
    
    X = train.values
    X = scale(X)
    best_dfm, ICp, r, results, ICp_r = dfm.dfm(X, m=6) 
    F = best_dfm[0]
    
    Y = train_all.values
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    Y = scale(Y)
    Lambda, D, v, resid = dfm.est_param(Y, F, m=6)
    pred_F = dfm.forecast_VAR(F, lag = 2)
    pred = dfm.forecast(Y,pred_F,Lambda, D)
    pred = pred*std+mean
    
    mse = np.mean((pred-test2.values)**2)
    MSE_DFM.loc[day] = [mse, ICp_r[0], ICp_r[1]]
    
    Har_factor = har.Factor_Har(train_all)
    pred,resid,mse = har.Fit(train_all,test2,Har_factor)
    MSE_HAR.loc[day,'HAR'] = mse 

MSE_DFM.to_csv('../results/MSE_DFM_1516.csv')
MSE_HAR.to_csv('../results/MSE_HAR_1516.csv')
    



