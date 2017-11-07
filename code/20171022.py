#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:02:24 2017

@author: Jack
"""

from statsmodels.tsa.arima_model import ARIMA

def catch(df,day=20160104,window=200):
    sp500_permno = sp500[(sp500[:,1]<=day)&(sp500[:,2]>=day),0]
    tt = df.loc[:str(day),sp500_permno].tail(window+1)
    tt = tt.dropna(axis = 1)
    tt = np.log(tt)
    train = tt.iloc[:-1]
    test = tt.iloc[-1]
    return train, test


def arima(train, test, order):
    N = train.shape[1]
    pred_arima = np.zeros(N)
    for i in range(N):
        s = train.iloc[:,i]
        try:
            model = ARIMA(s.values,order)
            fit = model.fit(disp=-1)
            pred_arima[i] = fit.forecast()[0][0]
        except:
            pred_arima[i] = np.nan
    mse = np.mean((test - pred_arima)**2)
    return pred_arima, mse
    
T = 400
MSE_ARIMA = pd.Series(index = date[5793:5793+T])

for t in range(T):
    train, test = catch(df=Vol,day=uni_date[t+5793])
    pred_arima, mse = arima(train,test,(1,1,1))    
    MSE_ARIMA[t] = mse    
    print(t)
        