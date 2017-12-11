#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 14:47:35 2017

@author: Jack
"""



import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import pyplot, pylab
from sklearn.preprocessing import scale
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
import time


'''
def catch(Vol, i = 5793, window = 200):
    sp500_permno = sp500[(sp500[:,1]<=uni_date[i])&(sp500[:,2]>=uni_date[i]),0]
    train = Vol[]
    test = Vol[np.searchsorted(sp500_list,sp500_permno),i]
    ind = np.all(train != 0, axis=1) & (test !=0)
    train = np.log(train[ind])
    test = np.log(test[ind])
    return train, test
'''
def catch(df,day=20160104,window=252):
    sp500_permno = sp500[(sp500[:,1]<=day)&(sp500[:,2]>=day),0]
    sp500_permno = [str(v) for v in sp500_permno]
    tt = (df.loc[:str(day),sp500_permno]).tail(window+1)
    tt = tt.dropna(axis = 1)
    tt = np.log(tt)
    train = tt.iloc[:-1]
    test = tt.iloc[-1]
    return train, test


def Factor_Har(train):
    N = train.shape[1]
    har = []
    
    for i in range(N):
        '''
        factor = pd.DataFrame(train.iloc[:,i].values,index=train.index,columns=['x'])
        for j in range(0,22):
                factor['x_lag'+str(j)] = factor['x'].shift(j)
        factor['x_w'] = factor.loc[:,'x_lag0':'x_lag4'].dropna().mean(axis=1)
        factor['x_m'] = factor.loc[:,'x_lag0':'x_lag21'].dropna().mean(axis=1)
        factor = factor.loc[:,['x','x_w','x_m']]
        '''
        fact = train.iloc[:,i]
        fact = pd.concat([fact,fact.rolling(5).mean(),fact.rolling(22).mean()],axis=1,join='inner')
        fact.columns = ['x','x_w','x_m']
        har.append(fact)
    return har



def Fit(train,test,Har_factor):
    N = train.shape[1]
    pred = np.zeros(N)
    residuals = []
    for i in range(N):
        y = train.iloc[:,i].shift(-1)
        factor = Har_factor[i]

        factor = sm.add_constant(factor)     
        yX = pd.concat([y,factor],axis=1,join='inner').dropna()
        model = sm.OLS(yX.iloc[:,0],yX.iloc[:,1:]).fit()
        #model.summary()
        pred[i] = model.predict(factor.iloc[-1,:])
        residuals.append(model.resid)
    mse = np.mean((pred-test)**2)
    return pred,residuals,mse

def main():
    # read data
    Vol = pd.read_csv("./Vol.csv",index_col=0)
    Vol.index = pd.to_datetime(Vol.index)
    global sp500
    sp500 = np.loadtxt('./sp500.txt',dtype="int64")
    uni_date = np.loadtxt("./uni_date.txt")
    
    T = 2000
    start = 2020
    #start = 5793
    #start = 6045
    MSE = pd.DataFrame(index = Vol.index[start:start+T],columns=['HAR'])
    ERROR = pd.DataFrame(index = Vol.index[start:start+T],columns=Vol.columns)
    #s = time.time()
    for t in range(T):
        global train, test, Har_factor
        train, test =  catch(df=Vol,day=int(uni_date[t+start]))
        
        Har_factor = Factor_Har(train)
        
        day = Vol.index[t+start]
        pred,resid,mse = Fit(method='HAR')
        MSE.loc[day,'HAR'] = mse 
        ERROR.loc[day,train.columns] = pred-test.values  
    #e = time.time()
    #print(e-s)
    MSE.to_csv('/home/zwma/volforecast/HAR_MSE_2020.csv')
    ERROR.to_csv('/home/zwma/volforecast/HAR_ERROR_2020.csv')
    
if __name__ == '__main__':
    main()

    




    
    