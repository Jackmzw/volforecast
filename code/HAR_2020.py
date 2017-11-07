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

'''
def SW(train, test, K=3):
    # PCA
    X = scale(train.T)
    pca = PCA(n_components=K)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    X1=pca.fit_transform(X)
    # Linear Regression
    X1 = np.concatenate((np.ones((len(X1),1)),X1),axis=1)
    X2 = X1[:-1,:]
    Y2 = train.T[1:,:]
    beta = np.linalg.solve(np.dot(X2.T,X2),np.dot(X2.T,Y2))
    pred_sw = np.dot(X1[-1,:],beta)
    mse = np.mean((pred_sw-test)**2)
    return pred_sw, mse
'''

def SW_predictor(train,test,predictor=0,K=3,lag = True):    
    N = train.shape[1]
    pred_sw = np.zeros(N)
    X = scale(train)
    pca = PCA(n_components=K)
    pca.fit(X)
    X1 = pd.DataFrame(pca.fit_transform(X),index=train.index)
    try:       
        factor = pd.concat([X1, predictor], axis=1, join='inner')
        factor = factor-factor.mean()
        if lag:
            for i in range(0,22):
                factor['VIX_lag'+str(i)] = factor['VIX'].shift(i)
            factor['VIX_w'] = factor.loc[:,'VIX_lag0':'VIX_lag4'].dropna().mean(axis=1)
            factor['VIX_m'] = factor.loc[:,'VIX_lag0':'VIX_lag21'].dropna().mean(axis=1)
            factor = factor.loc[:,[0,1,2,'VIX','VIX_w','VIX_m']]
    except:
        factor = X1  
    factor = sm.add_constant(factor)
    for i in range(N):
        y = train.iloc[:,i].shift(-1)
        yX = pd.concat([y,factor],axis=1,join='inner').dropna()
        model = sm.OLS(yX.iloc[:,0],yX.iloc[:,1:]).fit()
        #model.summary()
        pred_sw[i] = model.predict(factor.iloc[-1,:])
    mse = np.mean((pred_sw-test)**2)
    return pred_sw, mse


def SW_2(train, test, K=3):
    N = train.shape[1]
    pred_sw2 = np.zeros(N)
    dX = train.diff().dropna()
    dX = scale(dX)
    pca = PCA(n_components=K)
    pca.fit(dX)
    dX1= pd.DataFrame(pca.fit_transform(dX),index=train.index[1:])
    for i in range(N):
        y = train.iloc[:,i].shift(-1)
        factor = pd.concat([dX1,train.iloc[:,i]], axis=1, join='inner')
        factor = sm.add_constant(factor)
        yX = pd.concat([y,factor],axis=1,join='inner').dropna()
        model = sm.OLS(yX.iloc[:,0],yX.iloc[:,1:]).fit()
        #print(model.summary())
        pred_sw2[i] = model.predict(factor.iloc[-1,:])
    mse = np.mean((pred_sw2-test)**2)
    return pred_sw2, mse
        

'''
def HAR(train, test):
    N = np.size(train,0)
    L = np.size(train,1)
    pred_har = np.zeros(N)
    for i in range(N):
        vol = train[i,:]
        Y = vol[22:]
        X = np.zeros((L-22,4))
        for j in range(L-22):
            X[j,0] = 1
            X[j,1] = np.mean(vol[j:(j+22)])
            X[j,2] = np.mean(vol[(j+17):(j+22)])
            X[j,3] = vol[j+21]
        beta = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
        x = np.array([1,np.mean(vol[(L-22):L]),np.mean(vol[(L-5):L]),vol[-1]])
        pred_har[i] = np.sum(x*beta)
    mse = np.mean((pred_har-test)**2)
    return pred_har, mse
'''
def HAR_predictor(train,test,predictor=0,xlag=True,vixlag=True):    
    N = train.shape[1]
    pred_har = np.zeros(N)
    for i in range(N):
        y = train.iloc[:,i].shift(-1)
        factor = pd.DataFrame(train.iloc[:,i].values,index=train.index,columns=['x'])
        if xlag:
            for j in range(0,22):
                factor['x_lag'+str(j)] = factor['x'].shift(j)
            factor['x_w'] = factor.loc[:,'x_lag0':'x_lag4'].dropna().mean(axis=1)
            factor['x_m'] = factor.loc[:,'x_lag0':'x_lag21'].dropna().mean(axis=1)
            factor = factor.loc[:,['x','x_w','x_m']]
        try:
            factor = pd.concat([factor,predictor], axis=1, join='inner')
            if vixlag:
                for j in range(0,22):
                    factor['VIX_lag'+str(j)] = factor['VIX'].shift(j)
                factor['VIX_w'] = factor.loc[:,'VIX_lag0':'VIX_lag4'].dropna().mean(axis=1)
                factor['VIX_m'] = factor.loc[:,'VIX_lag0':'VIX_lag21'].dropna().mean(axis=1)
                factor = factor.loc[:,['x','x_w','x_m','VIX','VIX_w','VIX_m']].dropna(axis=1, how='all')
        except:
            pass
        factor = sm.add_constant(factor)
        yX = pd.concat([y,factor],axis=1,join='inner').dropna()
        model = sm.OLS(yX.iloc[:,0],yX.iloc[:,1:]).fit()
        #print(model.summary())
        pred_har[i] = model.predict(factor.iloc[-1,:])
    mse = np.mean((pred_har-test)**2)
    return pred_har, mse
'''
def AR1(train, test):
    N = np.size(train,0)
    L = np.size(train,1)
    pred_ar1 = np.zeros(N)
    for i in range(N):
        vol = train[i,:]
        Y = vol[1:]
        X = np.zeros((L-1,2))
        for j in range(L-1):
            X[j,0] = 1
            X[j,1] = vol[j]
        beta = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
        x = np.array([1,vol[-1]])
        pred_ar1[i] = np.sum(x*beta)
    mse = np.mean((pred_ar1-test)**2)
    return pred_ar1, mse
'''
def arima(order):
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

def Factor_SW(train,K = 15):
    X = scale(train)
    pca = PCA(n_components=K)
    pca.fit(X)
    X1 = pd.DataFrame(pca.fit_transform(X),index=train.index)
    return X1

def Factor_SW2(train,K = 15):
    dX = train.diff().dropna()
    dX = scale(dX)
    pca = PCA(n_components=K)
    pca.fit(dX)
    dX1= pd.DataFrame(pca.fit_transform(dX),index=train.index[1:])
    return dX1

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


def Factor_VIX(train,predictor):
    factor = predictor.loc[train.index]
    factor = factor-factor.mean()
    '''
    for j in range(0,22):
        factor['VIX_lag'+str(j)] = factor.iloc[:,0].shift(j)
    factor['VIX_w'] = factor.loc[:,'VIX_lag0':'VIX_lag4'].dropna().mean(axis=1)
    factor['VIX_m'] = factor.loc[:,'VIX_lag0':'VIX_lag21'].dropna().mean(axis=1)
    factor = factor.loc[:,['VIX','VIX_w','VIX_m']]
    '''
    factor = pd.concat([factor,factor.rolling(5).mean(),factor.rolling(22).mean()],axis=1,join='inner')
    factor.columns = ['VIX','VIX_w','VIX_m']
    return factor

def Fit(method = 'SW', K = 15):
    N = train.shape[1]
    pred = np.zeros(N)
    residuals = []
    for i in range(N):
        y = train.iloc[:,i].shift(-1)
        if method == 'SW':
            factor = SW_factor.iloc[:,:K]
        elif method == 'SW_m':
            factor = pd.concat([ train.iloc[:,i],SW2_factor.iloc[:,:K]],axis=1,join='inner').dropna()
        elif method == 'HAR':
            factor = Har_factor[i]
        elif method == 'AR(1)':
            factor = train.iloc[:,i]       
        elif method == 'VIX1':
            factor = VIX_factor.iloc[:,0]
        elif method == 'VIX2':
            factor = VIX_factor
        elif method == 'SW_VIX1':
            factor = pd.concat([SW_factor.iloc[:,:K],VIX_factor.iloc[:,0]],axis=1,join='inner').dropna()
        elif method == 'SW_VIX2':
            factor = pd.concat([SW_factor.iloc[:,:K],VIX_factor],axis=1,join='inner').dropna()
        elif method == 'HAR_VIX1':
            factor = pd.concat([Har_factor[i],VIX_factor.iloc[:,0]],axis=1,join='inner').dropna()
        elif method == 'HAR_VIX2':
            factor = pd.concat([Har_factor[i],VIX_factor],axis=1,join='inner').dropna()
        elif method == 'HAR_SW':
            factor = pd.concat([Har_factor[i],SW_factor.iloc[:,:K]],axis=1,join='inner').dropna()
        else:
            raise Exception('No method match!')
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

    




    
    