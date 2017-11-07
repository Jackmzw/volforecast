#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:10:45 2017

@author: Jack
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import pyplot, pylab
from sklearn.preprocessing import scale
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima_model import ARIMA
import time

#from methods import catch
from statsmodels.tsa.stattools import acf
import seaborn as sns


def catch(df,day=20160104,window=252):
    sp500_permno = sp500[(sp500[:,1]<=day)&(sp500[:,2]>=day),0]
    sp500_permno = [str(v) for v in sp500_permno]
    tt = (df.loc[:str(day),sp500_permno]).tail(window+1)
    tt = tt.dropna(axis = 1)
    tt = np.log(tt)
    train = tt.iloc[:-1]
    test = tt.iloc[-1]
    return train, test

def filteredX(X, D):
    '''
    Given D, compute the filtered X 
    Args:
        X: T*N matrix
        D: N*m matrix
    Return:
        Y: the filtered X, T*N matrix 
    '''
    T, N = X.shape
    m = D.shape[1]
    Y = np.zeros((T,N))
    for i in range(N):
        x = np.zeros((T, m+1))
        x[:,0] = X[:,i]
        for j in range(1,m+1):
            x[j:,j] = x[:T-j,0]
        d = np.concatenate([[1],-D[i]])
        y = np.dot(x,d)
        Y[:,i] = y
    return Y


def Factor(Y, r):
    '''
    Given the factor num r, decomposite Y to factor F
    
    '''
    pca = PCA(n_components=r)
    pca.fit(Y)
    F = pca.fit_transform(Y)
    return F

def est_param(X, F, m):
    T, N = X.shape 
    r = F.shape[1]
    Lambda = np.zeros((N,r))
    D = np.zeros((N, m))
    Error = np.zeros((N, T))
    for i in range(N):
        y = X[:,i]
        x = np.zeros((T, m+r))
        for j in range(m):
            x[j+1:,j] = y[:T-j-1]
        x[:,m:] = F
        param = np.linalg.solve(np.dot(x.T,x),np.dot(x.T,y))
        D[i,:] = param[:m]
        Lambda[i,:] = param[m:]
        Error[i,:] = y - np.dot(x,param)
    v = np.sum(Error**2)/T/N  ##objective function 
    return Lambda, D, v, Error
 
def dfm_fit(X, m = 4, r = 3, max_iter = 50, eps = 10**(-4)):
    
    T, N = X.shape
    D = np.zeros((N,m))
    V = []   
    for i in range(max_iter):
        #print(i)
    
        Y = filteredX(X, D)    
        F = Factor(Y, r)
        Lambda, D_new, v, resid = est_param(X, F, m)
        V.append(v)        
        e = np.max(np.abs(D_new-D))         
        D = D_new

        if e < eps:
            break
    return F, Lambda, D, V, r, resid
    
#F, Lambda, D, V , r = dfm_fit(X, m = 4, r = 3) 
#print(V[-1])   

def dfm(X, m = 4, R = range(1,6)):
    T, N = X.shape 
    results = []
    L = len(R)
    ICp1 = np.zeros(L)
    ICp2 = np.zeros(L)
    for l in range(L):
        r = R[l]
        result = dfm_fit(X,m,r)
        v = result[3][-1]
        #print(v)
        ICp1[l] = np.log(v) + r*(N+T)/N/T*np.log(N*T/(N+T))
        ICp2[l] = np.log(v) + r*(N+T)/N/T*np.log(min(N,T))
        results.append(result)
    flag1 = np.argmin(ICp1)
    flag2 = np.argmin(ICp2)
    ICp = np.concatenate((ICp1.reshape(L,1),ICp2.reshape(L,1)),axis=1) 
    best_dfm = results[max(flag1,flag2)]
    r = R[max(flag1,flag2)]
    return best_dfm, ICp, r, results, [R[flag1],R[flag2]]

def forecast_VAR(F, lag=2):
    r = len(F[0])
    if r > 1: 
        model = VAR(F)
        fit = model.fit(lag,trend='nc')   ##### should use trend='nc'!!!!!!!!
        pred = np.zeros(r)
        for i in range(lag):
            pred = pred + np.dot(fit.params[i*r:(i+1)*r].T, F[-(i+1)])    
    else:
        model = ARIMA(F[:,0],(lag,0,0))
        fit = model.fit()
        pred = fit.forecast()[0][0]
    return pred

def forecast(X, pred_F, Lambda, D):
    T, N = X.shape 
    m = len(D[0])
    pred = np.zeros(N)
    for i in range(N):
        pred[i] = np.sum(Lambda[i]*pred_F) + np.sum(D[i]*X[-1:-1-m:-1,i])
    return pred

def main():
    Vol = pd.read_csv("./Vol.csv",index_col=0)
    Vol.index = pd.to_datetime(Vol.index)
    #RV5 = pd.read_csv("./RV5.csv",index_col=0)
    #RV5.index = pd.to_datetime(RV5.index)
    global sp500
    sp500 = np.loadtxt('./sp500.txt',dtype="int64")
    uni_date = np.loadtxt("./uni_date.txt")
    
    T = 1000
    #start = 3527
    #start = 5793
    start = 2020
    
    MSE = pd.DataFrame(index = Vol.index[start:start+T],columns=['DFM','ICp1','ICp2'])
    ERROR = pd.DataFrame(index = Vol.index[start:start+T],columns=Vol.columns)
    
    #s = time.time()
    for t in range(T):
        #print(t)
        train, test =  catch(df=Vol,day=int(uni_date[t+start]))
    
        X = train.values
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = scale(X)
        best_dfm, ICp, r, results, ICp_r = dfm(X, m=6) 
        
        F = best_dfm[0]
        Lambda = best_dfm[1]
        D = best_dfm[2]
       
        pred_F = forecast_VAR(F, lag = 2)
        
        pred = forecast(X,pred_F,Lambda, D)
        
        pred = pred*std+mean
        
        #plt.plot(test.values)
        #plt.plot(pred)
        #plt.show()
        day = Vol.index[t+start]
        mse = np.mean((pred-test.values)**2)
        MSE.loc[day] = [mse, ICp_r[0], ICp_r[1]]
        ERROR.loc[day,train.columns] = pred-test.values
    MSE.to_csv('/home/zwma/volforecast/DFM_MSE_2020.csv')
    ERROR.to_csv('/home/zwma/volforecast/DFM_ERROR_2020.csv')
    
    #e = time.time()
    #print(e-s)

if __name__ == '__main__':
    main()
    

    
    