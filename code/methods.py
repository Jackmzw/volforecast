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
def catch(df,day=20160104,window=200):
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
    RV5 = pd.read_csv("./RV5.csv",index_col=0)
    RV5.index = pd.to_datetime(RV5.index)
    RV15 = pd.read_csv("./RV15.csv",index_col=0)
    RV15.index = pd.to_datetime(RV15.index)
    global sp500
    sp500 = np.loadtxt('./sp500.txt',dtype="int64")
    uni_date = np.loadtxt("./uni_date.txt")
    vix = pd.read_csv('./vixcurrent.csv',index_col=0)
    predictor = pd.DataFrame(vix['VIX Close'].values,index = pd.to_datetime(vix.index),columns=['VIX'])
    
    
    T = 200
    start = 5797
    #start = 5793
    #start = 6045
    MSE = pd.DataFrame(index = Vol.index[start:start+T],
                       columns=['SW','SW_m','HAR','Random','AR(1)','VIX1','VIX2',
                                'SW_VIX1','SW_VIX2','HAR_VIX1','HAR_VIX2','HAR_SW',
                                'ARIMA(1,0,1)','ARIMA(1,1,1)'])
    starttime = time.time()
    for t in range(T):
        global train, test, SW_factor, SW2_factor, Har_factor, VIX_factor 
        train, test =  catch(df=Vol,day=int(uni_date[t+start]))
        
        SW_factor = Factor_SW(train,K=15)
        SW2_factor = Factor_SW2(train,K = 15)
        Har_factor = Factor_Har(train)
        VIX_factor = Factor_VIX(train,predictor)
        
        day = Vol.index[t+start]
        pred,resid,mse = Fit(method='SW',K = 15)
        MSE.loc[day,'SW'] = mse
        pred,resid,mse = Fit(method='SW_m',K = 10)
        MSE.loc[day,'SW_m'] = mse
        pred,resid,mse = Fit(method='HAR')
        MSE.loc[day,'HAR'] = mse 
        pred,resid,mse = Fit(method='AR(1)')
        MSE.loc[day,'AR(1)'] = mse 
        pred_random = train.iloc[-1,:]
        mse = np.mean((pred_random-test)**2)
        MSE.loc[day,'Random'] = mse 
        pred,resid,mse = Fit(method='VIX1')
        MSE.loc[day,'VIX1'] = mse 
        pred,resid,mse = Fit(method='VIX2')
        MSE.loc[day,'VIX2'] = mse 
        pred,resid,mse = Fit(method='SW_VIX1')
        MSE.loc[day,'SW_VIX1'] = mse 
        pred,resid,mse = Fit(method='SW_VIX2')
        MSE.loc[day,'SW_VIX2'] = mse 
        pred,resid,mse = Fit(method='HAR_VIX1')
        MSE.loc[day,'HAR_VIX1'] = mse 
        pred,resid,mse = Fit(method='HAR_VIX2')
        MSE.loc[day,'HAR_VIX2'] = mse
        pred,resid,mse = Fit(method='HAR_SW')
        MSE.loc[day,'HAR_SW'] = mse
        '''
        pred,mse = arima(order=(1,0,1))
        MSE.loc[day,'ARIMA(1,0,1)'] = mse
        pred,mse = arima(order=(1,1,1))
        MSE.loc[day,'ARIMA(1,1,1)'] = mse
        '''
        print(t,uni_date[t+start])
        endtime = time.time()
        print(endtime-starttime)
    print(MSE.mean())
    
    '''
    for t in range(T):
        train, test =  catch(df=Vol,day=int(uni_date[t+start]))
        pred_sw, mse = SW_predictor(train,test,K=15)
        MSE.loc[Vol.index[t+start],'SW'] = mse
        pred_sw2, mse = SW_2(train,test,K=3)
        MSE.loc[Vol.index[t+start],'SW_m'] = mse
        pred_har, mse = HAR_predictor(train,test)
        MSE.loc[Vol.index[t+start],'HAR'] = mse        
        pred_ar1, mse = HAR_predictor(train,test,xlag=False)
        MSE.loc[Vol.index[t+start],'AR1'] = mse 
        pred_random = train.iloc[-1,:]
        mse = np.mean((pred_random-test)**2)
        MSE.loc[Vol.index[t+start],'Random'] = mse 
        pred_sw_vix1, mse = SW_predictor(train,test,predictor,K=15,lag=False)
        MSE.loc[Vol.index[t+start],'SW_VIX1'] = mse 
        pred_sw_vix2, mse = SW_predictor(train,test,predictor,K=10)
        MSE.loc[Vol.index[t+start],'SW_VIX2'] = mse
        pred_har_vix1, mse = HAR_predictor(train,test,predictor,vixlag=False)
        MSE.loc[Vol.index[t+start],'HAR_VIX1'] = mse
        pred_har_vix2, mse = HAR_predictor(train,test,predictor)
        MSE.loc[Vol.index[t+start],'HAR_VIX2'] = mse 
        print(t,uni_date[t+start])
    print(MSE.mean())
    
    
    AR1 = pd.Series(index = RV5.index[start:start+T])
    for t in range(T):
        train, test =  catch(df=Vol,day=int(uni_date[t+start]))
        AR1[t] = np.corrcoef(train.iloc[1:,100],train.iloc[:-1,100])[0,1]
    '''
    #np.corrcoef(vix1,X1[:,0])
    #np.corrcoef(vix1[1:],x1[:,0])
    
    
    
if __name__ == '__main__':
    main()

'''
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import seaborn as sns
from scipy.stats.mstats import zscore

days = [20150624,20160112,20170323]

for day in days:
    
    print(str(day))
    train, test = catch(df=Vol,day=day)
    SW_factor = Factor_SW(train,K=15)
    Har_factor = Factor_Har(train)
    VIX_factor = Factor_VIX(train,predictor)
    
    pred_sw,resid_sw,mse_sw = Fit(method='SW',K = 15)
    pred_sw3,resid_sw3,mse_sw3 = Fit(method='SW',K = 3)
    pred_har,resid_har,mse_har = Fit(method='HAR')
    pred_vix,resid_vix,mse_vix = Fit(method='VIX1')
    pred_harvix, resid_harvix, mse_harvix = Fit(method='HAR_VIX1')
    
    
    resid_sw = np.array([res.values for res in resid_sw])
    corr_sw = np.corrcoef(resid_sw)
    w, v = np.linalg.eigh(corr_sw)
    print('SW', w[-1])
    
    resid_sw3 = np.array([res.values for res in resid_sw3])
    corr_sw3 = np.corrcoef(resid_sw3)
    w, v = np.linalg.eigh(corr_sw3)
    print('SW3', w[-1])
    
    resid_har = np.array([res.values for res in resid_har])
    corr_har = np.corrcoef(resid_har)
    w, v = np.linalg.eigh(corr_har)
    print('HAR', w[-1])
    
    resid_harvix = np.array([res.values for res in resid_harvix])
    corr_harvix = np.corrcoef(resid_harvix)
    w, v = np.linalg.eigh(corr_harvix)
    print('HAR-VIX1', w[-1])
    
    resid_vix = np.array([res.values for res in resid_vix])
    corr_vix = np.corrcoef(resid_vix)
    w, v = np.linalg.eigh(corr_vix)
    print('VIX1', w[-1])
    
    
    
    
    #sns.heatmap(corr1)   
    
    ACF_har = np.array([acf(res) for res in resid_har])
    plt.hist(zscore(ACF_har[:,1]),50,normed=1)
    plt.title('zscore HAR'+str(day)+' MSE='+str(np.round(mse_har,5)))
    plt.show()
    
    plt.hist(ACF_har[:,1],50,normed=1)
    plt.title('Histogram of first lag ACF of HAR residuals for '+str(day)+' MSE='+str(np.round(mse_har,5)))
    plt.show()
    
    ACF_har_mean = np.mean(ACF_har,axis=0)
    plt.plot(ACF_har_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of HAR residuals for '+str(day)+' MSE='+str(np.round(mse_har,5)))
    plt.show()
    
    
    ACF_sw = np.array([acf(res) for res in resid_sw])
    
    ACF_sw_mean = np.mean(ACF_sw,axis=0)
    plt.plot(ACF_sw_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of SW (K = 15) residuals for '+str(day)+' MSE='+str(np.round(mse_sw,5)))
    plt.show()
    
    
    
    ACF_sw3 = np.array([acf(res) for res in resid_sw3])
    ACF_sw3_mean = np.mean(ACF_sw3,axis=0)
    plt.plot(ACF_sw3_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of SW (K = 3) residuals for '+str(day)+' MSE='+str(np.round(mse_sw3,5)))
    plt.show()
    
    
    ACF_vix = np.array([acf(res) for res in resid_vix])
    ACF_vix_mean = np.mean(ACF_vix,axis=0)
    plt.plot(ACF_vix_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of VIX1 residuals for '+str(day)+' MSE='+str(np.round(mse_vix,5)))
    plt.show()
    
    
    ACF_harvix = np.array([acf(res) for res in resid_harvix])
    plt.hist(zscore(ACF_harvix[:,1]),50,normed=1)
    plt.title('zscore HAR-VIX1'+str(day)+' MSE='+str(np.round(mse_harvix,5)))
    plt.show()
    
    plt.hist(ACF_harvix[:,1],50,normed=1)
    plt.title('Histogram of first lag ACF of HAR-VIX1 residuals for '+str(day)+' MSE='+str(np.round(mse_har,5)))
    plt.show()
    
    ACF_harvix_mean = np.mean(ACF_harvix,axis=0)
    plt.plot(ACF_harvix_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of HAR-VIX1 residuals for '+str(day)+' MSE='+str(np.round(mse_harvix,5)))
    plt.show()
    

days = [20150624,20160112,20170323]

for day in days:
    
    train, test = catch(df=Vol,day=day)
    SW_factor = Factor_SW(train,K=15)
    Har_factor = Factor_Har(train)
    VIX_factor = Factor_VIX(train,predictor)
    
    #pred_sw,resid_sw,mse_sw = Fit(method='SW',K = 15)
    #pred_sw3,resid_sw3,mse_sw3 = Fit(method='SW',K = 3)
    #pred_har,resid_har,mse_har = Fit(method='HAR')
    pred_vix,resid_vix,mse_vix = Fit(method='VIX1')
    #pred_harvix, resid_harvix, mse_harvix = Fit(method='HAR_VIX1')
       
    #sns.heatmap(corr1)   
    
    ACF = np.array([acf(res) for res in resid_vix])
    plt.hist(ACF[:,1]*np.sqrt(199),50,normed=1)
    plt.title('zscore VIX1 '+str(day)+' MSE='+str(np.round(mse_vix,5)))
    plt.show()
    
    
    plt.hist(ACF_har[:,1],50,normed=1)
    plt.title('Histogram of first lag ACF of HAR residuals for '+str(day)+' MSE='+str(np.round(mse_har,5)))
    plt.show()
    
    ACF_har_mean = np.mean(ACF_har,axis=0)
    plt.plot(ACF_har_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of HAR residuals for '+str(day)+' MSE='+str(np.round(mse_har,5)))
    plt.show()
    
    
    ACF_sw = np.array([acf(res) for res in resid_sw])
    
    ACF_sw_mean = np.mean(ACF_sw,axis=0)
    plt.plot(ACF_sw_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of SW (K = 15) residuals for '+str(day)+' MSE='+str(np.round(mse_sw,5)))
    plt.show()
    
    
    
    ACF_sw3 = np.array([acf(res) for res in resid_sw3])
    ACF_sw3_mean = np.mean(ACF_sw3,axis=0)
    plt.plot(ACF_sw3_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of SW (K = 3) residuals for '+str(day)+' MSE='+str(np.round(mse_sw3,5)))
    plt.show()
    
    
    ACF_vix = np.array([acf(res) for res in resid_vix])
    ACF_vix_mean = np.mean(ACF_vix,axis=0)
    plt.plot(ACF_vix_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of VIX1 residuals for '+str(day)+' MSE='+str(np.round(mse_vix,5)))
    plt.show()
    
    
    ACF_harvix = np.array([acf(res) for res in resid_harvix])
    plt.hist(zscore(ACF_harvix[:,1]),50,normed=1)
    plt.title('zscore HAR-VIX1'+str(day)+' MSE='+str(np.round(mse_harvix,5)))
    plt.show()
    
    plt.hist(ACF_harvix[:,1],50,normed=1)
    plt.title('Histogram of first lag ACF of HAR-VIX1 residuals for '+str(day)+' MSE='+str(np.round(mse_har,5)))
    plt.show()
    
    ACF_harvix_mean = np.mean(ACF_harvix,axis=0)
    plt.plot(ACF_harvix_mean,'o')
    plt.axhline(0)
    plt.title('Mean ACF of HAR-VIX1 residuals for '+str(day)+' MSE='+str(np.round(mse_harvix,5)))
    plt.show()
    '''


    
    #resid2 = np.array([res.values for res in resid_har])
    #corr2 = np.corrcoef(resid2)
    #sns.heatmap(corr2)


    
    