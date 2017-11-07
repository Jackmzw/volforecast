#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:09:03 2017

@author: Jack
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import statsmodels.formula.api as sm
#from pandas.stats.api import ols
from pandas.tools.plotting import autocorrelation_plot

url = 'http://dachxiu.chicagobooth.edu/portal.php?api=vol&permno=84398'
with urllib.request.urlopen(url) as f: data = f.read().decode('ascii')
rows = data.split("\n")[:-1]
date = [int(row.split(",")[0]) for row in rows]
date = [pd.to_datetime(str(d),format='%Y%m%d') for d in date]
vol = [np.sqrt(float(row.split(",")[1])) for row in rows]
SP500 = pd.DataFrame(vol,index=date,columns=['x'])
SP500[SP500==0] = np.nan
SP500 = SP500.dropna()
SP500 = np.log(SP500)

vix = pd.read_csv('./vixcurrent.csv',index_col=0)
predictor = pd.DataFrame(vix['VIX Close'].values,index = pd.to_datetime(vix.index),columns=['VIX'])

T = 400
pred_har = pd.Series(index=SP500.index[5709:5709+T])
pred_vix1 = pd.Series(index=SP500.index[5709:5709+T])
pred_vix2 = pd.Series(index=SP500.index[5709:5709+T])

for t in range(T):
    
    day = SP500.index[t+5709]
    #### HAR for S&P500
    tt = SP500.loc[:day].tail(201)
    tt = tt
    test = tt.iloc[-1]
    train = tt.iloc[:-1]
       
    y = train.shift(-1)
    factor = train
    factor = sm.add_constant(factor)
    for j in range(0,22):
        factor['x_lag'+str(j)] = factor.loc[:,'x'].shift(j)
    factor['x_w'] = factor.loc[:,'x_lag0':'x_lag4'].dropna().mean(axis=1)
    factor['x_m'] = factor.loc[:,'x_lag0':'x_lag21'].dropna().mean(axis=1)
    factor = factor.loc[:,['const','x','x_w','x_m']].dropna(axis=1, how='all')
    
    
    yX = pd.concat([y,factor],axis=1,join='inner').dropna()
    model = sm.OLS(yX.iloc[:,0],yX.iloc[:,1:]).fit()
    #model.summary()
    pred_har[t] = model.predict(factor.iloc[-1,:])
    
    
    df = pd.concat([SP500,predictor],axis=1,join='inner')
    tt = df.loc[:day].tail(201)
    test = tt.iloc[-1]
    train = tt.iloc[:-1]
    
    y = train.loc[:,'x'].shift(-1)
    factor = train.loc[:,'VIX']
    factor = factor-factor.mean()
    factor = sm.add_constant(factor)
    yX = pd.concat([y,factor],axis=1,join='inner').dropna()
    model = sm.OLS(yX.iloc[:,0],yX.iloc[:,1:]).fit()
    #model.summary()
    pred_vix1[t] = model.predict(factor.iloc[-1,:])
    
    for i in range(0,22):
        factor['VIX_lag'+str(i)] = factor.loc[:,'VIX'].shift(i)
    factor['VIX_w'] = factor.loc[:,'VIX_lag0':'VIX_lag4'].dropna().mean(axis=1)
    factor['VIX_m'] = factor.loc[:,'VIX_lag0':'VIX_lag21'].dropna().mean(axis=1)
    factor = factor.loc[:,['const','VIX','VIX_w','VIX_m']]
    
    yX = pd.concat([y,factor],axis=1,join='inner').dropna()
    model = sm.OLS(yX.iloc[:,0],yX.iloc[:,1:]).fit()
    #model.summary()
    pred_vix2[t] = model.predict(factor.iloc[-1,:])
    
test = SP500[5709:5709+T]

p  = pd.concat([test,pred_har,pred_vix1,pred_vix2],axis=1)
p.columns = ['SP500','HAR','VIX1','VIX2']
p.plot()
plt.show()


train.diff().plot()
autocorrelation_plot(train.diff().dropna())
autocorrelation_plot(train.dropna())
train = SP500.loc["20160104":,'x']
from statsmodels.tsa.stattools import adfuller
dftest = adfuller(train)
dftest = adfuller(train.diff().dropna(), autolag='AIC')

from statsmodels.tsa.arima_model import ARIMA

def objfunc(order, train):
    fit = ARIMA(train, order).fit()
    return fit.aic

from scipy.optimize import brute
grid = (slice(1, 3, 1), slice(1, 3, 1), slice(1, 3, 1))
brute(objfunc, grid, args=(train,), finish=None)

fit = ARIMA(train,order=(1,0,2)).fit()
fit.bic
fit = ARIMA(train,order=(1,1,1)).fit()
fit.bic
fit.forecast()

T = 100
pred_arima = pd.DataFrame(index=SP500.index[5709:5709+T],columns=['(1,1,1)'])
for t in range(T):  
    day = SP500.index[t+5709]
    #### HAR for S&P500
    tt = SP500.loc[:day,'x'].tail(253)
    tt = tt
    test = tt.iloc[-1]
    train = tt.iloc[:-1]
    try:
        model = ARIMA(train,order=(1,1,1))
        fit = model.fit()
        pred_arima.loc[day,'(1,1,1)'] = fit.forecast()[0][0]
    except:
        fit = ARIMA(train,order=(1,1,0)).fit(disp=0)
        pass
    
from statsmodels.tsa.arima_model import ARIMA
T = 400
pdq = [(1,0,0),(1,0,1),(1,0,2),(2,0,0),(2,0,1),(2,0,2),(1,1,0),(1,1,1)]
pred_arima = pd.DataFrame(index=SP500.index[5709:5709+T],columns=[str(v) for v in pdq])

for t in range(T):  
    day = SP500.index[t+5709]
    #### HAR for S&P500
    tt = SP500.loc[:day,'x'].tail(201)
    tt = tt
    test = tt.iloc[-1]
    train = tt.iloc[:-1]
    for order in pdq:
        try:
            model = ARIMA(train,order)
            fit = model.fit(disp=-1)
            pred_arima.loc[day,str(order)] = fit.forecast()[0][0]
        except:
            pass

for order in pdq:
    s = pred_arima[str(order)]
    print(len(s.dropna()))
   

test = SP500.iloc[5709:5709+T,0]
mse = np.mean((test.values - pred_arima['(1,1,1)'].values)**2)






