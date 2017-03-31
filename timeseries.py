import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
        
from statsmodels.tsa.stattools import adfuller        
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm
from datetime import datetime
import os

os.chdir('C:\\Users\\yanran.zhou')

#### reading data######

#data = pd.read_csv('PTF.csv')
dateparse = lambda x: pd.datetime.strptime(x, '%m-%d-%y')

df = pd.read_csv('PTF.csv', parse_dates=['date'],index_col=['date'], date_parser=dateparse)
#print (data.head())

ts = df['PTF'] 
#plt.plot(ts)
#plt.show()

#Determing rolling statistics
rolmean = ts.rolling(center=False,window=12).mean()
rolstd = ts.rolling(center=False,window=12).std()

#Plot rolling statistics:
#orig = plt.plot(ts, color='blue',label='Original')
#mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#plt.legend(loc='best')
#plt.title('Rolling Mean & Standard Deviation')
#plt.show(block=False)        
#plt.show()


#Perform Dickey-Fuller test:
#print 'Results of Dickey-Fuller Test:'
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)

#### test results indicate the current ts is not stationary###

ts_log = np.log(ts)
#plt.plot(ts_log)
#plt.show()

moving_avg = pd.rolling_mean(ts_log,12)
#plt.plot(ts_log)
#plt.plot(moving_avg, color='red')
#plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)

dftest2 = adfuller(ts_log_moving_avg_diff, autolag='AIC')
dfoutput2 = pd.Series(dftest2[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest2[4].items():
    dfoutput2['Critical Value (%s)'%key] = value

print (dfoutput2)

expwighted_avg = pd.ewma(ts_log, halflife=12)
ts_log_ewma_diff = ts_log - expwighted_avg


dftest3 = adfuller(ts_log_ewma_diff, autolag='AIC')
dfoutput3 = pd.Series(dftest3[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest3[4].items():
    dfoutput3['Critical Value (%s)'%key] = value

print(dfoutput3)

##### eliminate trend and seasonality####

ts_log_diff = ts_log - ts_log.shift()
#plt.plot(ts_log_diff)
#plt.show()

ts_log_diff.dropna(inplace=True)
dftest4 = adfuller(ts_log_diff, autolag='AIC')
dfoutput4 = pd.Series(dftest4[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest4[4].items():
    dfoutput4['Critical Value (%s)'%key] = value

print(dfoutput4)

rolmean2 = ts_log_diff.rolling(center=False,window=12).mean()
rolstd2 = ts_log_diff.rolling(center=False,window=12).std()

#orig2 = plt.plot(ts_log_diff, color='blue',label='Original')
#mean2 = plt.plot(rolmean2, color='red', label='Rolling Mean')
#std2 = plt.plot(rolstd2, color='black', label = 'Rolling Std')
#plt.legend(loc='best')
#plt.title('Rolling Mean & Standard Deviation')
#plt.show(block=False)        
#plt.show()


######### decomposition ##################

decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

#plt.subplot(411)
#plt.plot(ts_log, label='Original')
#plt.legend(loc='best')
#plt.subplot(412)
#plt.plot(trend, label='Trend')
#plt.legend(loc='best')
#plt.subplot(413)
#plt.plot(seasonal,label='Seasonality')
#plt.legend(loc='best')
#plt.subplot(414)
#plt.plot(residual, label='Residuals')
#plt.legend(loc='best')
#plt.tight_layout()
#plt.show()

#### use residual to check stationary########3

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)


dftest5 = adfuller(ts_log_decompose, autolag='AIC')
dfoutput5 = pd.Series(dftest5[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest5[4].items():
    dfoutput5['Critical Value (%s)'%key] = value

print(dfoutput5)

#### how to read the output:

##### The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset).
#### A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), 
######## otherwise a p-value above the threshold suggests we accept the null hypothesis (non-stationary).

############ now that we have a stationary time series, we can deploy ARIMA to forcast future trend############

###### x(t) = alpha *  x(t – 1) + error (t)  is Auto-Regressive Time Series Model formula #######
##### definition : the current data is a product of past data (t-1) * some constant plus the noise ############

###### x(t) = beta *  error(t-1) + error (t) is Moving Average Time Series Model formula ########
#### definition: the current data is a product of previous data (t-1) 's noise  * some constant plus current data's noise ########

######## DIFFERENCE BETWEEN AR and MA ###########

#### noise/shock in MA model would quickly vanish whereas noise in AR model will have longer lasting effect ############

##### The primary difference between an AR and MA model is based on the correlation between time series objects at different time points. ###########

########## compare between MA and AR #####

#### Autocorrelation Function (ACF): It is a measure of the correlation between the the TS with a lagged version of itself. 
#####For instance at lag 5, ACF would compare series at time instant ‘t1’…’t2’ with series at instant ‘t1-5’…’t2-5’ (t1-5 and t2 being end points).

#####Partial Autocorrelation Function (PACF): This measures the correlation between the TS with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons. 
######Eg at lag 5, it will check the correlation but remove the effects already explained by lags 1 to 4.

lag_acf = acf(ts_log_decompose, nlags=10)
lag_pacf = pacf(ts_log_decompose, nlags=10, method='ols')

##### There's no fixed rule on choosing lags. 
###It is a function of the noise in the time-series. I would show at least until no data-point crosses a confidence interval or n10n10, whichever comes first.




print(lag_acf)
print(lag_pacf)

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

######### AR model ########## 

model = ARIMA(ts_log, order=(1, 1, 0))  
results_AR = model.fit(disp=-1)  
#plt.plot(ts_log_diff)
#plt.plot(results_AR.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))
#plt.show()

print(sum((results_AR.fittedvalues-ts_log_diff)**2))

########## MA model ########

model2 = ARIMA(ts_log, order=(0, 1, 1))  
results_MA = model2.fit(disp=-1)  
#plt.plot(ts_log_diff)
#plt.plot(results_MA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))
#plt.show()

print(sum((results_MA.fittedvalues-ts_log_diff)**2))

######## combined model ##########

model3 = ARIMA(ts_log, order=(0, 1, 2))  
results_ARIMA = model3.fit(disp=-1)  
#plt.plot(ts_log_diff)
#plt.plot(results_ARIMA.fittedvalues, color='red')
#plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
#plt.show()

print(results_ARIMA.summary().tables[1])


print(sum((results_ARIMA.fittedvalues-ts_log_diff)**2))


####### combined model has the smallest RSS score #############

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
#print (predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#print(predictions_ARIMA_log.head())

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()

