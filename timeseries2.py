import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from pandas.tools.plotting import autocorrelation_plot
from matplotlib import pyplot
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from sklearn.metrics import mean_squared_error
        
from statsmodels.tsa.stattools import adfuller        
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

import statsmodels.api as sm
from datetime import datetime
import os

os.chdir('C:\\Users\\yanran.zhou')

dateparse = lambda x: pd.datetime.strptime(x, '%m-%d-%y')

df = pd.read_csv('mpcptf.csv', parse_dates=['Month'],index_col=['Month'], date_parser=dateparse)
#print (data.head())

ts = df['Sales of shampoo over a three year period'] 

X = ts.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()