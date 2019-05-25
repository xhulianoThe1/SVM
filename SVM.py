from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# for timeseries analysis
from statsmodels.graphics.tsaplots import plot_acf

# these two lines allow me to format dates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# sources:
# 1. https://www.learndatasci.com/tutorials/python-finance-part-yahoo-finance-api-pandas-matplotlib/
# 2. https://ntguardian.wordpress.com/2018/07/17/stock-data-analysis-python-v2/
#       this one has a function to generate candlestick plots (build from scratch)

# SVM paper on similar topic
# https://www.cs.princeton.edu/sites/default/files/uploads/saahil_madge.pdf


# global parameters (fixed)
start_date = '2014-03-27'
end_date   = '2019-03-27'

TRAIN_START = '2015-01-01'
TRAIN_END   = '2016-12-31'
TEST_START  = '2017-01-01'
TEST_END    = '2018-12-31'

def getData(ticker, type) :

    name_event      = type + "_Event"
    name_momentum   = type + "_M" # Momentum
    name_volatility = type + "_V" # Volatility

    # retrieve data using IEX API
    price_data = data.DataReader(ticker, 'iex', start_date, end_date)

    # convert closing price to returns
    price_data['return'] = price_data['close'].pct_change()

    # if returns are positive, event = 1 (otherwise event = 0)
    price_data.loc[price_data['return'] <  0, name_event] = 0
    price_data.loc[price_data['return'] >= 0, name_event] = 1

    # calculate rolling momentum / volatility (one month ~ 20 trading days)
    price_data[name_momentum]   = price_data[name_event].rolling(20).mean()
    price_data[name_volatility] = price_data['return'].rolling(20).std()

    # convert date from index to DATE format (for easy plotting)
    price_data.index = pd.to_datetime(price_data.index, format='%Y-%m-%d')

    if (type == 'stock') :
        # subset of closing price (lets keep the first prototype simple - only daily observations)
        price_data = price_data[[name_event, name_momentum, name_volatility]]
    else :
        price_data = price_data[[name_momentum, name_volatility]]

    return price_data

def runSVM(ticker, sector_etf) :

    # pull historical data from API
    stock  = getData(ticker , 'stock' )
    sector = getData(sector_etf, 'sector')
    market = getData('SPY' , 'market')

    # generate datasets for SVM implementation
    full_data     = pd.concat([stock, sector, market], axis = 1)
    training_data = full_data.loc[TRAIN_START:TRAIN_END]
    testing_Data  = full_data.loc[TEST_START :TEST_END ]

    y_train = training_data['stock_Event']
    X_train = training_data[['stock_M','stock_V','sector_M','sector_V','market_M','market_V']]

    y_true  = testing_Data['stock_Event']
    X_test  = testing_Data[['stock_M','stock_V','sector_M','sector_V','market_M','market_V']]

    svc = svm.SVC(kernel = 'linear')
    Cs = range(1, 20)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
    clf.fit(X_train, y_train)

    y_predic = clf.predict(X_test)

    y_test = pd.DataFrame(y_predic)
    y_true = pd.DataFrame(y_true)
    y_true = y_true.reset_index(drop=True)
    verify_data = pd.concat([y_true, y_test], axis=1)
    verify_data.columns = ['True_Event', 'Prediction']
    verify_data.loc[verify_data['True_Event'] != verify_data['Prediction'], 'Correct'] = 0
    verify_data.loc[verify_data['True_Event'] == verify_data['Prediction'], 'Correct'] = 1
    success_rate = verify_data['Correct'].mean() * 100
    print(ticker, 'correct prediction', round(success_rate, 2), "%")

runSVM('FB'   ,'XLK') # Facebook, Telecom
runSVM('AAPL' ,'XLK') # Apple, Telecom
runSVM('AMZN', 'XLY') # Amazon, Consumer Discretionary
runSVM('NFLX', 'XLY') # Netflix, Consumer Discretionary
runSVM('GOOGL','XLK') # Google, Telecom
runSVM('TSLA' ,'XLY') # Tesla, Consumer Discretionary


# def plotTS (stock_data, stock_name) :
#
#     # Calculate the 20 and 100 days moving averages of the closing prices
#     week_roll  = stock_data.rolling(window=5).mean()
#     month_roll = stock_data.rolling(window=20).mean()
#
#     fig, ax = plt.subplots()
#
#     ax.plot(stock_data.index, stock_data, label='time series')
#     ax.plot(week_roll.index,  week_roll,  label='1-week days rolling')
#     ax.plot(month_roll.index, month_roll, label='1-month days rolling')
#
#     ax.set_title(stock_name)
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Adjusted closing price ($)')
#     fig.autofmt_xdate()
#
#     ax.legend()
#     plt.show()