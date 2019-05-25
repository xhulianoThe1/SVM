from pandas_datareader import data
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from pandas.plotting import register_matplotlib_converters
from ast import literal_eval
register_matplotlib_converters()

# open up the logistic score, format it for later use
tsla_tweets = pd.read_csv("TeslaOut.csv")
tsla_tweets.prediction = tsla_tweets.prediction.astype(float)
tsla_tweets = tsla_tweets[['date','prediction','count']]
tsla_tweets.columns = ['NYSE_Date','Tweet_Score','Count']
tsla_tweets['NYSE_Date'] = pd.to_datetime(tsla_tweets['NYSE_Date'], format='%Y-%m-%d')

# global parameters
start_date = '2014-03-27'
end_date   = '2019-03-27'

# retrieve financial data from API
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

# pull historical data from API
stock = getData('TSLA', 'stock')
sector = getData('XLY', 'sector')
market = getData('SPY', 'market')

# combine three financial datasets
full_data = pd.concat([stock, sector, market], axis=1)
full_data["NYSE_Date"] = full_data.index

# append tweet_score to full dataset
full_data = full_data.merge(tsla_tweets, how = "inner", on = "NYSE_Date")
full_data.set_index('NYSE_Date', inplace=True)

TRAIN_START = '2015-01-01'
TRAIN_END   = '2016-12-31'
TEST_START  = '2017-01-01'
TEST_END    = '2018-12-31'

def runSVM(data, type) :
    training_data = full_data.loc[TRAIN_START:TRAIN_END]
    testing_Data = full_data.loc[TEST_START:TEST_END]

    y_train = training_data['stock_Event']
    y_true = testing_Data['stock_Event']

    if (type == "no_tweets") :
        X_train = training_data[['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V']]
        X_test = testing_Data[['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V']]

    elif (type == "with_tweets") :
        X_train = training_data[['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V', 'Tweet_Score', 'Count']]
        X_test = testing_Data[['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V', 'Tweet_Score', 'Count']]

    # implement support vector machine with a linear kernel
    svc = svm.SVC(kernel='linear')
    Cs = range(1, 20)
    clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10)
    clf.fit(X_train, y_train)

    # evaluate model performance (calculate success rate as percentage)
    y_predic = clf.predict(X_test)
    y_test = pd.DataFrame(y_predic)
    y_true = pd.DataFrame(y_true)
    y_true = y_true.reset_index(drop=True)

    verify_data = pd.concat([y_true, y_test], axis=1)
    verify_data.columns = ['True_Event', 'Prediction']
    verify_data.loc[verify_data['True_Event'] != verify_data['Prediction'], 'Correct'] = 0
    verify_data.loc[verify_data['True_Event'] == verify_data['Prediction'], 'Correct'] = 1

    success_rate = verify_data['Correct'].mean() * 100
    print(type, 'correct prediction', round(success_rate, 2), "%")

runSVM(full_data, "no_tweets")
runSVM(full_data, "with_tweets")