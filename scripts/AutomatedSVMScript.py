# Maksim Papenkov
# SVM for prediction score evaluation

from pandas_datareader import data
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn import svm
    from sklearn.grid_search import GridSearchCV

# global parameters
# note : set start date 1 month before TRUE start (to calculate momentum, volatility)
# this analysis is from jan 2015 through Dec 2018 (inclusive)
start_date = '2014-12-01'
end_date   = '2018-12-31'

# format prediction score data (from other script)
def formatPredictionScore (tweet_data) :
    # open up the logistic score, format it for later use
    predic_data = pd.read_csv(tweet_data)

    # average number of tweets per NYSE date
    mean_n_tweets = predic_data['count'].mean()

    # format dataframe for merging with financial data
    predic_data = predic_data[['date', 'prediction','count']]
    predic_data.columns = ['NYSE_Date', 'Tweet_Score', 'n_Tweets']
    predic_data['NYSE_Date'] = pd.to_datetime(predic_data['NYSE_Date'], format='%Y-%m-%d')

    return predic_data, mean_n_tweets

# pull financial data from API, format for future use
def getData(ticker, type) :

    name_event      = type + "_Event"
    name_momentum   = type + "_M" # Momentum
    name_volatility = type + "_V" # Volatility

    # retrieve data using IEX API
    price_data = data.DataReader(ticker, 'iex', start_date, end_date)

    # convert closing price to returns
    price_data['return'] = price_data['close'].pct_change()

    # if returns are positive, event = 1 (otherwise event = 0)
    price_data.loc[price_data['return'] < 0, name_event] = 0
    price_data.loc[price_data['return'] >= 0, name_event] = 1

    # calculate rolling momentum / volatility (one month ~ 20 trading days)
    price_data[name_momentum]   = price_data[name_event].rolling(20).mean()
    price_data[name_volatility] = price_data['return'].rolling(20).std()

    # convert date from index to DATE format (for easy plotting)
    price_data.index = pd.to_datetime(price_data.index, format='%Y-%m-%d')

    if (type == 'stock') :
        returns_data = price_data[[name_event, name_momentum, name_volatility]]
    else :
        returns_data = price_data[[name_momentum, name_volatility]]

    return returns_data

# run an svm (with or without tweet data)
def runSVM(full_data, tweets, svm_kernel) :

    # initial training dataset has all NYSE data for 2015 (ONE YEAR ~ 255 observations)
    # start = first NYSE date in calendar year; end = last NYSE date in calendar year
    train_start_date = min(full_data.index[full_data.index.year == 2015])
    train_end_date   = max(full_data.index[full_data.index.year == 2016])
    training_data    = full_data.loc[train_start_date:train_end_date]

    # start_i = first NYSE date after training end date; end_i = last NYSE date in full dataset
    start_i = full_data.index.get_loc(train_end_date) + 1
    end_i   = full_data.index.get_loc(max(full_data.index[full_data.index.year == 2018]))

    # create empty dataframe to fill with simulated data
    output_frame = pd.DataFrame()

    # iteratively run SVM, add last day's test data to next day's training data
    # (replicates real-world application, constantly update based on new information)
    for i in range(start_i, end_i) :

        # testing data is immediate next NYSE date after end of training
        testing_data = full_data.iloc[i]

        # note : y_train is vector, y_true is scalar
        y_train = training_data['stock_Event']
        y_true  = testing_data ['stock_Event']

        # if twitter score is not used, run a 6 parameter SVM
        if (tweets == False) :
           X_train = training_data[['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V']]
           X_test  = testing_data [['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V']]

        # if twitter score is used, run a 7 parameter SVM
        elif (tweets == True) :
           X_train = training_data[['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V', 'Tweet_Score']]
           X_test  = testing_data [['stock_M', 'stock_V', 'sector_M', 'sector_V', 'market_M', 'market_V', 'Tweet_Score']]

        # implement support vector machine, fit with training data
        svc = svm.SVC(kernel = svm_kernel)
        Cs = range(1, 20)
        clf = GridSearchCV(estimator = svc, param_grid = dict(C = Cs), cv = 10)
        clf.fit(X_train, y_train)

        # use fitted model to predict class for test data (one day's worth of data)
        y_predic = clf.predict(X_test)
        output = [{'NYSE_Date' : full_data.index[i], 'y_Predic' : y_predic[0], 'y_True' : y_true}]
        output = pd.DataFrame(output)
        output_frame = output_frame.append(output)

        # update training dataset
        training_data = training_data.append(testing_data)

        if   (tweets == False) :
            svm_type    = "model 1 : no tweets"
            iteratition = i - start_i + 1
        elif (tweets == True)  :
            svm_type = "model 2 : with tweets"
            iteratition = i - start_i*2 + 1 + end_i

        print svm_type, '- running svm #', iteratition , 'out of', (end_i - start_i) * 2

    # format data for output
    output_frame.loc[output_frame['y_True'] != output_frame['y_Predic'], 'Correct'] = 0
    output_frame.loc[output_frame['y_True'] == output_frame['y_Predic'], 'Correct'] = 1

    return output_frame

# run the entire process in a single line (fully automated process)
def compareSVM(ticker, sector_etf, tweet_data, svm_kernel) :

    # pre-process twitter prediction score
    predic_score, mean_n_tweets = formatPredictionScore(tweet_data)

    # pull historical data from API
    stock  = getData(ticker, 'stock')
    sector = getData(sector_etf, 'sector')
    market = getData('SPY', 'market')

    # combine three financial datasets
    full_data = pd.concat([stock, sector, market], axis=1)
    full_data["NYSE_Date"] = full_data.index

    # append tweet_score to full dataset (as seventh parameter)
    full_data = full_data.merge(predic_score, how="inner", on="NYSE_Date")
    full_data.set_index('NYSE_Date', inplace=True)

    # unitize data for better model performance (project to unit hypercube)
    # i.e. each vector is mapped to [0,1] so that features have equal weighting
    full_data = full_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    # get predicted outcomes from running svm simulation
    svm_no_tweet = runSVM(full_data, tweets = False, svm_kernel = svm_kernel)
    svm_tweet    = runSVM(full_data, tweets = True , svm_kernel = svm_kernel)

    # get success rates for two types of svm simulations
    success_rate_no_tweet = svm_no_tweet['Correct'].mean() * 100
    success_rate_tweet    = svm_tweet   ['Correct'].mean() * 100

    # write important info to text file
    l1 = '\n' + ticker + ", kernel = " + svm_kernel + '\n'
    l2 = 'average number of tweets per NYSE date ' + str(round(mean_n_tweets, 2)) + '\n'
    l3 = 'accuracy without tweets ' + str(round(success_rate_no_tweet, 2)) + '%' + '\n'
    l4 = 'updated accuracy ' + str(round(success_rate_tweet, 2)) + '%' + '\n'
    l5 = 'improvement = ' + str(round(success_rate_tweet - success_rate_no_tweet, 2)) + '%' + '\n'
    file = open('svm_comparison.txt', 'a').writelines([l1, l2, l3, l4, l5])

    # print important info for instant gratification
    print l1 + '\n' + l2 + '\n' + l3 + '\n' + l4 + '\n' + l5

# compareSVM(ticker="TSLA", sector_etf="XLY", tweet_data="TeslaOut.csv", svm_kernel="linear")

# compareSVM(ticker="FB", sector_etf="XLK", tweet_data="FacebookOut.csv", svm_kernel="linear")
# FB : 2.4 tweets per day, 55.49% -> 56.89% (1.4% improvement)

compareSVM(ticker="NFLX", sector_etf="XLY", tweet_data="NetflixOut.csv", svm_kernel="rbf")


compareSVM(ticker="NFLX", sector_etf="XLY", tweet_data="NetflixOut.csv", svm_kernel="linear")
# NFLX : 0.56 tweets per day, 57.49 -> 57.68 (0.2 gain)
