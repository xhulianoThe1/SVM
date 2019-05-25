from datetime import datetime, date
import dateutil
from pandas import DataFrame
from dateutil.relativedelta import relativedelta
from datetime import timedelta
from pandas.io.json import json_normalize
from collections import Counter
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from bdateutil import isbday
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
import pandas_market_calendars as mcal
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import pandas_datareader
import datetime
import pandas_datareader.data as web
import sys
import ast
import json
#dictonary initialized as d
d = {}
#Opened Text file containing data and initalizing df as our datframe
f = open("Netflix.txt")
df = pd.DataFrame()
#Iterate through the file and change its type from string to dict
#Then add the contents of the dictionary to a data frame
for x in f:
    d = ast.literal_eval(x)
    df = df.append(d, ignore_index=True)
#Creating a new column as a placeholder for now
df['new'] = df['date'] + " " + df['time']
#converts dates to datetime
df['date'] = pd.to_datetime(df['date'])
df['new'] = pd.to_datetime(df['new'])
#If after 4 pm add 1 to the date because it is now the next trading day
for i in range (len(df['new'])):
     if df['new'][i].hour >= 16:
         df['new'][i] += timedelta(days=1)
         df['date'][i] += timedelta(days=1)
#Check to see if date is on a business day
#If it's not aka its on a saturday (no tweet news on sundays) add 2
for j in range (len(df['date'])):
     val = isbday(date(df['date'][j].year, df['date'][j].month, df['date'][j].day))
     if val == False:
         df['date'][j] += timedelta(days=2)
#Now dropping our place holder column
df.drop(['new'], axis = 1, inplace = True)
df.head()
#stock data imported from Yahoo
start = datetime.datetime(2014,1,1)
end = datetime.datetime(2019,4,1)
tsla = web.DataReader('NFLX', 'yahoo', start, end)
tsla['Returns'] = tsla['Close'].pct_change(1)
#Getting rid of irrelevant data
tsla.drop(['High', 'Low', 'Open', 'Adj Close', 'Volume'], axis = 1, inplace = True)
#New column called NYSE date which is = to the index because the data
#Had the index as a date
tsla['date'] = tsla.index
tsla.sort_values('date',  inplace=True, ascending=False)
tsla.head()
#Merging the data
# OUTER-join does NOT delete any values
# each calendar date now has a corresponding NYSE_Date
mrg = pd.merge(df, tsla, how='outer', on = 'date')
mrg.sort_values('date',  inplace=True, ascending=False)
# KEEP AN EYE ON THIS : if ascending = False use BFill, if ascending = True use FFill
mrg['date'] = mrg['date'].bfill()
#mrg = mrg.dropna(axis = 0, subset = ["text"])
from pywsd.utils import lemmatize_sentence
np.set_printoptions(threshold=sys.maxsize)
cl = []
for i in mrg['Returns']:
    if i > 0:
        cl.append(1)
    else:
        cl.append(0)
mrg['Class'] = cl
a = open("NetflixPos.txt", "w")
b = open("NetflixNeg.txt", "w")
j = open("Netflixwords.txt", "w")
for i in range(mrg.shape[0]):
    if mrg["Class"][i] == 1:
        s = str(mrg["text"][i]) + "\n"
        a.write(s.replace('"', ''))
        j.write(s.replace('"', ''))
    else:
        s = str(mrg["text"][i]) + "\n"
        b.write(s.replace('"', ''))
        j.write(s.replace('"', ''))
a.close()
b.close()
j.close()
#cl is our classifier
#Classifying our returns as 1 or 0
# This will let us lemmatize
lem = WordNetLemmatizer()
# Opening word document we just created and separting all the words
a = open("NetflixPos.txt", "r")
b = open("NetflixNeg.txt", "r")
c = a.read()
d = b.read()
splitp = c.split()
splitn = d.split()
# Lemmatize all the words
i = 0
pos = []
while i < len(splitp):
    pos.append(lem.lemmatize(splitp[i]))
    i += 1
# Counting the 100 most common words
cnt1 = Counter(pos)
most_occurp = cnt1.most_common(100)
a.close()
# Lemmatize all the words
i = 0
neg = []
while i < len(splitn):
    neg.append(lem.lemmatize(splitn[i]))
    i += 1
# Counting the 100 most common words
cnt2 = Counter(neg)
most_occurn = cnt2.most_common(100)
b.close()
# Stop words we'll use
stopwords = ["fastft", "ha", "@", "new", "nan", "The", "says", "a", "about", "above", "above", "across", "after",
                 "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although",
                 "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow",
                 "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became",
                 "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below",
                 "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can",
                 "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
                 "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty",
                 "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few",
                 "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty",
                 "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt",
                 "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
                 "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest",
                 "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd",
                 "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most",
                 "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless",
                 "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of",
                 "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
                 "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re",
                 "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show",
                 "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something",
                 "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the",
                 "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore",
                 "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three",
                 "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
                 "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well",
                 "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby",
                 "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole",
                 "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours",
                 "yourself", "yourselves", "the", "-", "u"]
ten = []
q = 0
t = 0
# Iterate 10 times and add in 10 most common words
mon = []
for i in most_occurn:
    while q < 10:
        if most_occurn[t][0] not in stopwords:
            mon.append(most_occurn[t][0])
            q+=1
        t+=1
q=0
t=0
for i in most_occurp:
    while q < 10:
        if most_occurp[t][0] not in stopwords:
            if most_occurp[t][0] not in mon:
                ten.append(most_occurp[t][0])
                q += 1
        t += 1
    # 10 Most common words
k = 0
for j in ten:
    k += 1
print(ten)
f = open("Netflixwords.txt", "r")
#get the number of tweets in file
count = -1
ch = " "
while ch != "":
    ch = f.readline()
    count +=1
f.close()
f = open("Netflixwords.txt", "r")
#turn the count of the words into an array
X = np.ndarray(shape=(count, 10), dtype=int)
y = np.ndarray(shape=(count,), dtype=int)
u = []
for i in range(count):
    line = f.readline()
    if "nan" not in line:
        u.append(1)
    else:
        u.append(0)
    y[i] = mrg["Class"][i]
    for x in range(0, 10):
        X[i][x] = line.count(ten[x])
q = []
for i in range(count):
    q.append(X[i])
mrg["word"] = q
mrg["has"] = u
date = []
Crass = []
word = []
countt = []
#filling arrays to create a new dataframe
for index, row in mrg.iterrows():
    if len(date) == 0:
        date.append(row["date"])
        Crass.append(row["Class"])
        countt.append(row["has"])
        word.append([a + b for a, b in zip([0,0,0,0,0,0,0,0,0,0], row["word"])])
    elif row["date"] != date[len(date)-1]:
        date.append(row["date"])
        Crass.append(row["Class"])
        countt.append(row["has"])
        word.append([a + b for a, b in zip([0,0,0,0,0,0,0,0,0,0], row["word"])])
    else:
        word[len(word)-1] = [a + b for a, b in zip( word[len(word)-1], row["word"])]
        countt[len(countt) - 1] += 1
date = date[::-1]
Crass = Crass[::-1]
word = word[::-1]
countt = countt[::-1]
nf = pd.DataFrame({"date":date,"word":word,"count":countt})
nf.sort_values('date',  inplace=True, ascending=False)
#create a copy of mrg
mrg2 = pd.DataFrame()
mrg2 = mrg
mrg2 = mrg2.drop(columns = "word")
mrg2.drop_duplicates(subset = "date", keep = "first", inplace= True)
#merge mrg2 and nf
fin = pd.merge(nf, mrg2, how='outer', on = 'date')
fin = fin.drop(columns = ["Returns","Close","time","text"])
#perform logistic regression
datef = []
predict = []
train_in = []
train_class = []
cnt = []
#create initial  training set
for i in reversed(range(1076,1331)):
    train_in.append(fin["word"][i])
    train_class.append(fin["Class"][i])
#perform predictions, then add the test to training
for i in reversed(range(1076)):
    clf = LogisticRegression(solver = "liblinear").fit(train_in, train_class)
    predict.append(clf.predict_proba(np.array(fin["word"][i]).reshape(1,-1))[0][0])
    train_in.append(fin["word"][i])
    train_class.append(fin["Class"][i])
    datef.append(fin["date"][i])
    cnt.append(fin["count"][i])
#Returns the probability of the sample for each class in the model, where classes are ordered as they are in self.classes_.
pf = pd.DataFrame({"date":datef,"prediction":predict,"count":cnt})
pf.to_csv("NetflixOut.csv")