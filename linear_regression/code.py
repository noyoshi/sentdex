import pandas as pd 
import quandl  
import numpy as np 
import math 
from sklearn import preprocessing, cross_validation, svm 
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
# Now our data frame (df) only has these columns 
# Open price vs close price tells us how much they changed, but a linear
# regression will not look at these special relationships unless we define them
# ourselves 

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
# Defined the relationships we care about 

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# Volume is related to volatility, worth keeping 

forecast_col = 'Adj. Close' 
# You can change this to be whatever you want to predict?

df.fillna(-99999, inplace=True) 
# in ML you cannot get rid of data... also cannot use NaN... so use custom
# data like this for filler 

forecast_out = int(math.ceil(0.01*len(df)))
# we are trying to get 10% out? 0.1-> 10%
# using 10 days worth of data to predict tomorrow's price I think * 
# I did /2 since sentdex's df seems to be about half the size of mine?
# just changed the label size by 0.5. Reverted this.

df['label'] = df[forecast_col].shift(-forecast_out)
# the label column for each row will be the adj. close price 10 days in the
# future 
# features are attributes of what, in our mind, may cause the adj. close price
# in 10 days to change (10%? not 10 days, since we did not specify a time
# frame)
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1)) 
# df.drop RETURNS a new array
y = np.array(df['label'])

X = preprocessing.scale(X)
# In order to properly scale values, you need to include ALL values (training
# AND testing) since they are scaled RELATIVE to eachother 
# 
# This adds processing time, so if you were doing high frequency trading, you
# would want to skip this step!

df.dropna(inplace=True)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y,
        test_size = 0.2)
# 20% of data will be testing data. This shuffles things up and outpts out
# training and testing data. Neato. 

# LINEAR REGRESSION  
clf = LinearRegression(n_jobs=10)
# n_jobs specifies number of parallel jobs to run. Speeds up training!
# Parallel jobs (multi threads) only work well in some cases, like linear
# regression, do not work in others, like SVMs
#
# BUT WHAT IF YOU WANT TO USE ANOTHER ALGORITHM (EG. SUPPORT VECTOR MACHINE)
# aka SVM 
# 
# Then: 
# clf = svm.SVR()

clf.fit(X_train, y_train)
# fit == train 

accuracy = clf.score(X_test, y_test)
# score == test 

print(accuracy)

