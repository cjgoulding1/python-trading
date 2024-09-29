import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import random
import talib
from sklearn.preprocessing import StandardScaler
import tensorflow
import keras
from keras import layers

random.seed(42)

pd.set_option('display.max_rows', None)

price_AAPL = yf.download('AAPL',
                          start=datetime.datetime(2012, 10, 1),
                          end=datetime.datetime(2024, 9, 22))

price_AAPL['H-L'] = price_AAPL['High'] - price_AAPL['Low']
price_AAPL['O-C'] = price_AAPL['Close'] - price_AAPL['Open']
price_AAPL['3day MA'] = price_AAPL['Close'].shift(1).rolling(window = 3).mean()
price_AAPL['10day MA'] = price_AAPL['Close'].shift(1).rolling(window = 10).mean()
price_AAPL['30day MA'] = price_AAPL['Close'].shift(1).rolling(window = 30).mean()
price_AAPL['Std_dev']= price_AAPL['Close'].rolling(5).std()
price_AAPL['RSI'] = talib.RSI(price_AAPL['Close'].values, timeperiod = 9)
price_AAPL['Williams %R'] = talib.WILLR(price_AAPL['High'].values, price_AAPL['Low'].values, price_AAPL['Close'].values, 7)
price_AAPL['Price_Rise'] = np.where(price_AAPL['Close'].shift(-1) > price_AAPL['Close'], 1, 0)
price_AAPL = price_AAPL.dropna()

X = price_AAPL.iloc[:, 4:-1]
y = price_AAPL.iloc[:, -1]

split = int(len(price_AAPL)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

input = keras.Input(shape = (X.shape[1],))
layer1 = layers.Dense(128, activation = "relu")(input)
layer2 = layers.Dense(128,activation = "relu")(layer1)
output = layers.Dense(1,activation = "sigmoid")(layer2)

network = keras.Model(input,output)

network.compile(optimizer = 'adam', loss = 'mse')

network.fit(X_train, y_train,
                epochs=100,
                batch_size=10,
                shuffle=True,
                validation_data=(X_test, y_test))


price_NKE = yf.download('NKE',
                          start=datetime.datetime(2012, 10, 1),
                          end=datetime.datetime(2024, 9, 22))

price_NKE['H-L'] = price_NKE['High'] - price_NKE['Low']
price_NKE['O-C'] = price_NKE['Close'] - price_NKE['Open']
price_NKE['3day MA'] = price_NKE['Close'].shift(1).rolling(window = 3).mean()
price_NKE['10day MA'] = price_NKE['Close'].shift(1).rolling(window = 10).mean()
price_NKE['30day MA'] = price_NKE['Close'].shift(1).rolling(window = 30).mean()
price_NKE['Std_dev']= price_NKE['Close'].rolling(5).std()
price_NKE['RSI'] = talib.RSI(price_NKE['Close'].values, timeperiod = 9)
price_NKE['Williams %R'] = talib.WILLR(price_NKE['High'].values, price_NKE['Low'].values, price_NKE['Close'].values, 7)
price_NKE['Price_Rise'] = np.where(price_NKE['Close'].shift(-1) > price_NKE['Close'], 1, 0)
price_NKE = price_NKE.dropna()

test = sc.transform(price_NKE.iloc[:, 4:-1])

prediction = network.predict(test)

binary = [1 if prediction[i] > 0.5 else 0 for i in range(len(prediction))]

price_NKE['Prediction'] = binary

print(price_NKE)

