import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
NKE = yf.download('NKE',
                          start=datetime.datetime(2012, 10, 1),
                          end=datetime.datetime(2024, 9, 22))

pd.set_option('display.max_rows', None)

def gen_signal(df):
    df['30 Day Avg'] = df['Adj Close'].rolling(window=30).mean()
    df['100 Day Avg'] = df['Adj Close'].rolling(window=100).mean()
    df = df.iloc[100:df.shape[0]].reset_index()
    df['Higher or Lower'] = [1 if df['30 Day Avg'].iloc[i] > df['100 Day Avg'].iloc[i]
                    else (-1 if df['30 Day Avg'].iloc[i] < df['100 Day Avg'].iloc[i] else 0) for i in range(df.shape[0])]
    df['Signal'] = [1 if df['Higher or Lower'].iloc[i] > df['Higher or Lower'].iloc[i-1]
                    else (-1 if df['Higher or Lower'].iloc[i] < df['Higher or Lower'].iloc[i-1] else 0) for i in range(df.shape[0])]
    df['Signal'].iloc[0] = 0
    return(df)

def backtest(df, capital):
    stocks = 0
    position = 0
    for index, row in df.iterrows():
        if row['Signal'] == 1 and position == 0:
            stocks = capital/row['Adj Close']
            capital = 0
            position = 1
        if row['Signal'] == -1 and position == 1:
            capital = stocks * row['Adj Close']
            stocks = 0
            position = 0
    return(stocks, capital, stocks * df['Adj Close'].iloc[df.shape[0]-1])

df = gen_signal(NKE)

print(backtest(df, 100))

plt.plot(df['Date'], df['Adj Close'])
plt.plot(df['Date'], df['30 Day Avg'])
plt.plot(df['Date'], df['100 Day Avg'])
plt.show()


