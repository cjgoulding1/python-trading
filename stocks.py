import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
aapl = yf.download('AAPL',
                          start=datetime.datetime(2023, 10, 1),
                          end=datetime.datetime(2024, 1, 1))

aapl['Close'].plot(grid = True)
plt.show()