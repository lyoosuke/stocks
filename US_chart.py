import pandas_datareader as pdr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from mpl_finance import candlestick_ohlc
df = pdr.get_data_yahoo('AMZN', '2019-06-01', '2019-12-30')
ax = plt.subplot()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
candlestick_ohlc(ax, zip(mdates.date2num(df.index), df['Open'], df['High'], df['Low'], df['Close']), width=0.4)
plt.title('AMZN')
plt.savefig('stock.png')
