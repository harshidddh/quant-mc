import numpy as np
import yfinance as yf

def load_log_returns(ticker="^NSEI", start="2010-01-01", end=None):
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"].squeeze()
    return np.log(px).diff().dropna()