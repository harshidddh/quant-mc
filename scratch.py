from src.quantlab.data import load_log_returns

R = load_log_returns()
print(R.head())
print(len(R), R.index[0].date(), R.index[-1].date())
print("annualised vol:", R.std() * (252 ** 0.5))