import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from metrics.MAPE import MAPE_value
from metrics.RMSE import RMSE_value
from metrics.R2 import R2_value
from visualizations.comparision_plot import visualization_plot

current_dir = Path(__file__).resolve().parent
env_path = current_dir / '.env'

load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

SYMBOL = "NVDA"
START_DATE = "2020-01-01"
END_DATE = "2026-02-15"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

request_parameters = StockBarsRequest(
    symbol_or_symbols = SYMBOL,
    timeframe = TimeFrame.Day,
    start = START_DATE,
    end = END_DATE
)

bars = client.get_stock_bars(request_parameters)
df = bars.df.reset_index()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

df["return"] = df["close"].pct_change()
df["volatility"] = df["return"].rolling(10).std()
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_10"] = df["close"].rolling(10).mean()
df["ma_15"] = df["close"].rolling(15).mean()
df["ma_20"] = df["close"].rolling(20).mean()

df["target"] = df["close"].shift(-1)

features = [
    "close",
    "volume",
    "volatility",
    "ma_5",
    "ma_10",
    "ma_15",
    "ma_20"
]

df.dropna(inplace=True)

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mape = MAPE_value(y_test, y_pred)
rmse = RMSE_value(y_test, y_pred)
r2 = R2_value(y_test, y_pred)

print(f'MAPE: {round(mape,2)} %')
print(f'RMSE: {round(rmse,2)}')
print(f'R2: {round(r2,2)}')

visualization_plot(y_test, y_pred, SYMBOL)

future_prediction = model.predict(X.iloc[-1:].values)[0]
last_date = df.index[-1].date()

print(f"--------------------------------------------------")
print(f"Last date in dataset: {last_date}")
print(f"Predicted price for next day: ${future_prediction:.2f}")
print(f"--------------------------------------------------")

