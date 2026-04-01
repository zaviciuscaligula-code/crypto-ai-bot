import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. ΛΗΨΗ ΔΕΔΟΜΕΝΩΝ
btc_raw = yf.download("BTC-USD", period="2y", interval="1d", auto_adjust=True)
ethbtc_raw = yf.download("ETH-BTC", period="2y", interval="1d", auto_adjust=True)
eth_raw = yf.download("ETH-USD", period="2y", interval="1d", auto_adjust=True)

df = pd.DataFrame(index=btc_raw.index)
df['BTC_Close'] = btc_raw['Close']
df['ETHBTC_Close'] = ethbtc_raw['Close']
df['ETH_Close'] = eth_raw['Close']
df = df.dropna()

# Cointegration
window_coint = 30
y = df['ETH_Close'].iloc[-window_coint:]
x = df['BTC_Close'].iloc[-window_coint:]
x = sm.add_constant(x)
model_coint = sm.OLS(y, x).fit()
spread = df['ETH_Close'].iloc[-1] - (model_coint.params.iloc[0] + model_coint.params.iloc[1] * df['BTC_Close'].iloc[-1])
historical_spreads = y - (model_coint.params.iloc[0] + model_coint.params.iloc[1] * x['BTC_Close'])
coint_z = (spread - historical_spreads.mean()) / historical_spreads.std()

# Data Prep for ML
window_z = 30
df['BTC_Z'] = (df['BTC_Close'] - df['BTC_Close'].rolling(window=window_z).mean()) / df['BTC_Close'].rolling(window=window_z).std()
df['ETHBTC_Z'] = (df['ETHBTC_Close'] - df['ETHBTC_Close'].rolling(window=window_z).mean()) / df['ETHBTC_Close'].rolling(window=window_z).std()

df['BTC_T1'] = (df['BTC_Z'].shift(-1) > df['BTC_Z']).astype(int)
df['BTC_T7'] = (df['BTC_Z'].shift(-7) > df['BTC_Z']).astype(int)
df['ETH_T1'] = (df['ETHBTC_Z'].shift(-1) > df['ETHBTC_Z']).astype(int)
df['ETH_T7'] = (df['ETHBTC_Z'].shift(-7) > df['ETHBTC_Z']).astype(int)
df = df.dropna()

seq_length = 10
X, Y_b1, Y_b7, Y_e1, Y_e7 = [], [], [], [], []
btc_z, eth_z = df['BTC_Z'].values, df['ETHBTC_Z'].values
y_btc1, y_btc7, y_eth1, y_eth7 = df['BTC_T1'].values, df['BTC_T7'].values, df['ETH_T1'].values, df['ETH_T7'].values

for i in range(len(df) - seq_length):
    X.append(np.column_stack((btc_z[i : i + seq_length], eth_z[i : i + seq_length])))
    Y_b1.append(y_btc1[i + seq_length])
    Y_b7.append(y_btc7[i + seq_length])
    Y_e1.append(y_eth1[i + seq_length])
    Y_e7.append(y_eth7[i + seq_length])

X, Y_b1, Y_b7, Y_e1, Y_e7 = map(np.array, [X, Y_b1, Y_b7, Y_e1, Y_e7])
X_flat = X.reshape(X.shape[0], -1)

# Training & Prediction on last day
inputs = Input(shape=(seq_length, 2))
lstm1 = LSTM(32)(inputs)
out_btc1 = Dense(1, activation='sigmoid')(lstm1)
model_lstm = Model(inputs=inputs, outputs=out_btc1)
model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
model_lstm.fit(X[:-1], Y_b1[:-1], epochs=10, batch_size=32, verbose=0)
lstm_prob = model_lstm.predict(X[-1:])[0][0]

xgb_m = xgb.XGBClassifier().fit(X_flat[:-1], Y_b1[:-1])
xgb_prob = xgb_m.predict_proba(X_flat[-1:])[0][1]

lgb_m = lgb.LGBMClassifier(verbose=-1).fit(X_flat[:-1], Y_b1[:-1])
lgb_prob = lgb_m.predict_proba(X_flat[-1:])[0][1]

# ARIMA
m_b = ARIMA(df['BTC_Z'], order=(2, 1, 0)).fit()
f_b1 = m_b.get_forecast(steps=1)
arima_prob = 1 - norm.cdf(df['BTC_Z'].iloc[-1], loc=f_b1.predicted_mean.iloc[0], scale=f_b1.se_mean.iloc[0])

# Σύνθεση Μηνύματος
def get_p(prob):
    return f"ΑΥΞΗΣΗ 📈 ({prob*100:.1f}%)" if prob >= 0.50 else f"ΠΤΩΣΗ 📉 ({(1-prob)*100:.1f}%)"

message = (
    "📊 *ΠΡΩΙΝΟ AI DASHBOARD REPORT*\n"
    f"💰 *BTC:* ${df['BTC_Close'].iloc[-1]:,.2f} | *Coint Z:* {coint_z:.2f}\n\n"
    "*ΠΡΟΒΛΕΨΕΙΣ ΓΙΑ ΑΥΡΙΟ*\n"
    f"📊 ARIMA: {get_p(arima_prob)}\n"
    f"🤖 LSTM: {get_p(lstm_prob)}\n"
    f"🌳 XGBoost: {get_p(xgb_prob)}\n"
    f"💡 LightGBM: {get_p(lgb_prob)}"
)

# Αποστολή στο Telegram
token = os.environ.get('TELEGRAM_BOT_TOKEN')
chat_id = os.environ.get('TELEGRAM_CHAT_ID')
url = f"https://api.telegram.org/bot{token}/sendMessage"
requests.post(url, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'})
print("✅ Το μήνυμα στάλθηκε!")
