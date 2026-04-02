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
df = df.dropna()

seq_length = 10
X, Y_b1, Y_b7 = [], [], []
btc_z, eth_z = df['BTC_Z'].values, df['ETHBTC_Z'].values
y_btc1, y_btc7 = df['BTC_T1'].values, df['BTC_T7'].values

for i in range(len(df) - seq_length):
    X.append(np.column_stack((btc_z[i : i + seq_length], eth_z[i : i + seq_length])))
    Y_b1.append(y_btc1[i + seq_length])
    Y_b7.append(y_btc7[i + seq_length])

X, Y_b1, Y_b7 = map(np.array, [X, Y_b1, Y_b7])
X_flat = X.reshape(X.shape[0], -1)

# Training LSTM
inputs = Input(shape=(seq_length, 2))
lstm1 = LSTM(32)(inputs)
out_btc1 = Dense(1, activation='sigmoid')(lstm1)
out_btc7 = Dense(1, activation='sigmoid')(lstm1)
model_lstm = Model(inputs=inputs, outputs=[out_btc1, out_btc7])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
model_lstm.fit(X[:-1], [Y_b1[:-1], Y_b7[:-1]], epochs=10, batch_size=32, verbose=0)
lstm_p1, lstm_p7 = model_lstm.predict(X[-1:])

# Training XGBoost
xgb_1 = xgb.XGBClassifier().fit(X_flat[:-1], Y_b1[:-1])
xgb_7 = xgb.XGBClassifier().fit(X_flat[:-1], Y_b7[:-1])
xgb_p1 = xgb_1.predict_proba(X_flat[-1:])[0][1]
xgb_p7 = xgb_7.predict_proba(X_flat[-1:])[0][1]

# Training LightGBM
lgb_1 = lgb.LGBMClassifier(verbose=-1).fit(X_flat[:-1], Y_b1[:-1])
lgb_7 = lgb.LGBMClassifier(verbose=-1).fit(X_flat[:-1], Y_b7[:-1])
lgb_p1 = lgb_1.predict_proba(X_flat[-1:])[0][1]
lgb_p7 = lgb_7.predict_proba(X_flat[-1:])[0][1]

# ARIMA
m_b = ARIMA(df['BTC_Z'], order=(2, 1, 0)).fit()
f_b1 = m_b.get_forecast(steps=1)
f_b7 = m_b.get_forecast(steps=7)

arima_p1 = 1 - norm.cdf(df['BTC_Z'].iloc[-1], loc=f_b1.predicted_mean.iloc[0], scale=f_b1.se_mean.iloc[0])
arima_p7 = 1 - norm.cdf(df['BTC_Z'].iloc[-1], loc=f_b7.predicted_mean.iloc[-1], scale=f_b7.se_mean.iloc[-1])

# Σύνθεση Μηνύματος
def get_p(prob):
    return f"ΑΥΞΗΣΗ 📈 ({prob*100:.1f}%)" if prob >= 0.50 else f"ΠΤΩΣΗ 📉 ({(1-prob)*100:.1f}%)"

message = (
    "📊 *ΠΡΩΙΝΟ AI DASHBOARD REPORT*\n"
    f"💰 *BTC:* ${df['BTC_Close'].iloc[-1]:,.2f} | *Coint Z:* {coint_z:.2f}\n"
    "━━━━━━━━━━━━━━━━━━━\n\n"
    "🔮 *ΠΡΟΒΛΕΨΕΙΣ ΓΙΑ ΑΥΡΙΟ (1D)*\n"
    f"📊 ARIMA: {get_p(arima_p1)}\n"
    f"🤖 LSTM: {get_p(lstm_p1[0][0])}\n"
    f"🌳 XGBoost: {get_p(xgb_p1)}\n"
    f"💡 LightGBM: {get_p(lgb_p1)}\n\n"
    "🔮 *ΠΡΟΒΛΕΨΕΙΣ ΓΙΑ 7 ΜΕΡΕΣ (7D)*\n"
    f"📊 ARIMA: {get_p(arima_p7)}\n"
    f"🤖 LSTM: {get_p(lstm_p7[0][0])}\n"
    f"🌳 XGBoost: {get_p(xgb_p7)}\n"
    f"💡 LightGBM: {get_p(lgb_p7)}"
)

# Αποστολή στο Telegram
token = os.environ.get('TELEGRAM_BOT_TOKEN')
chat_id = os.environ.get('TELEGRAM_CHAT_ID')
url = f"https://api.telegram.org/bot{token}/sendMessage"
requests.post(url, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'})
print("✅ Το μήνυμα στάλθηκε!")
