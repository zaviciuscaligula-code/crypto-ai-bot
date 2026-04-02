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

# Training LSTM
inputs = Input(shape=(seq_length, 2))
lstm1 = LSTM(32)(inputs)
out_b1 = Dense(1, activation='sigmoid')(lstm1)
out_b7 = Dense(1, activation='sigmoid')(lstm1)
out_e1 = Dense(1, activation='sigmoid')(lstm1)
out_e7 = Dense(1, activation='sigmoid')(lstm1)
model_lstm = Model(inputs=inputs, outputs=[out_b1, out_b7, out_e1, out_e7])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
model_lstm.fit(X[:-1], [Y_b1[:-1], Y_b7[:-1], Y_e1[:-1], Y_e7[:-1]], epochs=10, batch_size=32, verbose=0)
l_pb1, l_pb7, l_pe1, l_pe7 = model_lstm.predict(X[-1:])

# Training XGBoost
xgb_b1 = xgb.XGBClassifier().fit(X_flat[:-1], Y_b1[:-1])
xgb_b7 = xgb.XGBClassifier().fit(X_flat[:-1], Y_b7[:-1])
xgb_e1 = xgb.XGBClassifier().fit(X_flat[:-1], Y_e1[:-1])
xgb_e7 = xgb.XGBClassifier().fit(X_flat[:-1], Y_e7[:-1])

# Training LightGBM
lgb_b1 = lgb.LGBMClassifier(verbose=-1).fit(X_flat[:-1], Y_b1[:-1])
lgb_b7 = lgb.LGBMClassifier(verbose=-1).fit(X_flat[:-1], Y_b7[:-1])
lgb_e1 = lgb.LGBMClassifier(verbose=-1).fit(X_flat[:-1], Y_e1[:-1])
lgb_e7 = lgb.LGBMClassifier(verbose=-1).fit(X_flat[:-1], Y_e7[:-1])

# ARIMA
m_b = ARIMA(df['BTC_Z'], order=(2, 1, 0)).fit()
m_e = ARIMA(df['ETHBTC_Z'], order=(2, 1, 0)).fit()
fb1, fb7 = m_b.get_forecast(steps=1), m_b.get_forecast(steps=7)
fe1, fe7 = m_e.get_forecast(steps=1), m_e.get_forecast(steps=7)

a_pb1 = 1 - norm.cdf(df['BTC_Z'].iloc[-1], loc=fb1.predicted_mean.iloc[0], scale=fb1.se_mean.iloc[0])
a_pb7 = 1 - norm.cdf(df['BTC_Z'].iloc[-1], loc=fb7.predicted_mean.iloc[-1], scale=fb7.se_mean.iloc[-1])
a_pe1 = 1 - norm.cdf(df['ETHBTC_Z'].iloc[-1], loc=fe1.predicted_mean.iloc[0], scale=fe1.se_mean.iloc[0])
a_pe7 = 1 - norm.cdf(df['ETHBTC_Z'].iloc[-1], loc=fe7.predicted_mean.iloc[-1], scale=fe7.se_mean.iloc[-1])

# Σύνθεση Μηνύματος
def get_p(prob):
    return f"ΑΥΞΗΣΗ 📈 ({prob*100:.1f}%)" if prob >= 0.50 else f"ΠΤΩΣΗ 📉 ({(1-prob)*100:.1f}%)"

message = (
    "📊 *ΠΡΩΙΝΟ AI DASHBOARD REPORT*\n"
    f"💰 *BTC:* ${df['BTC_Close'].iloc[-1]:,.2f}\n"
    f"Ξ *ETH-BTC:* {df['ETHBTC_Close'].iloc[-1]:.5f}\n"
    f"🔗 *Coint Z:* {coint_z:.2f}\n"
    "━━━━━━━━━━━━━━━━━━━\n\n"
    "🔮 *ΠΡΟΒΛΕΨΕΙΣ BITCOIN (BTC)*\n"
    "-------------------\n"
    f"📊 ARIMA: 1D {get_p(a_pb1)} | 7D {get_p(a_pb7)}\n"
    f"🤖 LSTM: 1D {get_p(l_pb1[0][0])} | 7D {get_p(l_pb7[0][0])}\n"
    f"🌳 XGB:  1D {get_p(xgb_b1.predict_proba(X_flat[-1:])[0][1])} | 7D {get_p(xgb_b7.predict_proba(X_flat[-1:])[0][1])}\n"
    f"💡 LGBM: 1D {get_p(lgb_b1.predict_proba(X_flat[-1:])[0][1])} | 7D {get_p(lgb_b7.predict_proba(X_flat[-1:])[0][1])}\n\n"
    "🔮 *ΠΡΟΒΛΕΨΕΙΣ ETH - BTC*\n"
    "-------------------\n"
    f"📊 ARIMA: 1D {get_p(a_pe1)} | 7D {get_p(a_pe7)}\n"
    f"🤖 LSTM: 1D {get_p(l_pe1[0][0])} | 7D {get_p(l_pe7[0][0])}\n"
    f"🌳 XGB:  1D {get_p(xgb_e1.predict_proba(X_flat[-1:])[0][1])} | 7D {get_p(xgb_e7.predict_proba(X_flat[-1:])[0][1])}\n"
    f"💡 LGBM: 1D {get_p(lgb_e1.predict_proba(X_flat[-1:])[0][1])} | 7D {get_p(lgb_e7.predict_proba(X_flat[-1:])[0][1])}"
)

# Αποστολή στο Telegram
token = os.environ.get('TELEGRAM_BOT_TOKEN')
chat_id = os.environ.get('TELEGRAM_CHAT_ID')
url = f"https://api.telegram.org/bot{token}/sendMessage"
requests.post(url, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'})
print("✅ Το μήνυμα στάλθηκε!")
