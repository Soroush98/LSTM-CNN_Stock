import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from keras.models import Sequential
from keras.layers import Conv1D,MaxPool1D,Bidirectional,LSTM,Dropout,TimeDistributed
from keras.layers import Dense,GlobalAveragePooling2D
from ta.trend import IchimokuIndicator
from sklearn.linear_model import LinearRegression
from ta import add_all_ta_features
from ta.utils import dropna
import matplotlib.pyplot as plt
filename = 'AAPL'
stock = pd.read_csv('Data/' + filename + '.csv')
indicator_bb = BollingerBands(close=stock["Close"], n=20, ndev=2)
macd = MACD(close=stock["Close"])
rsi = RSIIndicator(close=stock["Close"])
ichi = IchimokuIndicator(high=stock["High"],low=stock["Low"])
stock['macd'] = macd.macd()
stock['rsi'] = rsi.rsi()
stock['bb_bbm'] = indicator_bb.bollinger_mavg()
stock['bb_bbh'] = indicator_bb.bollinger_hband()
stock['bb_bbl'] = indicator_bb.bollinger_lband()
stock['ichi_a'] = ichi.ichimoku_a()
stock['ichi_b'] = ichi.ichimoku_b()
stock['ichi_base'] = ichi.ichimoku_base_line()
stock['ichi_conv'] = ichi.ichimoku_conversion_line()
stock = stock.fillna(0)
print(stock)
scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,1:4])
stock.iloc[:,1:4] = scaled_values

y_scaler = preprocessing.MinMaxScaler()
scaled_values = y_scaler.fit_transform(np.array(stock.iloc[:,4]).reshape(-1,1))
stock.iloc[:,4] = scaled_values
scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,5:])
stock.iloc[:,5:] = scaled_values
window_size = 50
X = []
Y = []


Lstock = stock.drop(['Date','Close'],1)
model = LinearRegression()
model.fit(Lstock.iloc[:,:], stock.iloc[:,4])
importance = model.coef_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([Lstock.columns[x] for x in range(len(importance))], importance)
plt.show()
stock_final = Lstock.drop(['Open','Volume','Adj Close','rsi','bb_bbm','bb_bbh','bb_bbl','ichi_a','ichi_b','ichi_conv'],1)
print(stock_final)
for i in range(0 , len(stock_final) - window_size -1 , 1):
    X.append(np.array(stock_final.iloc[i:i+window_size,:]).reshape(window_size*4,1))
    Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.1,shuffle=False)
train_X = np.array(train_X)
test_X = np.array(test_X)
train_label = np.array(train_label)
test_label = np.array(test_label)
model = Sequential()
#add model layers

model.add((LSTM(128,return_sequences=True)))
model.add((LSTM(64,return_sequences=False)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='RMSprop', loss='mse')
model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=50,shuffle=False)
print(model.evaluate(test_X,test_label))
# model.summary()
predicted  = model.predict(test_X)
test_label[:,0] = y_scaler.inverse_transform(test_label[:,0])
predicted = np.array(predicted[:,0]).reshape(-1,1)
predicted = y_scaler.inverse_transform(predicted)
plt.plot(test_label[:,0], color = 'black', label = ' Stock Price')
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()