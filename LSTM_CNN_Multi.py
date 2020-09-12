import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn import preprocessing
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from keras.models import Sequential
from keras.layers import Conv1D,MaxPool1D,Bidirectional,LSTM,Dropout,TimeDistributed
from keras.layers import Dense,GlobalAveragePooling2D
from ta.trend import IchimokuIndicator
from sklearn.linear_model import LinearRegression
from keras.layers import Conv1D,Flatten,MaxPooling1D,Bidirectional,LSTM,Dropout,TimeDistributed,MaxPool2D
from keras.layers import Dense,GlobalAveragePooling2D
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
#
scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,1:4])
stock.iloc[:,1:4] = scaled_values

y_scaler = preprocessing.MinMaxScaler()
scaled_values = y_scaler.fit_transform(np.array(stock.iloc[:,4]).reshape(-1,1))
stock.iloc[:,4] = scaled_values

scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,5:])
stock.iloc[:,5:] = scaled_values

Lstock = stock.drop(['Close','Date'],1)
model = LinearRegression()
model.fit(Lstock.iloc[:,:], stock.iloc[:,4])
importance = model.coef_
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([Lstock.columns[x] for x in range(len(importance))], importance)
plt.show()
stock_final = stock.drop(['Date','Open','Volume','macd','bb_bbm','bb_bbh','bb_bbl','ichi_a','ichi_conv'],1)

window_size = 50
week = 7
X = []
Y = []
print(stock_final)
for i in range(0 , len(stock) - window_size -1 , 1):
    # first = stock.iloc[i, 4]
    # temp = []
    # temp2 = []
    # for j in range(window_size):
    #     temp.append((stock_final.iloc[i + j, 4] - first) / first)
   # for j in range(week):
   #  temp2.append((stock.iloc[i +window_size, 4] - first) / first)
    X.append(np.array(stock_final.iloc[i:i+window_size,:]).reshape(window_size * 7,1))
    Y.append(np.array(stock.iloc[i+window_size,4]).reshape(1,1))
    # print(stock2.iloc[i:i+window_size,4])
    # X.append(np.array(temp).reshape(50, 1))
    # Y.append(np.array(temp2).reshape(1,1))
train_X,test_X,train_label,test_label = train_test_split(X, Y, test_size=0.1,shuffle=False)
len_t = len(train_X)
# train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_label, test_size=0.2,shuffle=True)
train_X = np.array(train_X)
test_X = np.array(test_X)
train_label = np.array(train_label)
test_label = np.array(test_label)
# valid_label = np.array(valid_label)
# valid_X = np.array(valid_X)
train_X = train_X.reshape(train_X.shape[0],7,50,1)
test_X = test_X.reshape(test_X.shape[0],7,50,1)
model = Sequential()
#add model layers
model.add(TimeDistributed(Conv1D(128, kernel_size=1, activation='relu', input_shape=(None,50,1))))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(256, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Conv1D(512, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(2)))
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200,return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_label, validation_data=(test_X,test_label), epochs=200)
print(model.summary())
print(model.evaluate(test_X,test_label))
# model.summary()
# predicted  = model.predict(test_X)
# test_label = (test_label[:,0])
# predicted = np.array(predicted[:,0]).reshape(-1,1)
# for j in range(len_t , len_t + len(test_X)):
#     temp =stock.iloc[j,4]
#     test_label[j - len_t] = test_label[j - len_t] * temp + temp
#     predicted[j - len_t] = predicted[j - len_t] * temp + temp
# plt.plot(test_label, color = 'black', label = ' Stock Price')
# plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
# plt.title(' Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel(' Stock Price')
# plt.legend()
# plt.show()
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
