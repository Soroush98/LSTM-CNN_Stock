import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Conv1D,MaxPool1D,Bidirectional,LSTM,Dropout,TimeDistributed
from keras.layers import Dense,GlobalAveragePooling2D
import matplotlib.pyplot as plt
filename = 'AAPL'
stock = pd.read_csv('Data/' + filename + '.csv')
scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,1:4])
stock.iloc[:,1:4] = scaled_values

y_scaler = preprocessing.MinMaxScaler()
scaled_values = y_scaler.fit_transform(np.array(stock.iloc[:,4]).reshape(-1,1))
stock.iloc[:,4] = scaled_values

scaler = preprocessing.MinMaxScaler()
scaled_values = scaler.fit_transform(stock.iloc[:,5:])
stock.iloc[:,5:] = scaled_values
#
# y_trans = preprocessing.MinMaxScaler()
# temp = y_trans.fit_transform(np.array(stock.iloc[:,4]).reshape(-1,1))


window_size = 50
week = 7
X = []
Y = []

for i in range(0 , len(stock) - window_size -week , 1):
    X.append(np.array(stock.iloc[i:i+window_size,4]).reshape(50,1))
    Y.append(np.array(stock.iloc[i+window_size:i+window_size+week,4]).reshape(week,1))
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
model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_label, validation_split=0.2, epochs=10)
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