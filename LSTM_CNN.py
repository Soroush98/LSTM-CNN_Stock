import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D,MaxPool1D,Bidirectional,LSTM,Dropout
from keras.layers import Dense,GlobalAveragePooling2D
filename = 'AAPL'
stock = pd.read_csv('Data/' + filename + '.csv')
# scaler = preprocessing.StandardScaler()
# scaled_values = scaler.fit_transform(stock.iloc[:,1:])
# stock.iloc[:,1:] = scaled_values
window_size = 50
week = 7
X = []
Y = []

for i in range(0 , len(stock) - window_size -week , 1):
    first = stock.iloc[i, 4]
    temp = []
    temp2 = []
    for j in range(window_size):
        temp.append((stock.iloc[i + j, 4] - first) / first)
    for j in range(week):
        temp2.append((stock.iloc[i +window_size+ j, 4] - first) / first)
    # X.append(np.array(stock2.iloc[i:i+window_size,4]).reshape(50,1))
    #Y.append(np.array(stock2.iloc[i+window_size:i+window_size+week,4]).reshape(week,1))
    # print(stock2.iloc[i:i+window_size,4])
    X.append(np.array(temp).reshape(50, 1))
    Y.append(np.array(temp2).reshape(week,1))
train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size=0.2,shuffle=False)
train_X = np.array(train_X)
valid_X = np.array(valid_X)
train_label = np.array(train_label)
valid_label = np.array(valid_label)
model = Sequential()
#add model layers
model.add(Conv1D(128, kernel_size=5, activation='relu', input_shape=(50,1)))
model.add(MaxPool1D(2))
model.add(Conv1D(256, kernel_size=5, activation='relu'))
model.add(MaxPool1D(2))
model.add(Conv1D(512, kernel_size=5, activation='relu'))
model.add(MaxPool1D(2))
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(200,return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_label, validation_data=(valid_X, valid_label), epochs=50)
# model.summary()
