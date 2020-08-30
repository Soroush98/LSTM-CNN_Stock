import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Bidirectional,LSTM,Dropout
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
filename = 'TSLA'
stock = pd.read_csv('Data/' + filename + '.csv')


model = Sequential()
#add model layers
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2))
model.add((2))
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout)

model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
model.summary()
