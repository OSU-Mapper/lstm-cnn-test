# -*- coding: utf-8 -*-
"""
Created on Mon May  1 05:38:04 2017

@author: mw352
"""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Flatten
from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling2D

# Input sequence
X_test = np.loadtxt("D:\\tandon\\AI_for_game\\osu_project\\training_input.csv",delimiter=",",skiprows=0)
X_test = X_test[:,0:3]
Y_test = np.loadtxt("D:\\tandon\\AI_for_game\\osu_project\\training_output.csv",delimiter=",",skiprows=0)
Y_test = Y_test[:,-1:]
# Preprocess Data:
X_test = np.array(X_test, dtype=float) # Convert to NP array.

# target = wholeSequence[0:2] 
Y_test = np.array(Y_test, dtype=float)
X_test = X_test.reshape((500, 240, 3))
Y_test = Y_test.reshape((500, 240, 1))

#build lstm model
model = Sequential()
model.add(LSTM(32, input_shape=(240,3),activation='tanh', recurrent_activation='hard_sigmoid',
                            use_bias=True, kernel_initializer='glorot_uniform',
                            recurrent_initializer='orthogonal', bias_initializer='zeros',
                            unit_forget_bias=True, 
                            return_sequences=True))

model.add(LSTM(32,return_sequences=True))
#model.add(LSTM(32,input_length = 24000, input_dim=3,return_sequences=True))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_test, Y_test, epochs=20, batch_size=16, verbose=1)
X_target = X_test[0:100]
predict = model.predict(X_target, batch_size = 32, verbose = 1)
print (predict)
#np.savetxt('D:\\tandon\\AI_for_game\\osu_project\\training_result.csv', predict, delimiter = ',') 
'''
print("data",data)
print("target",target)
# Reshape training data for Keras LSTM model
# The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
# Single batch, 9 time steps, 11 dimentions
data = data.reshape((2, 4, 11))
target = target.reshape((2, 4))
print("data",data)
print("target",target)

# Build Model
model = Sequential() 
# model.add(Conv1D(11, kernel_size=2,
#                  activation='relu',
#                  input_shape=(2, 4, 11)))
# model.add(Flatten())

model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=False))
model.add(Dense(4))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, target, epochs=10, batch_size=1, verbose=1)
'''