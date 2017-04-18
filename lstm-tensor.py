import numpy as np

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Flatten
from keras.layers import LSTM

from keras.layers import Conv1D, MaxPooling2D

# Input sequence
wholeSequence = [[0,0,0,0,0,0,0,0,0,2,1],
                 [0,0,0,0,0,0,0,0,2,1,0],

                 [0,0,0,0,0,0,0,2,1,0,0],
                 [0,0,0,0,0,0,2,1,0,0,0],
                 [0,0,0,0,0,2,1,0,0,0,0],
                 [0,0,0,0,2,1,0,0,0,0,0],

                 [0,0,0,2,1,0,0,0,0,0,0],
                 [0,0,2,1,0,0,0,0,0,0,0],
                 [0,2,1,0,0,0,0,0,0,0,0],
                 [2,1,0,0,0,0,0,0,0,0,0]]

Y_test = [
    [7,6,5,4],
    [0,1,2,3]
]
# Preprocess Data:
wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
data = wholeSequence[:-2] # all but last
# target = wholeSequence[0:2] 
target= np.array(Y_test, dtype=float)


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
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=True))
model.add(LSTM(11, input_shape=(4, 11), unroll=True, return_sequences=False))
model.add(Dense(4))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, target, nb_epoch=1000, batch_size=1, verbose=1)
