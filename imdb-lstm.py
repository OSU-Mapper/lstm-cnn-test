'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, TimeDistributed
from keras.layers import LSTM

max_features = 20000
max_len = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train) = (
    [[
        9,3,7,6
    ]],
    [[0.1, 0.2, 0.3]]
)
# (x_test, y_test) = (

# )
print(len(x_train), 'train sequences')
#print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
#x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(LSTM(64, input_shape=(max_len, 1),
                 dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(1, activation='sigmoid'))
model.add(TimeDistributed(Dense(1)))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          verbose=1,
          validation_data=(x_train, y_train))
score, acc = model.evaluate(x_train, y_train,
                            verbose=1,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)