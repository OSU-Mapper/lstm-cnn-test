from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=4)

print(x_train)
print(len(x_train))
print(len(x_train[0]))
#print(len(x_train[0][0]))
from keras.preprocessing import sequence
maxlen = 80
x_train = sequence.pad_sequences(x_train)
print('x_train shape:', x_train.shape)
