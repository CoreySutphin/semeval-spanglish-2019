"""
Initial CNN model
 @author Cove Soyars
"""

import pickle
import numpy as np

from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences

# load data from file:
data = pickle.load(open('../data/data.p', 'rb'))

# make mapping of chars -> index
chars = sorted(set(list("".join([i[1] for i in data]))))
char_index = dict((c, i+1) for i, c in enumerate(chars))  # save 0 to be a padding number (i+1)
vocab_size = len(char_index)

# encode tweets with their char indices:
encoded_tweets = []
for tweet in data:
    encoded_tweets.append([char_index[char] for char in tweet[1]])

# pad sequences
max_length = len(max(encoded_tweets, key=len))  # get length of longest tweet
encoded_tweets = pad_sequences(encoded_tweets, maxlen=max_length, padding="post")

# convert these to one-hot vectors:
encoded_tweets = [to_categorical(x, num_classes=vocab_size + 1) for x in encoded_tweets]
X = np.array(encoded_tweets)
y = to_categorical([i[0] for i in data], num_classes=3)

# model:
model = Sequential()
model.add(Conv1D(kernel_size=3, filters=128, input_shape=(max_length, vocab_size+1)))
model.add(MaxPooling1D())
model.add(Flatten())
# ADD LSTM HERE?
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, verbose=2)

