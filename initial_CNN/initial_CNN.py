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

from sklearn.metrics import classification_report


VALIDATION_SPLIT = 0.4  # Reserve 40% of tweets for validation

# load data from file:
data = pickle.load(open('data.p', 'rb'))

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

# Split the data into a training set and a validation set (modified from word_embeddings_CNN.py)
indices = np.arange(len(data))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
num_validation_samples = int(VALIDATION_SPLIT * len(data))

x_train = X[:-num_validation_samples]
y_train = y[:-num_validation_samples]
x_val = X[-num_validation_samples:]
y_val = y[-num_validation_samples:]

# model:
model = Sequential()
model.add(Conv1D(kernel_size=3, filters=128, input_shape=(max_length, vocab_size+1)))
model.add(MaxPooling1D())
model.add(Flatten())
# ADD LSTM HERE
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=2, validation_data=(x_val, y_val))

# make predictions for test set
predictions = model.predict(x_val)
truth = y_val

report = classification_report(y_val, predictions, labels=[0, 1, 2],
                               target_names=['positive', 'negative', 'neutral'])

print("See initial_CNN_results.txt for accuracy, precision and recall")
with open('initial_CNN_results.txt', 'w') as out:
    out.write(report)



