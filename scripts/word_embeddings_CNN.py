"""
@author Corey Sutphin
Utilizing English + Spanish word embeddings with a CNN fed into two dense layers.

English embeddings are GloVe 6b 300d.
Spanish embeddings are trained on the Spanish Billion Words Corpus using word2vec.
Tweets have been pre-processed into a pickled file.

FUTURE WORK:
1. Use contextual embeddings(BERT)
2. Experiment with different network parameters.
3. Some words in the spanish embeddings could be overwriting the vectors for some english
   words. For example 'con' has two different meaning in spanish and english.
4. When doing the 60/40 split for training/validation, it is completely random and
   there is no guarantee of equal class distribution across the testing and validation sets.
"""
import re
import numpy as np
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.metrics import Precision, Recall

EMBEDDING_DIM = 300
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.4  # Reserve 40% of tweets for validation


# First, build index mapping words in the embeddings set
# to their embedding vector.
print('\nIndexing word vectors.')

embeddings_index = {}
with open('../data/SBW-vectors-300-min5.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

with open('../data/glove.6B.300d.txt') as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

# Prepare our data set
print('\nProcessing tweets')

tweets = pickle.load(open('../data/data.p', 'rb'))

# Vectorize the tweets into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts([tweet[1] for tweet in tweets])
sequences = tokenizer.texts_to_sequences([tweet[1] for tweet in tweets])
labels = [tweet[0] for tweet in tweets]

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

max_length = len(max(tweets, key=len))  # Get length of longest tweet
data = pad_sequences(sequences, maxlen=max_length)
labels = to_categorical(np.asarray(labels), num_classes=3)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print('Embedding Matrix:', embedding_matrix)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)

print('Training model.')

# Train a 1D convnet with global maxpooling
sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(filters=128, kernel_size=3, input_shape=(max_length, len(word_index) + 1), data_format='channels_first')(embedded_sequences)
x = MaxPooling1D()(x)
x = Conv1D(filters=128, kernel_size=3, input_shape=(max_length, len(word_index) + 1), data_format='channels_first')(x)
x = MaxPooling1D()(x)
x = Conv1D(filters=128, kernel_size=3, input_shape=(max_length, len(word_index) + 1), data_format='channels_first')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(3, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', Precision(), Recall()])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
