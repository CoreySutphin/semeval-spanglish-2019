from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
# tf.enable_eager_execution()
# from tensorboard.plugins.hparams import api as hp
from keras.utils import to_categorical
# import talos
from model import model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def main():


    save_char_array = False
    char_output = 'char_matrix_dev'

    # read in data
    data = pd.read_csv('../../data/train/cmsa_train.csv', header=None)
    sentences = []
    for i, sent in data.iloc[:, 1:].iterrows():
        sentences.append(sent.dropna().tolist())

    # word embeddings:
    embeddings_index = {}
    with open('../../data/SBW-vectors-300-min5.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    with open('../../data/glove.6B.300d.txt') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    EMBEDDING_DIM = 300
    MAX_NUM_WORDS = 20000

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts([" ".join(sentence) for sentence in sentences])
    sequences = tokenizer.texts_to_sequences([" ".join(sentence) for sentence in sentences])
    max_len = max([len(sent) for sent in sentences])
    word_index = tokenizer.word_index

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    print('building word embedding matrix...')
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    X_word = pad_sequences(sequences, maxlen=max_len, padding='post')

    # character embeddings:
    word_to_lang = pd.read_csv('../../data/langs.csv', header=None,
                               dtype={0: str, 1: str},
                              names=['word', 'lang'])
    words = word_to_lang['word'].values.tolist()
    max_len_char = max([len(str(x)) for x in words])


    chars = set([c for w in words for c in str(w)])
    n_chars = len(chars)
    char_count = len(chars)
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0
    # Plot density of word lengths
    pd.DataFrame([len(str(x)) for x in words]).plot.density()



    ''' Creates a list of arrays. Each array represents
    one sentence where each row is a word, each column each is a character in that word.
    Words and sentences are padded with zeros according to their max length.
    '''
    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    X_char = np.array(X_char)

    if save_char_array:
        np.save(file=char_output, arr=X_char)

    # One Hot Encode labels
    raw_labels = data[0]
    label_encoder = LabelEncoder()
    label_ints = label_encoder.fit_transform(raw_labels)
    label_one_hot = to_categorical(label_ints)

    # Split Data
    char_train, char_test, label_train, label_test = train_test_split(X_char, label_one_hot, test_size=.1, random_state=0)
    word_train, word_test, _, _ = train_test_split(X_word, label_one_hot, test_size=.1, random_state=0)

    ### Parameter Tune
    iterations = 10
    p = {
        'max_num_words': [max_len]*iterations,
        'max_chars_in_word': [max_len_char]*iterations,
        'num_of_unique_chars': [n_chars]*iterations,
        'lstm_units_char_emb': np.random.randint(10, 50, iterations),
        'dropout_rate_char_emb': [10**np.random.uniform(-0.3, -2) for x in range(iterations)],
        'bilstm_units': np.random.randint(5, 100, iterations),
        'bilstm_dropout_rate': [10**np.random.uniform(-0.3, -2) for x in range(iterations)],
        'epochs': np.random.randint(10, 30, iterations),
        'batch_size': np.random.randint(10, 20, iterations),
        'vocab_size': [num_words]*iterations
    }

    assert all([len(p_i) == iterations for p_i in p.values()])
    p_i = {key: p[key][0] for key in p}
    for _ in p_i.items(): print(_)
    model([word_train, char_train], label_train, embedding_matrix, params=p_i, fit_model=True)

main()
