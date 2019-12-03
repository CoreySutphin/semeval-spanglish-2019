from keras.models import Model, Input
from keras.layers import TimeDistributed, LSTM
from keras.layers import Embedding, Bidirectional, Dense, Add
from keras.metrics import Precision, Recall
# from talos.metrics.keras_metrics import *

def char_model(x_train, y_train, x_test, y_test, params=None, fit_model=True):
    ''' params is a dictionary containing hyperparameter values. See main.py
    for current definition.
    '''
    # input and embeddings for characters
    char_in = Input(shape=(params['max_num_words'], params['max_chars_in_word']),
                    name='input')
    emb_char = TimeDistributed(Embedding(input_dim=params['num_of_unique_chars']+2,
                                        output_dim=params['lstm_units_char_emb'],
                                        input_length=params['max_chars_in_word'],
                                        mask_zero=True,
                                        trainable=True),
                               name='embed_dense_char')(char_in)
    emb_char = TimeDistributed(LSTM(units=params['lstm_units_char_emb'],
                                    return_sequences=False),
                                    # dropout=params['dropout_rate_char_emb']),
                                    name='learn_embed_char')(emb_char)
    bilstm = Bidirectional(LSTM(units=params['bilstm_units'],
                                # recurrent_dropout=params['bilstm_dropout_rate'],
                                return_sequences=False),
                            merge_mode='sum')(emb_char)


    dense = Dense(params['bilstm_units'], activation='relu',
                    name='linear_decode2')(bilstm)
    out = Dense(3, activation='softmax', name='output_softmax1')(dense)

    model = Model(char_in, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])

    if fit_model:
        history = model.fit(x_train, y_train,
                          batch_size=params['batch_size'], epochs=params['epochs'],
                          validation_data=(x_test, y_test),
                          verbose=2)
        return history, model
    else:
        return model
