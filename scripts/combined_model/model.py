from keras.models import Model, Input
from keras.layers import TimeDistributed, LSTM
from keras.layers import Embedding, Bidirectional, Dense, Add, concatenate, Flatten
from keras.initializers import Constant



def model(x_train, y_train, embed_matrix, params=None, fit_model=True):
    ''' params is a dictionary containing hyperparameter values. See main.py
    for current definition.
    '''
    # input and embeddings for characters
    word_in = Input(shape=(params['max_num_words'],))

    emb_word = Embedding(input_dim=params['vocab_size'], output_dim=300,
						 input_length=params['max_num_words'], mask_zero=True,
                         embeddings_initializer=Constant(embed_matrix))(word_in)

    char_in = Input(shape=(params['max_num_words'], params['max_chars_in_word']),
                    name='input')
    emb_char = TimeDistributed(Embedding(input_dim=params['num_of_unique_chars']+2,
                                        output_dim=params['lstm_units_char_emb'],
                                        input_length=params['max_chars_in_word'],
                                        mask_zero=True,
                                        trainable=True),
                               name='embed_dense_char')(char_in)
    emb_char = TimeDistributed(LSTM(units=params['lstm_units_char_emb'],
                                    return_sequences=False,
                                    recurrent_dropout=0.5),
                                 name='learn_embed_char')(emb_char)

    x = concatenate([emb_word, emb_char])
    bilstm = Bidirectional(LSTM(units=params['bilstm_units'],
                                 recurrent_dropout=0.5,
                                return_sequences=False),
                                merge_mode='sum')(x)


    bilstm = Dense(params['bilstm_units'], activation='relu',
                    name='linear_decode1')(bilstm)


    out = Dense(3, activation='softmax', name='output_softmax1')(bilstm)

    model = Model([word_in, char_in], out)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    if fit_model:
        history = model.fit(x_train, y_train,
                          batch_size=params['batch_size'], epochs=params['epochs'],
                          validation_split=0.1,
                          verbose=2)
        return history, model
    else:
        return model
