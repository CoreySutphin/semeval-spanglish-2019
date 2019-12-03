from numpy.random import seed
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from char_model import char_model
from sklearn.model_selection import StratifiedKFold
seed(1)


def main():
	save_char_array = False
	char_output = 'char_matrix_dev'

	# read in data
	data = pd.read_csv('../../data/dev/cmsa_dev.csv', header=None)
	sentences = []
	for i, sent in data.iloc[:, 1:].iterrows():
		sentences.append(sent.dropna().tolist())
		# character embeddings:
	max_len = max([len(sent) for sent in sentences])

	word_to_lang = pd.read_csv('../../data/langs.csv', header=None,
							   dtype={0: str, 1: str},
							   names=['word', 'lang'])
	words = word_to_lang['word'].values.tolist()
	max_len_char = max([len(str(x)) for x in words])

	chars = set([c for w in words for c in str(w)])
	n_chars = len(chars)
	char2idx = {c: i + 2 for i, c in enumerate(chars)}
	char2idx["UNK"] = 1
	char2idx["PAD"] = 0

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

	p = {
		'max_num_words': max_len,
		'max_chars_in_word': max_len_char,
		'num_of_unique_chars': n_chars,
		'lstm_units_char_emb': 50,
		'dropout_rate_char_emb': 0.5,
		'bilstm_units': 100,
		'bilstm_dropout_rate': 0.5,
		'epochs': 10,
		'batch_size': 32,
	}
	n_splits = 5
	current_fold = 1
	skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

	for train, test in skf.split(data, np.asarray(label_ints)):
		print(
			'==============================================  Fold %d  ===================================' % current_fold)
		y_train = label_one_hot[train]

		y_test = label_one_hot[test]
		X_tr_chars = X_char[train]
		X_te_chars = X_char[test]
		current_fold += 1

		char_model(X_tr_chars, y_train, X_te_chars, y_test, params=p,fit_model=True)


main()
