'''prepare the sentences for character embedding (CNN)
inspired from :
https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/
'''
import pandas as pd
import pandas as pd
import numpy as np

# read in data
data = pd.read_csv('../data/dev/cmsa_dev.csv', header=None)
word_to_lang = pd.read_csv('../data/langs.csv', header=None,
                           dtype={0: str, 1: str},
                           names=['word', 'lang'])

words = word_to_lang['word'].values.tolist()
chars = set([c for w in words for c in str(w)])
char_count = len(chars)
char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

sentences = []
for i, sent in data.iloc[:,1:].iterrows():
    sentences.append(sent.dropna().tolist())

max_len_char = max([len(str(x)) for x in words])
max_len = max([len(sent) for sent in sentences])

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
