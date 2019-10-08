"""
Dumps data as list where first element is label of tweet and second element is text of tweet
@author: Cove Soyars

ISSUES:
1. apostrophes have not been combined with their words
2. removing double quotes (not sure why they are here)
3. extra spaces in between words and punctuation
"""

import pickle

data_file = "../scripts/preprocessing/train_spanglish.csv"
label_to_cls = {'positive': 0, 'negative': 1, 'neutral': 2}

sentences = []
with open(data_file, 'r') as csvin:
    # open data file and combine tweets into full sentences and grab their labels
    for line in csvin:
        line = line.split(',')
        try:
            label = label_to_cls[line[0]]
        except KeyError:   # skip lines with bad data
            continue

        sentence = " ".join(line[1:]).strip().replace("\"", "")  # remove quotes and trailing whitespace
        sentences.append([label, sentence])

pickle.dump(sentences, open('data.p', 'wb'))
