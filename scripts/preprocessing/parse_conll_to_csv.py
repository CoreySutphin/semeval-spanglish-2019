"""
    Parse a CONLL document, converting each sentence to a list of objects representing
    the tokens.
    Created for the 2019 SemEval Challenge.

    Args: CONLL file (1), remove_users (2): y or n, remove_lang (3): y or n.
"""

import sys
import os
import re
import pandas as pd

assert len(sys.argv) > 1, "Must pass in CONLL file to convert"

# interpret command line args:
conll_file = sys.argv[1]
remove_users = True if sys.argv[2].lower() == "y" else False
remove_lang = True if sys.argv[3].lower() == "y" else False

label_counts = {}
tweets = []

with open(conll_file, 'r') as file:
    sentences = re.split('\n\n', file.read())

    for sentence in sentences:
        if sentence == '':
            continue
        lines = sentence.split('\n')
        meta_info = lines[0].split('\t')
        label = meta_info[2]
        tweet = [label]

        # Record label frequency in the data
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

        # Create a list of objects representing the tokens
        for token in lines[1:]:
            word, language = token.split('\t')
            if word.startswith('@'):

                if remove_users:
                    word = "USER"
            if remove_lang:
                tweet.append(word)
            else:
                tweet.append((word,language))
        tweets.append(tweet)

print(label_counts)
tweets_df = pd.DataFrame(tweets)


tweets_df.to_csv('my_csv.csv', index=False, header=False)
