"""
    @author Corey Sutphin,
    @author Cove Soyars
    @author Nick Rodriguez
    Parse a CONLL document, converting each sentence to a list of objects representing
    the tokens.
    Created for the 2019 SemEval Challenge.

    Args: CONLL file (1), remove_users (2): y or n, remove_lang (3): y or n,
          remove_stop_words(4): y or n, convert_emojis(5) y or n
"""

import sys
import os
import re
import pandas as pd
import emoji


assert len(sys.argv) > 1, "Must pass in CONLL file to convert"

# interpret command line args:
conll_file = sys.argv[1]
remove_users = True if sys.argv[2].lower() == "y" else False
remove_lang = True if sys.argv[3].lower() == "y" else False
remove_stop_words = True if sys.argv[4].lower() == 'y' else False
convert_emojis = True if sys.argv[5].lower() == 'y' else False

if remove_stop_words:
    with open('spanish-stopwords.txt') as spanish_file:
        stopwords_es = [line.rstrip('\n') for line in spanish_file]
    with open('english-stopwords.txt') as english_file:
        stopwords_en = [line.rstrip('\n') for line in english_file]

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
            if remove_stop_words:
                if word in stopwords_en or word in stopwords_es:
                    continue

            if convert_emojis:
                word = emoji.demojize(word)

            if word.startswith('@'):
                if remove_users:
                    word = "@"

            if remove_lang:
                tweet.append(word)
            else:
                tweet.append((word,language))

        tweets.append(tweet)

print(label_counts)
tweets_df = pd.DataFrame(tweets)


tweets_df.to_csv('train_spanglish.csv', index=False, header=False)
