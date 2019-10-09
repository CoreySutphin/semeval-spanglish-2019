# semeval-spanglish-2019

This repo contains the data, scripts, and results from the SemEval 2019/2020 Sentimix Spanglish challenge.

## Sentiment Analysis on Code-Mixed Tweets

The challenge deals with the problem of identifying the sentiment of a set of 'Spanglish' tweets, where both English and Spanish are used in a single tweet. Misspellings, new words, mixed grammar, and the short length of tweets make this task difficult, and currently prevailing methods of using pre-trained contextual word embeddings may not be as effective. Vocabulary and embedding sizes will have to be large to accommodate both languages, and the odds of running into an out-of-vocabulary word are very high.

We have ran experiments using one-hot encoded character embeddings and concatenated Spanish + English word embeddings.

Spanish + English Word Embeddings
3 CNN layers, Max Pooling, 2 Dense Layers for classification

| Precision | Recall | F1     |
| --------- | ------ | ------ |
| 0.61      | 0.3348 | 0.4323 |

One-Hot Encoded Character Embeddings
1 CNN layer, Max Pooling, 2 Dense Layers for classification

| Precision | Recall | F1   |
| --------- | ------ | ---- |
| 0.42      | 0.56   | 0.41 |

## Roles

Corey Sutphin - Preprocessing scripts, model utilizing Spanish word embeddings, English word embeddings, and then the two concatenated on each other.
Cove Soyars - Preprocessing scripts, bash script, model using one-hot encoded character embeddings with a CNN.
Nick Rodriguez - Preprocessing scripts, BiLSTM model
