# semeval-spanglish-2019

This repo contains the data, scripts, and results from the SemEval 2019/2020 Sentimix Spanglish challenge.

## Sentiment Analysis on Code-Mixed Tweets

The challenge deals with the problem of identifying the sentiment of a set of 'Spanglish' tweets, where Spanglish is a mix of English and Spanish. Misspellings, new words, mixed grammar, and the short length of tweets make this task difficult, and currently prevailing methods of using pre-trained contextual word embeddings may not be as effective. Vocabulary and embedding sizes will have to be large to accommodate both languages, and the odds of running into an out-of-vocabulary word is very high.

We have ran experiments using one-hot encoded character embeddings, concatenated Spanish + English word embeddings.

Spanish + English Word Embeddings
3 CNN layers, Max Pooling, 2 Dense Layers for classification
Accuracy: 0.5378 - Precision: 0.6100 - Recall: 0.3348

One-Hot Encoded Character Embeddings
1 CNN layer, Max Pooling, 2 Dense Layers for classification

## Examples of Problem

TODO

##

## Roles

Corey Sutphin - Preprocessing scripts, model utilizing Spanish word embeddings, English word embeddings, and then the two concatenated on each other.
Cove Soyars - Preprocessing scripts, model using one-hot encoded character embeddings with a CNN.
Nick Rodriguez - Preprocessing scripts, BiLSTM model
