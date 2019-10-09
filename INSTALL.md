# How to Run Experiments

1. Create a virtual environment for the project using `python -m venv semeval`.
2. In the root folder of the repo, run `pip install -r requirements.txt`. This will install all required python dependencies.
3. The results of our pre-processing scripts are contained in a pickled file located at `/data/data.p`. This file should already exist in the repo, but to regenerate it:

```
cd scripts/preprocessing
python parse_conll_to_csv ../../data/train_conll_spanglish.txt
python data_prep.py ./train_spanglish.csv
mv data.p ../../data
```

4. We utilize GloVe English word embeddings and Spanish word embeddings trained with Word2Vec on the 'Spanish Billion Words Corpus'. To run the word embeddings model you will need to install these embeddings and place them in the 'data' folder. http://nlp.stanford.edu/data/glove.6B.zip - Copy the `glove.6B.300d.txt` file into the `data` folder.
   https://www.kaggle.com/rtatman/pretrained-word-vectors-for-spanish - Copy this file into the `data` folder.

## Running Word Embeddings Model

From root:

```
cd scripts
python word_embeddings_CNN.py
```

## Running One-Hot Char Embeddings

From root:

```
cd initial_CNN
python initial_CNN.py
```
