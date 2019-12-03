#!/usr/bin/env bash
# Author: Cove Soyars, Corey Sutphin
python -m venv semeval
source semeval/bin/activate
semeval/bin/pip install -r requirements.txt

echo Running word embedding CNN...
cd ./scripts
python word_embeddings_CNN.py

echo Running character LSTM + BiLSTM...
cd char_model
python char_model.py



# Remove the created virtual environment
cd ../
rm -rf semeval
