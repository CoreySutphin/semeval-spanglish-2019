#!/usr/bin/env bash
# Author: Cove Soyars
python -m venv semeval
source semeval/bin/activate
pip install -r requirements.txt

echo Running char embedding CNN...
cd initial_CNN
python initial_CNN.py

echo Running word embedding CNN...
cd ../scripts
python word_embeddings_CNN.py

