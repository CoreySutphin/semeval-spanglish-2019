#!/usr/bin/env bash

echo Running char embedding CNN...
cd initial_CNN
python initial_CNN.py

echo Running word embedding CNN...
cd ../scripts
python word_embeddings_CNN.py

