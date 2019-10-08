#!/usr/bin/env bash

echo Running char embedding CNN...
cd initial_CNN
python3 initial_CNN.py

echo Running word embedding CNN...
cd cd ../scripts
python3 word_embeddings_CNN.py

