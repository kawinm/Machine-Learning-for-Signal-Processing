#!/usr/bin/env bash

SQUADDIR=SQuAD
GLOVE_DIR=glove
mkdir -p $SQUADDIR
mkdir -p $GLOVEDIR

URLS=(
    "http://nlp.stanford.edu/data/glove.840B.300d.zip"
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
)
FILES=(
    "$GLOVEDIR/glove.840B.300d.zip"
    "$SQUADDIR/train-v1.1.json"
    "$SQUADDIR/dev-v1.1.json"
)
# Iterate through every dataset urls
for ((i=0;i<${#URLS[@]};++i)); do
    # Get the file name to save to
    file=${FILES[i]}
    # URL of the corresponding dataset file.
    url=${URLS[i]}
    
    # If the file already exists then do not download.
    if [ -f $file ]; then
        echo "$file already exists, skipping download."
    else
        # Using wget utility to obtain 
        wget $url -O $file
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        # Unzip the file if the file  has a .zip extension.
        if [ ${file: -4} == ".zip" ]; then
            unzip $file -d "$(dirname "$file")"
        fi
    fi
done

# Download the Spacy Language Model.
python3 -m spacy download en_core_web_md

