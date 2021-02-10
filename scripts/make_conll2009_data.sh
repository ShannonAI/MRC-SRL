#!/bin/bash

CONLL09_PATH=$1

SRL_PATH="./data/conll2009"

if [ ! -d $SRL_PATH ]; then
  mkdir -p $SRL_PATH
fi


if [ ! -d $SRL_PATH ]; then
  mkdir -p $SRL_PATH
fi

# Convert CoNLL to json format.
python ./scripts/conll09_to_json.py "${CONLL09_PATH}/CoNLL2009-ST-English-train.txt"  "${SRL_PATH}/train.english.conll09.jsonlines"
python ./scripts/conll09_to_json.py "${CONLL09_PATH}/CoNLL2009-ST-English-development.txt"  "${SRL_PATH}/dev.english.conll09.jsonlines"
python ./scripts/conll09_to_json.py "${CONLL09_PATH}/CoNLL2009-ST-evaluation-English.txt"  "${SRL_PATH}/test_wsj.english.conll09.jsonlines"
python ./scripts/conll09_to_json.py "${CONLL09_PATH}/CoNLL2009-ST-evaluation-English-ood.txt"   "${SRL_PATH}/test_brown.english.conll09.jsonlines"

python ./scripts/preprocess.py  "${SRL_PATH}/train.english.conll09.jsonlines"  "${SRL_PATH}/train.english.conll09.json" 
python ./scripts/preprocess.py  "${SRL_PATH}/dev.english.conll09.jsonlines"  "${SRL_PATH}/dev.english.conll09.json"
python ./scripts/preprocess.py  "${SRL_PATH}/test_wsj.english.conll09.jsonlines"  "${SRL_PATH}/test_wsj.english.conll09.json"
python ./scripts/preprocess.py  "${SRL_PATH}/test_brown.english.conll09.jsonlines"  "${SRL_PATH}/test_brown.english.conll09.json"