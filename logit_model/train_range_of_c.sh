#!/bin/bash

# Preprocess once, then train with different C values
TRAIN_DATASET="data/a8_9_10_wins_train"
TRAIN_CMD="poetry run logit-model train --gpu -o data/models/model_cFNAME -C CVAL TRAIN_DATASET"

EVAL_DATASET="data/a8_9_10_wins_eval"
EVAL_CMD="poetry run logit-model evaluate --gpu data/models/model_cFNAME EVAL_DATASET"


c_values=(
  0.01
  0.1
  0.5
  1.0
  2.0
  5.0
  10.0
  100.0
)


for c in "${c_values[@]}"; do
  fname=$(echo $c | sed 's/\./_/g')
  cmd=${TRAIN_CMD//CVAL/$c}
  cmd=${cmd//FNAME/$fname}
  cmd=${cmd//TRAIN_DATASET/$TRAIN_DATASET}
  echo "Running command: $cmd"
  eval "$cmd"

  fname=$(echo $c | sed 's/\./_/g')
  cmd=${EVAL_CMD//CVAL/$c}
  cmd=${cmd//FNAME/$fname}
  cmd=${cmd//EVAL_DATASET/$EVAL_DATASET}
  echo ""
  echo "Running command: $cmd"
  eval "$cmd"
done
