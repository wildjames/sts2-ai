#!/bin/bash

# Preprocess once, then train with different C values
DATASET_DIR="data/dataset"
poetry run sts2-card-pick preprocess -o "$DATASET_DIR" --cards-json data/cards.json --relics-json data/relics.json data/a10_training_data.jsonl

CMD_TEMPLATE="poetry run sts2-card-pick train -o data/models/model_cFNAME -C CVAL DATASET_DIR"


c_values=(
  0.01
  0.1
  # 0.5
  # 1.0
  # 2.0
  # 5.0
  # 10.0
  # 100.0
)


for c in ${c_values[@]}; do
  fname=$(echo $c | sed 's/\./_/g')
  cmd=${CMD_TEMPLATE//CVAL/$c}
  cmd=${cmd//FNAME/$fname}
  cmd=${cmd//DATASET_DIR/$DATASET_DIR}
  echo "Running command: $cmd"
  eval $cmd &
done
wait
