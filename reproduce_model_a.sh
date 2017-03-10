#!/bin/bash
# If the COCO/grefexp dataset is not installed, run ./start.sh to download it

MODEL_NAME="reproduce_model_a.h5"
echo "Reproducing experiment A to generate trained network model $MODEL_NAME"
python train.py caption.py $MODEL_NAME
python evaluate.py caption.py $MODEL_NAME
