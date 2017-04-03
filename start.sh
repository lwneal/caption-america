#!/bin/bash

echo "Downloading the COCO dataset"
mkdir $HOME/data
./download_coco.sh $HOME/data
python load_grefexp_to_redis.py $HOME/data
python load_coco_to_redis.py $HOME/data
python train.py caption.py
