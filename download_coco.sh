#!/bin/bash
set -e

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 DATA_DIR/"
    echo "DATA_DIR: Directory for dataset download eg /home/nealla/data/"
    exit
fi
DATA_DIR=$1


pushd $DATA_DIR

echo "Downloading COCO dataset from windows.net, total size is ~40GB. Please be patient."
wget -N "http://msvocds.blob.core.windows.net/coco2014/train2014.zip"
wget -N "http://msvocds.blob.core.windows.net/coco2014/val2014.zip"
wget -N "http://msvocds.blob.core.windows.net/coco2014/test2014.zip"
wget -N "http://msvocds.blob.core.windows.net/coco2015/test2015.zip"
wget -N "http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip"
wget -N "http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip"
wget -N "http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
wget -N "http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip"
wget -N "http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2015.zip"
echo "COCO dataset has been downloaded"


echo "Downloading Google gRefExp dataset..."
wget -N "https://storage.googleapis.com/refexp/google_refexp_dataset_release.zip"
echo "GRefExp dataset has been downloaded"

echo "Downloading Visual Genome annotations for COCO"
wget -N "http://visualgenome.org/static/data/dataset/image_data.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/region_descriptions.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/question_answers.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/objects.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/attributes.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/relationships.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/synsets.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/region_graphs.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/scene_graphs.json.zip"
wget -N "http://visualgenome.org/static/data/dataset/qa_to_region_mapping.json.zip"
echo "Finished downloading Visual Genome"

echo "Unzipping all files"
mkdir -p coco
unzip -d coco train2014.zip
unzip -d coco val2014.zip
unzip -d coco test2014.zip
unzip -d coco test2015.zip
unzip -d coco instances_train-val2014.zip
unzip -d coco person_keypoints_trainval2014.zip
unzip -d coco captions_train-val2014.zip
unzip -d coco image_info_test2014.zip
unzip -d coco image_info_test2015.zip

mkdir grefexp
unzip -d grefexp google_refexp_dataset_release.zip
echo "All files unzipped"

popd
