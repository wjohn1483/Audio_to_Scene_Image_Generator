#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify place or object"
    exit
fi

place_or_object=$1

tar zxvf ./mp3_public.tar.gz -T ./training_and_testing_data_list/training_data_${place_or_object}.txt
bash ./scripts/move_files.sh $place_or_object
rm -r mp3
cd ./soundnet_tensorflow/
(time bash extract_feat.sh $place_or_object) 2>&1 | tee ./log
echo "Remember to split training set and testing set by yourself !!!"
