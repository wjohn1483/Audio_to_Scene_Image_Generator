#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify place or object"
    exit
fi

place_or_object=$1

if [ -f tmp.txt ]; then
    rm tmp.txt
fi
cat ./training_and_testing_data_list/training_data_${place_or_object}.txt ./training_and_testing_data_list/testing_data_${place_or_object}.txt > tmp.txt
tar zxvf ./mp3_public.tar.gz -T tmp.txt
rm tmp.txt
bash ./scripts/move_files.sh $place_or_object
rm -r mp3
cd ./soundnet_tensorflow/
(time bash extract_feat.sh $place_or_object `pwd`) 2>&1 | tee ./log
echo "Remember to split training set and testing set by yourself !!!"
