#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify place or object"
    exit
fi

place_or_object=$1
file_dir=./mp3_files/mp3_files_${place_or_object}/

if [ ! -d $file_dir ]; then
    mkdir $file_dir
fi

for f in ./training_and_testing_data_list/training_data_${place_or_object}.txt ./training_and_testing_data_list/testing_data_${place_or_object}.txt
do
    while read -r line
    do
        echo $line
        mv $line $file_dir
    done < $f
done
