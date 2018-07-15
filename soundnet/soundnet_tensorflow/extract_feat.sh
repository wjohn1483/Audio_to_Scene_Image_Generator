#!/bin/bash


if [ -z "$1" ]; then
    echo "Need to specify place or object"
    exit
fi

place_or_object=$1
parent_dir=$2
file_dir=${parent_dir}/mp3_files/mp3_files_${place_or_object}/
file_list=./mp3_list.txt
segment_dir=./segments/
output_dir=${parent_dir}/mp3_soundnet_feat/mp3_soundnet_feat_${place_or_object}_18/

if [ ! -d "$segment_dir" ]; then
    mkdir "$segment_dir"
fi

if [ -d "$output_dir" ]; then
    rm -r $output_dir
    mkdir "$output_dir"
else
    mkdir "$output_dir"
fi

ls -d -1 $file_dir/* > $file_list
rm $segment_dir/*
split -l 5 $file_list $segment_dir/segment

for f in `ls $segment_dir/*`
do
    echo $f
    CUDA_VISIBLE_DEVICES=0 python3 ./extract_feat.py -m 18 -x 19 -s -p extract -o $output_dir -t $f
done


