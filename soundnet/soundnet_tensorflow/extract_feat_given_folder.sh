#!/bin/bash


if [ -z "$1" ]; then
    echo "Need to specify input folder"
    exit
fi

if [ -z "$2" ]; then
    echo "Need to specify output folder"
    exit
fi

file_dir=$1
file_list=./mp3_list.txt
segment_dir=./segments/
output_dir=$2

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
    CUDA_VISIBLE_DEVICES="" python3 ./extract_feat.py -c 1 -m 18 -x 19 -s -p extract -o $output_dir -t $f
done


