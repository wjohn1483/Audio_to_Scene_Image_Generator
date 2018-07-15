#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify input dir"
    exit
fi

if [ -z "$2" ]; then
    echo "Need to specify output dir"
    exit
fi

if [ "$1" = "$2" ]; then
    echo "Input dir and output dir cannot be the same"
    exit
fi

input_dir=$1
output_dir=$2
amplify_scale=0.5

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

for filename in `ls $input_dir`
do
    echo $filename
    lame -h --scale $amplify_scale $input_dir/$filename $output_dir/$filename
done
