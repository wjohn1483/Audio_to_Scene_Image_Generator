#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to give place or object name"
    exit
fi

place_object=$1
classified_result=./all_images/results_all.txt
output_dir=./images/images_${place_object}_inception_keywords

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

paths=`grep $place_object $classified_result | cut -d ',' -f 1`
for path in $paths
do
    echo $path
    mp4_name=`echo "$path" | rev | cut -d '/' -f 2 | rev | cut -d '.' -f 1`
    filename=`echo "$path" | rev | cut -d '/' -f 1 | rev`
    cp $path $output_dir/${mp4_name}_$filename
done
ls $output_dir | wc -l
