#!/bin/bash

image_dir=./frames
output_dir=./all_images
image_filelist_path=image_paths_continue.txt
output_filepath=results_continue.txt

if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

echo "Finding all images..."
#find $image_dir -name "*.jpg" 2>&1 | tee $output_dir/$image_filelist_path

echo "Classifying all images..."
CUDA_VISIBLE_DEVICES=0 python ./scripts/classify_image.py --image_filelist $output_dir/$image_filelist_path --output_filepath $output_dir/$output_filepath
