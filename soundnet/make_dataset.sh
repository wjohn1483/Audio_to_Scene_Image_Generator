#!/bin/bash

soundnet_mp3_url="http://data.csail.mit.edu/soundnet/mp3_public.tar.gz"
soundnet_mp3_tar_filename="mp3_public.tar.gz"
soundnet_frames_url="http://data.csail.mit.edu/soundnet/frames_public.tar.gz"
soundnet_frames_tar_filename="frames_public.tar.gz"
soundnet_frame_directory="./frames"

mp3_file_dir="./mp3_files"
mp3_soundnet_feat_dir="./mp3_soundnet_feat"
image_dir="./images"
filelist_dir="./training_and_testing_data_list"

# Download SoundNet dataset
if [ ! -f $soundnet_mp3_tar_filename ]; then
    wget $soundnet_mp3_url
fi

if [ ! -d $soundnet_frame_directory ]; then
    wget $soundnet_frames_url
    tar zxvf $soundnet_frames_tar_filename
fi

# Extract sound files from compressed file
if [ ! -d $mp3_file_dir ]; then
    mkdir $mp3_file_dir
fi
if [ ! -d $mp3_soundnet_feat_dir ]; then
    mkdir $mp3_soundnet_feat_dir
fi
if [ ! -d $image_dir ];then
    mkdir $image_dir
fi

class_list=("plane" "speedboat" "piano" "dam" "baseball" "soccer" "guitar" "drum" "dog")
for var in ${class_list[*]};
do
    bash ./scripts/extract_files_and_feats.sh ${var}_keywords
    if [ -f tmp.txt ];then
        rm tmp.txt
    fi
    cat $filelist_dir/training_data_${var}_keywords.txt $filelist_dir/testing_data_${var}_keywords.txt > tmp.txt
    bash ./scripts/copy_images_given_file.sh tmp.txt $image_dir/images_${var}_keywords
    rm tmp.txt
done

