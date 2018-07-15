#!/bin/bash

soundnet_mp3_url="http://data.csail.mit.edu/soundnet/mp3_public.tar.gz"
soundnet_mp3_tar_filename="mp3_public.tar.gz"
soundnet_frames_url="http://data.csail.mit.edu/soundnet/frames_public.tar.gz"
soundnet_frames_tar_filename="frames_public.tar.gz"
soundnet_frame_directory="./frames"

# Download SoundNet dataset
if [ ! -f $soundnet_mp3_tar_filename ]; then
    wget $soundnet_mp3_url
fi

if [ ! -d $soundnet_frame_directory]; then
    wget $soundnet_frames_url
fi

tar zxvf $soundnet_frames_tar_filename

class_list=("plane" "speedboat" "piano" "dam" "baseball" "soccer" "guitar" "drum" "dog")
for var in ${class_list[*]};
do
done

