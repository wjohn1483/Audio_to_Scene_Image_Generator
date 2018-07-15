#!/bin/bash

file_index=$(($1 + 1))
dir_title="plane"
data_list=./training_and_testing_data_list/testing_data_plane_keywords.txt
filename=`echo "sed '${file_index}q;d' $data_list" | bash | rev | cut -d '/' -f 1 | rev | cut -d '.' -f 1`

echo "$filename"
xdg-open ./mp3_files/mp3_files_$dir_title/$filename.mp4.mp3 &
xdg-open ./images/images_${dir_title}_size64/$filename.jpg
