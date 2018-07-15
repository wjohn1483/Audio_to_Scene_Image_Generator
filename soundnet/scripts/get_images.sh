#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify place or object name"
    exit
fi

place_object=$1

# Copy images classified by SoundNet
zsh ./scripts/get_filenames_of_specific_place_or_object.sh $place_object > ./indexs/${place_object}_id.txt
zsh ./scripts/transform_number_to_path.sh ./indexs/${place_object}_id.txt ./training_and_testing_data_list/training_data_${place_object}.txt
if [ -d ./images/images_${place_object} ]; then
    rm -r ./images/images_${place_object}
    mkdir ./images/images_${place_object}
else
    mkdir ./images/images_${place_object}
fi
bash ./scripts/copy_images_given_file.sh ./training_and_testing_data_list/training_data_${place_object}.txt ./images/images_${place_object}
