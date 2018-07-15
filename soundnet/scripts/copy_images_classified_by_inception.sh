#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to give place or object name"
    exit
fi

place_object=$1

# Old version
#bash ./scripts/transform_number_to_path.sh ./indexs/${place_object}_id_contain_keywords.txt ./training_and_testing_data_list/training_data_${place_object}_keywords.txt
#mkdir ./images/images_${place_object}_keywords
#bash ./scripts/copy_images_given_file.sh ./training_and_testing_data_list/training_data_${place_object}_keywords.txt ./images/images_${place_object}_keywords
#ls ./images/images_${place_object}_keywords | wc -l

#############################

if [ -d ./images/images_${place_object}_keywords ]; then
    rm -r ./images/images_${place_object}_keywords
    mkdir ./images/images_${place_object}_keywords
else
    mkdir ./images/images_${place_object}_keywords
fi
while read line
do
    echo $line
    mp4_filename=`echo $line | cut -d '_' -f 1`
    echo $mp4_filename >> ./tmp_mp4_file
    cp ./images/images_${place_object}/$line.jpg ./images/images_${place_object}_keywords/$line.jpg
done < ./indexs/${place_object}_id_contain_keywords.txt
bash ./scripts/transform_number_to_path.sh ./tmp_mp4_file ./tmp_filepath
sort -u ./tmp_filepath > ./training_and_testing_data_list/training_data_${place_object}_keywords.txt
rm ./tmp_mp4_file
rm ./tmp_filepath
echo `ls ./images/images_${place_object}_keywords | wc -l` "images"
echo `wc -l ./training_and_testing_data_list/training_data_${place_object}_keywords.txt | cut -d ' ' -f 1` "waves"
