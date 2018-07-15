#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify place or object"
    exit
fi

place_or_object=$1
image_dir=./images/images_${place_or_object}_inception_keywords
#image_dir=./images/images_all
output_index_file=./indexs/${place_or_object}_id_contain_inception_keywords.txt
output_training_filelist=./training_and_testing_data_list/training_data_${place_or_object}_inception_keywords.txt

if [ -f $output_index_file ]; then
    rm $output_index_file
fi

for path in `ls $image_dir`
do
    echo "$path"
    echo $path | cut -d '_' -f 1 >> $output_index_file
done

echo "Transforming index to filepath..."
bash ./scripts/transform_number_to_path.sh $output_index_file $output_training_filelist
tar zxvf ./mp3_public.tar.gz -T $output_training_filelist
bash ./scripts/move_files.sh ${place_or_object}_inception_keywords
rm -r mp3
cd ./soundnet_tensorflow/
(time bash extract_feat.sh ${place_or_object}_inception_keywords) 2>&1 | tee ./log
echo "Remember to split training set and testing set by yourself !!!"
