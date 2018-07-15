#/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify place or object name"
    exit
fi

place_or_object=$1
# Remember to change the keywords
CUDA_VISIBLE_DEVICES=0 python ./scripts/classify_image.py --image_dir ./images/images_${place_or_object}/ --output_filepath ./indexs/${place_or_object}_id_contain_keywords.txt
wc -l ./indexs/${place_or_object}_id_contain_keywords.txt
