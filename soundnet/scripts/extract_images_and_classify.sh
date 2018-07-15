#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify place or object"
fi

place_or_object=$1

bash ./scripts/get_images.sh $place_or_object
bash ./scripts/classify_image_by_directory.sh $place_or_object
bash ./scripts/copy_images_classified_by_inception.sh $place_or_object
