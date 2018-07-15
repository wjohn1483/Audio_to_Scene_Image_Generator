#!/bin/bash

image_dir=./images_size64/
file=./training_data_list_more.txt
output_file=./training_data_exist.txt

if [ -f $output_file ]; then
    rm $output_file
fi

while read -r line
do
    echo "$line"
    filename=`echo "$line" | cut -d '/' -f 6 | cut -d '.' -f 1`
    if [ -f $image_dir/$filename.jpg ]; then
        echo "$filename" >> $output_file
    fi
done < $file

