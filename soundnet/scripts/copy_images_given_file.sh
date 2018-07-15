#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to give filename and output_dir"
    exit
fi

if [ -z "$2" ]; then
    echo "Need to give filename and output_dir"
    exit
fi

list_name=$1
output_dir=$2

if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

while read -r line;
do
    filename=`echo "$line" | rev | cut -d '/' -f 1 | rev | cut -d '.' -f 1`
    str_needed_to_be_replaces_from_start="mp3/"
    str_needed_to_be_replaces_from_end=".mp3"
    path=${line//$str_needed_to_be_replaces_from_start/""}
    path=${path//$str_needed_to_be_replaces_from_end/""}
    echo "$path"
    cp ./frames/$path/00000003.jpg $output_dir/${filename}_00000003.jpg
    cp ./frames/$path/00000008.jpg $output_dir/${filename}_00000008.jpg
    cp ./frames/$path/00000013.jpg $output_dir/${filename}_00000013.jpg
    cp ./frames/$path/00000018.jpg $output_dir/${filename}_00000018.jpg
done < $1
