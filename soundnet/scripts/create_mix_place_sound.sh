#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "usage: $0 [place 1 name] [place 2 name] [number of samples from both places] [output dir]"
    echo "The number of total output files will be square of [number of samples from both places]"
    exit
fi

place1_dirpath=$1
place2_dirpath=$2
num_of_samples=$3
output_dir=$4

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

i=1
for file_place1 in `ls $place1_dirpath | head -n $num_of_samples`
do
    for file_place2 in `ls $place2_dirpath | head -n $num_of_samples`
    do
        echo "$file_place1, $file_place2, $i"
        i=$(($i + 1))
        filename_place1=`echo $file_place1 | cut -d '.' -f 1`
        filename_place2=`echo $file_place2 | cut -d '.' -f 1`
        sox -M $place1_dirpath/$file_place1 $place2_dirpath/$file_place2 $output_dir/${filename_place1}_${filename_place2}.mp3
    done
done
