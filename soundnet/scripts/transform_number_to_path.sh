#!/bin/bash

if [ -z "$1" ]; then
    echo "Need to specify id and output file"
    exit
fi

if [ -z "$2" ]; then
    echo "Need to specify id and output file"
    exit
fi

id_file=$1
output_file=$2
all_id_file=./mp3_public.list

if [ -f "$output_file" ]; then
    rm $output_file
fi

while read -r line
do
    echo "$line"
    echo `grep "$line" $all_id_file` >> $output_file
done < $id_file
