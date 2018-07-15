#!/bin/bash

# Check arguments
if [ -z "$1" ]; then
    echo "There need to be at least one argument"
    exit
fi

store_path=./temp_mp3_files/
if [ ! -d $store_path ]; then
    mkdir $store_path
fi
rm $store_path/*

echo "Start moving files..."
while read -r line
do
    #echo "$line"
    mv $line $store_path
done < $1

rm -r mp3
