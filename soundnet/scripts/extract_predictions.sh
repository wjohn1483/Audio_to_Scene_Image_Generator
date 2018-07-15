#!/bin/bash

list=$1
temp_mp3_store_path=./temp_mp3_files/

# Check arguments
if [ -z "$1" ]; then
    echo "There need to be at least one argument"
    exit
fi

# Copy to docker
echo "Copying..."
docker cp temp_mp3_files/ epic_yalow:/home

# Extract predictions
echo "Start extracting..."
docker exec epic_yalow bash extract_prediction.sh $1

# Remove files
rm -r $temp_mp3_store_path
