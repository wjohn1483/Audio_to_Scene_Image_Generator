#!/bin/bash

# Check arguments
if [ -z "$1" ]; then
    echo "Need to specify place or object"
    exit
fi

predictions_dir=./results/
place=$1

grep $place $predictions_dir* | cut -d ' ' -f 2 | cut -d '/' -f 5 | cut -d '.' -f 1
