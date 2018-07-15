#!/bin/bash

#filename=`echo "$1" | cut -d '.' -f 1 | cut -d '/' -f 3`

while read -r line
do
    echo "$line"
    filename=`echo "$line" | cut -d '.' -f 1 | cut -d '/' -f 3`
    ffmpeg -y -i $line ${filename}.wav
    sox ${filename}.wav -n spectrogram -l -r -x 2000 -y 513 -o ${filename}.png
    #sox ${filename}.wav -n spectrogram -o ${filename}.png
done < ./test.txt
