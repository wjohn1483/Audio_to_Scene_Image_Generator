#!/bin/bash

for f in ./training_and_testing_data_list/training_data_soccer.txt ./training_and_testing_data_list/testing_data_soccer.txt
do
    echo "$f"
    tar zxvf mp3_public.tar.gz -T $f
done
