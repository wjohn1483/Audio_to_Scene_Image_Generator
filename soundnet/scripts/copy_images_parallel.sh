#!/bin/bash

list_name=./training_and_testing_data_list/training_data_reef.txt
output_dir=./images/images_reef
temp_dir=./data_list_segments/
num_cpu=16

if [ ! -d $output_dir ]; then
    mkdir $output_dir
fi

if [ ! -d $temp_dir ]; then
    mkdir $temp_dir
fi

# Split list
split -n $num_cpu $list_name $temp_dir/segment

# Copy
for f in $temp_dir/segment*
do
    echo "$f"
    bash ./scripts/copy_images_given_file.sh $f $output_dir &
done

# Wait for all threads to finish
wait

# Remove temp dir
rm -r $temp_dir
