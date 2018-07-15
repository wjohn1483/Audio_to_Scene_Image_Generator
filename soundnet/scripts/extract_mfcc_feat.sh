#!/bin/bash

if [ -z "$1" ]; then
    echo "usage: bash $0 [mp3 file dir] [output feat dir]"
    exit
fi

if [ -z "$2" ]; then
    echo "usage: bash $0 [mp3 file dir] [output feat dir]"
    exit
fi

data_dir=$1
feat_dir=$2

if [ ! -d $feat_dir ]; then
    mkdir -p $feat_dir
fi

# Conver mp3 to wav
#for filepath in $data_dir/*
#do
#    filename=`echo $filepath | rev | cut -d '/' -f 1 | rev | cut -d '.' -f 1`
#    echo "Processing $filename..."
#    # Convert .mp3 to .wav
#    ffmpeg -y -i $data_dir/$filename.mp4.mp3 $feat_dir/$filename.wav
#done

# Extract mfcc and apply cmvn
if [ -f $feat_dir/all.scp ]; then
    rm $feat_dir/all.scp
fi
for filepath in $feat_dir/*
do
    filename=`echo $filepath | rev | cut -d '/' -f 1 | rev | cut -d '.' -f 1`
    echo "$filename $filepath" >> $feat_dir/all.scp
done
compute-mfcc-feats --allow-downsample --channel=0 scp:$feat_dir/all.scp ark,t:$feat_dir/all.13.mfcc.ark
add-deltas --delta-order=2 ark,t:$feat_dir/all.13.mfcc.ark ark,t:$feat_dir/all.39.mfcc.ark
compute-cmvn-stats ark,t:$feat_dir/all.39.mfcc.ark ark,t:$feat_dir/all.39.cmvn.comput_result
apply-cmvn ark,t:$feat_dir/all.39.cmvn.comput_result ark,t:$feat_dir/all.39.mfcc.ark ark,t:$feat_dir/all.39.cmvn.ark

#python ./scripts/transform_ark_to_npy_mean.py $feat_dir/all.39.cmvn.ark $feat_dir

# Remove redundant files
#rm $feat_dir/*.wav
#rm $feat_dir/all.13.mfcc.ark
#rm $feat_dir/all.39.mfcc.ark
#rm $feat_dir/all.39.cmvn.comput_result
#rm $feat_dir/all.39.cmvn.ark
#rm $feat_dir/all.scp

