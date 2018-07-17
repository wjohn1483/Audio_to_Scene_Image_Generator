#!/bin/bash
# Using tensorflow 1.7

if [ -z "$1" ]; then
    echo "usage: bash $0 [train or test]"
    exit
fi

# Configurations
filelist_dir=./soundnet/training_and_testing_data_list
feat_dir=./soundnet/mp3_soundnet_feat
images_generated_path=./Data/val_samples
dataset="soundnet_mix_multi_auxiliary_classifier"
testing_epoch=300
testing_type="testing"
caption_vector_length=256
epochs=301
batch_size=32
gf_dim=200
df_dim=200
t_dim=128
z_dim=10
num_class=9
dir=Data/Models

# Train
if [ "$1" = "train" ]; then
    if [ ! -d "$dir" ]; then
        mkdir -p $dir
    fi
    # For testing
    cp $0 $dir/$0
    (time CUDA_VISIBLE_DEVICES=0 python ./train.py --data_set=$dataset --caption_vector_length=$caption_vector_length --epochs=$epochs --batch_size=$batch_size --gf_dim=$gf_dim --df_dim=$df_dim --t_dim=$t_dim --z_dim=$z_dim --num_class=$num_class --save_dir=$dir/) 2>&1 | tee $dir/log
# Test
elif [ "$1" = "test" ]; then
    if [ ! -d "$images_generated_path" ]; then
        mkdir -p $images_generated_path
    fi
    rm -r $images_generated_path/*
    class_list=("plane" "speedboat" "piano" "dam" "baseball" "soccer" "guitar" "drum" "dog")
    for var in ${class_list[*]};
    do
        echo "Generating $var images..."
        mkdir $images_generated_path/$var
        python ./create_testdata_for_soundnet.py --list_path $filelist_dir/${testing_type}_data_${var}_keywords.txt --feat_path $feat_dir/mp3_soundnet_feat_${var}_keywords_18/
        CUDA_VISIBLE_DEVICES=0 python ./generate_images.py --n_images=1 --gf_dim=$gf_dim --df_dim=$df_dim --t_dim=$t_dim --z_dim=$z_dim --num_class=$num_class --caption_vector_length=$caption_vector_length --model_path=$dir/model_after_${dataset}_epoch_${testing_epoch}.ckpt --caption_thought_vectors=./Data/soundnet_testing_vectors.hdf5
        mv $images_generated_path/*.jpg $images_generated_path/$var
    done
else
    echo 'The argument must be "train" or "test"'
fi
