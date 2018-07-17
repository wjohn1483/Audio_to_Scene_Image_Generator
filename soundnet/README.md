# Soundnet_processing_scripts
Some scripts for preprocessing Soundnet dataset for sound to image.

The original repo of directory **soundnet_tensorflow** is
[SoundNet-tensorflow](https://github.com/eborboihuc/SoundNet-tensorflow).

# To extract images and soundnet feats of specific place or object classified by both SoundNet and Inception
```
# Remember to change the keywords you want to extract in ./scripts/classify_image.py
bash ./scripts/extract_images_and_classify.sh [place or object name]

bash ./scripts/extract_files_and_feats.sh [place or object name]

# Remember to change the directory path in ./scripts/image_processing.py
python ./scripts/image_processing.py

e.g.
bash ./scripts/extract_images_and_classify.sh birdhouse
bash ./scripts/extract_files_and_feats.sh birdhouse_keywords
python ./scripts/image_processing.py
```
The belows are some scripts contained in the scripts above.

# To extract images and soundnet feats of specific place or object classified by Inception only

```
bash ./scripts/copy_all_images_classified_by_inception.sh [place or object name]
bash ./scripts/extract_files_and_feats_classified_by_inception.sh [place or ]bject name]

e.g.
bash ./scripts/copy_all_images_classified_by_inception.sh birdhouse
bash ./scripts/extract_files_and_feats_classified_by_inception.sh birdhouse
```

## To copy specific place or object's pictures
```
bash ./scripts/get_images.sh [place or object name]
```
The pictures will be placed in *./images/images_[place of object name]/*

## To use image classification model to classify images
```
# This code need to be run under Tensorflow r1.2
bash ./scripts/classify_image.sh [place or object name]
```
Remember to modify the variable **keywords** (which is around line 298) in *classify_image.py* to the
keywords you want. The original file is from repo
[Tensorflow](https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py).

Please be attention to the path in the script.

## To copy the images after we get the list classified by image classification model
```
bash ./scripts/copy_images_classified_by_inception.sh [place or object name]
```
Be careful that the script will move the list from *./*, so you need to put the
list classified by image classification model to *./*.

## To resize the image size
```
python ./scripts/image_processing.py
```
Need to change the path in *image_processing.py* by yourself.

## To extract soundnet feat of specific place or object
```
# This code need to be run under Tensorflow r1.3
bash ./scripts/extract_files_and_feats.sh [place or object name]
```
Feats will be stored in *./mp3_soundnet_feat/mp3_soundnet_feat_[place or object
name]*.
