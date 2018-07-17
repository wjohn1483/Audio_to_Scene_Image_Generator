# Audio_to_Scene_Image_Generator

More information about extracting features can refer to
[soundnet](./soundnet).

## To run full experiment
```
bash run.sh
```
This script contains extracting features, training model, generating images,
and evaluation with Inception score.

## To extract features
```
cd ./soundnet/
bash make_dataset.sh
```
This script will extract files, images, and sound features for nine different
classes.

## To train model
```
bash run_soundnet.sh train
```
The default model is with hinge loss, spectral normalization, projection
discriminator, and auxiliary classifier.

If you want to use different model, modify import model in **train.py**.

In addition, while using different model, you have to use different data loader
to load data. All you need is to specify what model you are using in
**run_soundnet.sh**.

If you want to use different classes to train the model, you have to change the
path in corresponding data loader in **data_loader**.

## To test model
```
bash run_soundnet.sh test
```
All generated images will be placed in **Data/val_samples**.

Be careful that the model imported in **generate_images.py** must be the same
as the model you used in **train.py**.

## To evaluate
```
python compute_inception_score.py [path to dir]
e.g.
python compute_inception_score.py Data/val_samples**
```
This script will calculate all the images in subfolders in given argument.
