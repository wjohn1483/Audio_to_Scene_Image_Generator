#!/bin/bash

cd ./soundnet/
bash make_dataset.sh
cd ..
bash ./run_soundnet.sh train
bash ./run_soundnet.sh test
python ./compute_inception_score.py Data/val_samples
