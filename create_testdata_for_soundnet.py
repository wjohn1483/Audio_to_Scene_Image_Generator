import numpy as np
import json
import h5py
import random
import os
import argparse
from tqdm import tqdm

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--list_path", type=str, help="The path to training/testing data filelist", required=True)
parser.add_argument("--feat_path", type=str, help="Path to feat", required=True)
args = parser.parse_args()

data_path = args.list_path
vectors_path = args.feat_path + "/"
vector_output_path = "./Data/soundnet_testing_vectors.hdf5"
filename_output_path = "./Data/soundnet_testing_filenames.txt"

def main():
    testing_data = open(data_path, 'r')
    acoustic_features = []
    filename_output_file = open(filename_output_path, 'w')
    for line in tqdm(testing_data, ncols=60):
        filename = line.rstrip('\n').split('/')[-1].split('.')[0]
        if os.path.isfile(vectors_path + filename + ".npy"):
            acoustic_features.append(np.load(vectors_path + filename + ".npy"))
            filename_output_file.write(filename + "\n")

    print("There are %d testing data" % (len(acoustic_features)))

    if os.path.isfile(vector_output_path):
        os.remove(vector_output_path)
    h = h5py.File(vector_output_path)
    h.create_dataset("vectors", data=acoustic_features)
    h.close()

if __name__ == "__main__":
    main()
