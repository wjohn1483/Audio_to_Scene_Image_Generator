import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

if len(sys.argv) != 3:
    print("Usage: python draw_tsne_for_specific_class.py [class name] [output path]")
    exit()

class_name = sys.argv[1]
output_path = sys.argv[2]
training_and_testing_list_dir = "./training_and_testing_data_list/"
feat_dir = "./mp3_soundnet_feat/"

def main():
    feats = []
    feat_class = []

    print("Reading training data...")
    training_data_list = open(training_and_testing_list_dir + "training_data_" + class_name + ".txt", 'r')
    for line in tqdm(training_data_list, ncols=80):
        filename = line.split('/')[-1].split('.')[0] + ".npy"
        feats.append(np.load(feat_dir + "mp3_soundnet_feat_" + class_name + '_18/' + filename))
        feat_class.append(int(0))

    print("Reading testing data...")
    testing_data_list = open(training_and_testing_list_dir + "testing_data_" + class_name + ".txt", 'r')
    for line in tqdm(testing_data_list, ncols=80):
        filename = line.split('/')[-1].split('.')[0] + ".npy"
        feats.append(np.load(feat_dir + "mp3_soundnet_feat_" + class_name + '_18/' + filename))
        feat_class.append(int(1))

    print("TSNE...")
    tsneModel = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    code = tsneModel.fit_transform(feats)

    print("Plotting...")
    plt.figure(figsize=(6, 6))
    plt.scatter(code[:, 0], code[:, 1], c=feat_class, cmap=plt.cm.get_cmap('viridis', 2))
    cbar = plt.colorbar(ticks=range(0, 2))
    plt.clim(-0.5, 2-0.5)
    cbar.ax.set_yticklabels(["train", "test"])
    plt.savefig(output_path)

if __name__ == "__main__":
    main()

