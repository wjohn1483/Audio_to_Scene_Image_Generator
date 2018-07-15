import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

if len(sys.argv) <= 2:
    print("Usage: python draw_tsne.py [output path] [feat1 dir] [feat2 dir] [feat3 dir]...")
    exit()

def main():
    feats = []
    feat_class = []
    feat_name = []
    for i in range(0, len(sys.argv)-2):
        print("Reading feat %d..." % (i+1))
        filenames = os.listdir(sys.argv[i+2])
        feat_name.append(sys.argv[i+2].split('/')[1].rstrip('/').replace("mp3_soundnet_feat_", "").replace("_keywords", "").replace("_18", ""))
        for filename in tqdm(filenames, ncols=80):
            feats.append(np.load(sys.argv[i+2] + '/' + filename))
            feat_class.append(i+1)

    print("TSNE...")
    tsneModel = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    code = tsneModel.fit_transform(feats)

    print("Plotting...")
    number_of_color = len(sys.argv)-1
    plt.figure(figsize=(6, 6))
    plt.scatter(code[:, 0], code[:, 1], c=feat_class, cmap=plt.cm.get_cmap('viridis', number_of_color-1))
    cbar = plt.colorbar(ticks=range(1, number_of_color))
    plt.clim(0.5, number_of_color-0.5)
    cbar.ax.set_yticklabels(feat_name)
    plt.savefig(sys.argv[1])

if __name__ == "__main__":
    main()

