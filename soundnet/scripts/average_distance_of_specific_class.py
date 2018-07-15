import os
import sys
import numpy as np
from tqdm import tqdm

if len(sys.argv) != 2:
    print("Need to specify class name")
    exit()

feature_dir = sys.argv[1]
mean_vector = np.zeros(256)
mean = False

print("Loading data...")
features = []
for filename in tqdm(os.listdir(feature_dir), ncols=50):
    feat = np.load(feature_dir + "/" + filename)
    mean_vector += feat
    features.append(feat)


# Mean vector
if mean == True:
    mean_vector /= len(features)

    print("Calculating distances...")
    distance = 0
    for feat in tqdm(features, ncols=50):
        distance += np.linalg.norm(mean_vector-feat)

    print("Mean distance = {}".format(distance/len(features)))

# All distances
else:
    distance = 0
    count = 0
    for i in tqdm(range(0, len(features)), ncols=50):
        for j in range(i, len(features)):
            distance += np.linalg.norm(features[i]-features[j])
            count += 1

    print("Mean distance = {}".format(distance/count))

