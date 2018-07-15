import numpy as np
import traceback
import os
from tqdm import tqdm

feature_dir = "./mp3_feats/"
VECTOR_LENGTH = 39
NUMBER_OF_FRAME = 20

def read_ark(file_path):
    mfcc_features = []
    f = open(file_path)
    f.readline()
    for line in f:
        line = line.split()
        temp = np.zeros(VECTOR_LENGTH, dtype=float)
        for i in range(VECTOR_LENGTH):
            temp[i] = float(line[i])
        mfcc_features.append(temp)

    return np.vstack(mfcc_features)

if __name__ == "__main__":
    count = 0
    file_list = os.listdir(feature_dir)
    for f in tqdm(file_list, ncols=60):
        if f.endswith(".cmvn.ark") and (not os.path.isfile(feature_dir + f.split('.')[0] + ".mfcc.npy")):
            #print(f)
            try:
                mfcc_features = read_ark(feature_dir + f)
                temp = []
                # Sample MFCC vector to desired length
                for i in range(0, len(mfcc_features), int(len(mfcc_features) / NUMBER_OF_FRAME)):
                    if len(temp) != NUMBER_OF_FRAME:
                        temp.append(mfcc_features[i])
                    else:
                        break
                np.save(feature_dir + f.split('.')[0] + ".cmvn.npy", np.vstack(temp))
                count += 1
            except:
                print("Some errors occurred while processing : " + f)
                traceback.print_exc()
    print("Extracted " + str(count) + " files")
