import os
import pickle
import lasagne
import numpy as np
import random

class ImageNetDataloader:

    def __init__(self):
        self.data_folder = "./Data/"
        self.fileprefix = "imagenet_train_data_batch_"
        self.num_of_train_data_batch = 1


    # Note that this will work with Python3
    def unpickle(self, file):
        print("Unpickling data...")
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        print("Done")
        return dict

    def load_databatch(self, data_folder, fileprefix, idx, img_size=64):
        data_file = os.path.join(data_folder, fileprefix)

        d = self.unpickle(data_file + str(idx))
        x = d['data']
        y = d['labels']
        mean_image = d['mean']

        print("Processing images...")
        x = x/np.float32(255)
        mean_image = mean_image/np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i-1 for i in y]
        data_size = x.shape[0]

        x -= mean_image

        img_size2 = img_size * img_size

        print("After normalized images")

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        print("Creating mirrored images...")

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train, X_train_flip), axis=0)
        Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

        print("Done")

        #return dict(X_train=lasagne.utils.floatX(X_train),
        #            Y_train=Y_train.astype('int32'),
        #            mean=mean_image)
        return X_train, Y_train

    def load_training_data(self):
        x_train, y_train = self.load_databatch(self.data_folder, self.fileprefix, self.num_of_train_data_batch)

        return {
            'image_list' : x_train,
            'image_label' : y_train,
            'data_length' : len(x_train)
        }

    def get_training_batch(self, batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, data_set, loaded_data = None):
        real_images = []
        wrong_images = []
        acoustic_features = []
        z_noise = []
        image_files = []
        for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
            idx = i % len(loaded_data["image_list"])
            real_images.append(np.transpose(loaded_data["image_list"][idx], (1, 2, 0)))
            random_index = random.randint(0, len(loaded_data["image_list"])-1)
            while loaded_data["image_label"][idx] == loaded_data["image_label"][random_index]:
                random_index = random.randint(0, len(loaded_data["image_list"])-1)
            wrong_images.append(np.transpose(loaded_data["image_list"][random_index], (1, 2, 0)))
            condition = np.zeros((caption_vector_length))
            condition[loaded_data["image_label"][idx]] = 1
            acoustic_features.append(condition)
            filename = str(idx)
            z_noise.append(np.random.normal(scale=0.01, size=[z_dim]))
            image_files.append(filename)

        return np.array(real_images), np.array(wrong_images), np.array(acoustic_features), np.array(z_noise), image_files

    def shuffle(self, loaded_data):
        image_list = loaded_data["image_list"]
        image_label = loaded_data["image_label"]
        combined = list(zip(image_list, image_label))
        random.shuffle(combined)
        image_list, image_label = zip(*combined)

        return {
            'image_list' : image_list,
            'image_label' : image_label,
            'data_length' : len(image_list)
        }

