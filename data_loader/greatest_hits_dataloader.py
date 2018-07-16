import os
import random
import numpy as np
from Utils import image_processing

class GreatestHitsDataloader:

    def __init__(self):
        self.training_data = open("/mnt/Greatest_hits/training_and_testing_data/train.txt", 'r')
        self.image_path = "/mnt/Greatest_hits/images_size64/"
        self.feat_path = "/mnt/Greatest_hits/soundnet_feat/"

    def load_training_data(self):
        training_image_list = []
        for line in self.training_data:
            filename = line.rstrip('\n')
            if os.path.isfile(self.image_path + filename + "_denoised.jpg"):
                training_image_list.append(self.image_path + filename + "_denoised.jpg")

        print("There are %d training data" % (len(training_image_list)))

        return {
            'image_list' : training_image_list,
            'data_length' : len(training_image_list)
        }

    def get_training_batch(self, batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, data_set, loaded_data = None):
        real_images = []
        wrong_images = []
        acoustic_features = []
        z_noise = []
        image_files = []
        for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
            idx = i % len(loaded_data["image_list"])
            real_images.append(image_processing.load_image_array(loaded_data["image_list"][idx], image_size))
            wrong_images.append(image_processing.load_image_array(loaded_data["image_list"][random.randint(0, len(loaded_data["image_list"])-1)], image_size))
            filename = loaded_data["image_list"][idx].split('/')[-1].split('.')[0]
            acoustic_features.append(np.load(self.feat_path + filename + ".npy"))
            z_noise.append(np.random.uniform(-1, 1, [z_dim]))
            image_files.append(filename)

        return np.array(real_images), np.array(wrong_images), np.array(acoustic_features), np.array(z_noise), image_files

    def shuffle(self, loaded_data):
        image_list = loaded_data["image_list"]
        random.shuffle(image_list)

        return {
            'image_list' : image_list,
            'data_length' : len(image_list)
        }
