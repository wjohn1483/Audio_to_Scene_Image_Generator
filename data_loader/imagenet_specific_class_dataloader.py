import os
import pickle
import lasagne
import numpy as np
import random
from Utils import image_processing

class ImageNetSpecificClassDataloader:

    def __init__(self):
        self.image_dir = ["/mnt/Imagenet/images_warplane/", "/mnt/Imagenet/images_speedboat/", "/mnt/Imagenet/images_grand_piano/", "/mnt/Imagenet/images_drum/", "/mnt/Imagenet/images_baseball/", "/mnt/Imagenet/images_dam/"]

    def load_training_data(self):
        training_image_list = []
        for directory in self.image_dir:
            for filename in os.listdir(directory):
                training_image_list.append(directory + filename)

        return {
            'image_list' : training_image_list,
            'data_length' : len(training_image_list)
        }

    def get_training_batch(self, batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, data_set, loaded_data = None):
        real_images = []
        image_files = []
        for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
            idx = i % len(loaded_data["image_list"])
            real_images.append(image_processing.load_image_array(loaded_data["image_list"][idx], 64))
            filename = loaded_data["image_list"][idx]
            image_files.append(filename)

        return np.array(real_images), image_files

    def shuffle(self, loaded_data):
        image_list = loaded_data["image_list"]
        random.shuffle(image_list)

        return {
            'image_list' : image_list,
            'data_length' : len(image_list)
        }

