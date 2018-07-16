import os
import random
import numpy as np
from Utils import image_processing

class FlowersMyselfDataloader:

    def __init__(self):
        self.image_path = "./soundnet/images_flower_size64/"

    def load_training_data(self):
        training_image_list = []
        for filename in os.listdir(self.image_path):
            training_image_list.append(self.image_path + filename)

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
            acoustic_features.append(np.zeros(caption_vector_length))
            z_noise.append(np.random.uniform(-1, 1, [z_dim]))
            image_files.append(0)

        return np.array(real_images), np.array(wrong_images), np.array(acoustic_features), np.array(z_noise), image_files

    def shuffle(self, loaded_data):
        image_list = loaded_data["image_list"]
        random.shuffle(image_list)

        return {
            'image_list' : image_list,
            'data_length' : len(image_list)
        }
