import os
import random
import numpy as np
from Utils import image_processing

class MsrvttDataloader:

    def __init__(self):
        self.image_path = '/mnt/ntu/text_to_video/data/resized_images_size64/'
        self.training_data = json.load(open("/mnt/ntu/text_to_video/data/sentences/train_val_data_simpler.json", 'r'))

    def load_training_data(self):
        training_image_list = []
        for data in self.training_data:
            training_image_list.append(self.image_path + data["id"] + "-001.jpeg")

        return {
            'image_list' : training_image_list,
            'data_length' : len(training_image_list)
        }

    def get_training_batch(self, batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, data_set, loaded_data = None):
        real_images = [] # (batch_size, image_size, image_size, channel)
        wrong_images = [] # (batch_size, image_size, image_size, channel)
        captions = [] # (batch_size, sentence_vector_size=2400)
        z_noise = [] # (batch_size, noise_vector_size=100)
        image_files = [] # (batch_size)
        for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
            idx = i % len(loaded_data["image_list"])
            real_images.append(image_processing.load_image_array(loaded_data["image_list"][idx], image_size))
            wrong_images.append(image_processing.load_image_array(loaded_data["image_list"][random.randint(0, len(loaded_data["image_list"])-1)], image_size))
            video_name = loaded_data["image_list"][idx].split('/')[-1].split('.')[0].split('-')[0]
            caption_path = "/mnt/ntu/text_to_video/data/sentences/vectors/"
            caption_npy = np.load(open(caption_path + video_name + ".npy", 'r'))
            captions.append(caption_npy[random.randint(0, caption_npy.shape[0]-1)])
            z_noise.append(np.random.uniform(-1, 1, [z_dim]))
            image_files.append(video_name)

        #return real_images, wrong_images, captions, z_noise, image_files
        return np.array(real_images), np.array(wrong_images), np.array(captions), np.array(z_noise), image_files

    def shuffle(self, loaded_data):
        image_list = loaded_data["image_list"]
        random.shuffle(image_list)

        return {
            'image_list' : image_list,
            'data_length' : len(image_list)
        }
