import os
import random
import numpy as np
from Utils import image_processing

possible_image_numbers = ["00000003", "00000008", "00000013", "00000018"]

class SoundnetMixOnehotDataloader:
    def __init__(self):
        self.training_data_list = ["/mnt/SoundNet_dataset/training_and_testing_data_list/training_data_plane_keywords.txt", "/mnt/SoundNet_dataset/training_and_testing_data_list/training_data_speedboat_keywords.txt", "/mnt/SoundNet_dataset/training_and_testing_data_list/training_data_guitar_keywords.txt", "/mnt/SoundNet_dataset/training_and_testing_data_list/training_data_piano_keywords.txt", "/mnt/SoundNet_dataset/training_and_testing_data_list/training_data_drum_keywords.txt", "/mnt/SoundNet_dataset/training_and_testing_data_list/training_data_dog_keywords.txt"]
        self.image_path_list = ["/mnt/SoundNet_dataset/images/images_plane_keywords/", "/mnt/SoundNet_dataset/images/images_speedboat_keywords/", "/mnt/SoundNet_dataset/images/images_guitar_keywords/", "/mnt/SoundNet_dataset/images/images_piano_keywords/", "/mnt/SoundNet_dataset/images/images_drum_keywords/", "/mnt/SoundNet_dataset/images/images_dog_keywords/"]
        self.feat_path_list = ["/mnt/SoundNet_dataset/mp3_soundnet_feat/mp3_soundnet_feat_plane_keywords_18/", "/mnt/SoundNet_dataset/mp3_soundnet_feat/mp3_soundnet_feat_speedboat_keywords_18/", "/mnt/SoundNet_dataset/mp3_soundnet_feat/mp3_soundnet_feat_guitar_keywords_18/", "/mnt/SoundNet_dataset/mp3_soundnet_feat/mp3_soundnet_feat_piano_keywords_18/", "/mnt/SoundNet_dataset/mp3_soundnet_feat/mp3_soundnet_feat_drum_keywords_18/", "/mnt/SoundNet_dataset/mp3_soundnet_feat/mp3_soundnet_feat_dog_keywords_18/"]

    def load_training_data(self):
        training_image_list = []
        class_label = []

        for i, training_data_filepath in enumerate(self.training_data_list):
            training_data = open(training_data_filepath, 'r')
            for line in training_data:
                filename = line.rstrip('\n').split('/')[-1].split('.')[0]
                for image_number in possible_image_numbers:
                    if os.path.isfile(self.image_path_list[i] + filename + "_" + image_number + ".jpg"):
                        training_image_list.append(self.image_path_list[i] + filename + "_" + image_number + ".jpg")
                        class_label.append(int(i))

        print("There are %d training data" % (len(training_image_list)))

        return {
            'image_list' : training_image_list,
            'data_length' : len(training_image_list),
            'class_label' : class_label
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
            onehot = np.zeros(len(self.training_data_list))
            onehot[loaded_data["class_label"][idx]] = 1
            acoustic_features.append(onehot)
            #z_noise.append(np.random.uniform(-1, 1, [z_dim]))
            z_noise.append(np.random.normal(scale=0.01, size=[z_dim]))
            image_files.append(filename)

        return np.array(real_images), np.array(wrong_images), np.array(acoustic_features), np.array(z_noise), image_files

    def shuffle(self, loaded_data):
        image_list = loaded_data["image_list"]
        class_label = loaded_data["class_label"]
        combined = list(zip(image_list, class_label))
        random.shuffle(combined)
        image_list, class_label = zip(*combined)

        return {
            'image_list' : image_list,
            'data_length' : len(image_list),
            'class_label' : class_label
        }
