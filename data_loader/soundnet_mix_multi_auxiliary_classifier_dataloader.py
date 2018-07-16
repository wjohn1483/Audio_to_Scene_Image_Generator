import os
import random
import numpy as np
from Utils import image_processing

possible_image_numbers = ["00000003", "00000008", "00000013", "00000018"]

class SoundnetMixMultiAuxiliaryClassifierDataloader:

    def __init__(self, add_noise=False):
        self.add_noise = add_noise
        self.training_data_list = ["./soundnet/training_and_testing_data_list/training_data_plane_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_speedboat_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_guitar_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_piano_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_drum_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_dog_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_dam_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_baseball_keywords.txt", "./soundnet/training_and_testing_data_list/training_data_soccer_keywords.txt"]
        self.image_path_list = ["./soundnet/images/images_plane_keywords/", "./soundnet/images/images_speedboat_keywords/", "./soundnet/images/images_guitar_keywords/", "./soundnet/images/images_piano_keywords/", "./soundnet/images/images_drum_keywords/", "./soundnet/images/images_dog_keywords/", "./soundnet/images/images_dam_keywords/", "./soundnet/images/images_baseball_keywords/", "./soundnet/images/images_soccer_keywords/"]
        self.feat_path_list = ["./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_plane_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_speedboat_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_guitar_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_piano_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_drum_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_dog_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_dam_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_baseball_keywords_18/", "./soundnet/mp3_soundnet_feat/mp3_soundnet_feat_soccer_keywords_18/"]
        self.num_class = len(self.training_data_list)

    def load_training_data(self):
        training_image_list = []
        number_of_training_data = 0
        for i, training_data_filepath in enumerate(self.training_data_list):
            training_data = open(training_data_filepath, 'r')
            training_image_single_place_or_object = []
            for line in training_data:
                filename = line.rstrip('\n').split('/')[-1].split('.')[0]
                if os.path.isfile(self.feat_path_list[i] + filename + ".npy"):
                    for image_number in possible_image_numbers:
                        if os.path.isfile(self.image_path_list[i] + filename + "_" + image_number + ".jpg"):
                            training_image_single_place_or_object.append(self.image_path_list[i] + filename + "_" + image_number + ".jpg")
                            number_of_training_data += 1
            training_image_list.append(training_image_single_place_or_object)

        print("There are %d training data" % (number_of_training_data))

        self.number_of_training_data = number_of_training_data

        return {
            'image_list' : training_image_list,
            'data_length' : number_of_training_data
        }

    def get_training_batch(self, batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, data_set, loaded_data = None):
        real_images = []
        wrong_images = []
        acoustic_features = []
        z_noise = []
        class_labels = []
        image_files = []
        for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
            #idx = i % len(loaded_data["image_list"])
            place_or_object_id = random.randint(0, len(loaded_data["image_list"])-1)
            image_index = random.randint(0, len(loaded_data["image_list"][place_or_object_id])-1)
            real_images.append(image_processing.load_image_array(loaded_data["image_list"][place_or_object_id][image_index], image_size))
            different_place_or_object_id = random.randint(0, len(loaded_data["image_list"])-1)
            while different_place_or_object_id == place_or_object_id:
                different_place_or_object_id = random.randint(0, len(loaded_data["image_list"])-1)
            wrong_images.append(image_processing.load_image_array(loaded_data["image_list"][different_place_or_object_id][random.randint(0, len(loaded_data["image_list"][different_place_or_object_id])-1)], image_size))
            filename = loaded_data["image_list"][place_or_object_id][image_index].split('/')[-1].split('.')[0].split('_')[0]
            if self.add_noise == False:
                acoustic_features.append(np.load(self.feat_path_list[place_or_object_id] + filename + ".npy"))
            elif self.add_noise == True:
                feat = np.load(self.feat_path_list[place_or_object_id] + filename + ".npy")
                acoustic_features.append(feat + np.random.normal(scale=0.01, size=[caption_vector_length]))
            #z_noise.append(np.random.uniform(-1, 1, [z_dim]))
            z_noise.append(np.random.normal(scale=0.01, size=[z_dim]))
            class_label = np.zeros(self.num_class)
            class_label[place_or_object_id] = 1
            class_labels.append(class_label)
            image_files.append(filename)

        return np.array(real_images), np.array(wrong_images), np.array(acoustic_features), np.array(z_noise), np.array(class_labels), image_files

    def shuffle(self, loaded_data):
        return loaded_data
