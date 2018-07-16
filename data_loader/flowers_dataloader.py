import os
import random
import numpy as np
from Utils import image_processing

class FlowersDataloader:

    def load_training_data(self, data_dir, data_set):
        h = h5py.File(join(data_dir, 'flower_tv.hdf5'))
        flower_captions = {}
        for ds in h.iteritems():
            flower_captions[ds[0]] = np.array(ds[1])
        image_list = [key for key in flower_captions]
        image_list.sort()

        img_75 = int(len(image_list)*0.75)
        training_image_list = image_list[0:img_75]
        random.shuffle(training_image_list)

        return {
            'image_list' : training_image_list,
            'captions' : flower_captions,
            'data_length' : len(training_image_list)
        }

    def get_training_batch(self, batch_no, batch_size, image_size, z_dim, caption_vector_length, split, data_dir, data_set, loaded_data = None):
        real_images = np.zeros((batch_size, 64, 64, 3))
        wrong_images = np.zeros((batch_size, 64, 64, 3))
        captions = np.zeros((batch_size, caption_vector_length))

        cnt = 0
        image_files = []
        for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
            idx = i % len(loaded_data['image_list'])
            image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][idx])
            image_array = image_processing.load_image_array(image_file, image_size)
            real_images[cnt,:,:,:] = image_array

            # Improve this selection of wrong image
            wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
            wrong_image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][wrong_image_id])
            wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
            wrong_images[cnt, :,:,:] = wrong_image_array

            random_caption = random.randint(0,4)
            captions[cnt,:] = loaded_data['captions'][ loaded_data['image_list'][idx] ][ random_caption ][0:caption_vector_length]
            image_files.append( image_file )
            cnt += 1

        z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
        return real_images, wrong_images, captions, z_noise, image_files

    def shuffle(self, loaded_data):
        image_list = loaded_data["image_list"]
        captions = loaded_data["captions"]
        combined = list(zip(image_list, captions))
        random.shuffle(combined)
        image_list, captions = zip(*combined)

        return {
            'image_list' : image_list,
            'captions' : captions,
            'data_length' : len(image_list)
        }
