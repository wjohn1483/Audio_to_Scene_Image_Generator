import tensorflow as tf
import numpy as np
import models.model_conditional_gan_with_hinge_loss_spectral_norm_projection_discriminator_auxiliary_classifier as model
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os
import shutil
import sys
from data_loader import flowers_dataloader
from data_loader import soundnet_dataloader
from data_loader import msrvtt_dataloader
from data_loader import soundnet_mix_onehot_dataloader
from data_loader import flowers_myself_dataloader
from data_loader import greatest_hits_dataloader
from data_loader import soundnet_mix_multi_dataloader
#from data_loader import imagenet_dataloader
from data_loader import soundnet_mix_multi_auxiliary_classifier_dataloader
from data_loader import soundnet_mix_multi_auxiliary_classifier_one_to_many_dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_dim', type=int, default=100,
                       help='Noise dimension')

    parser.add_argument('--t_dim', type=int, default=256,
                       help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')

    parser.add_argument('--image_size', type=int, default=64,
                       help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=64,
                       help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                       help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                       help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=2400,
                       help='Caption Vector Length')

    parser.add_argument('--num_class', type=int, default=0,
                       help='Number of classes')

    parser.add_argument('--data_dir', type=str, default="Data",
                       help='Data Directory')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                       help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=600,
                       help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=30,
                       help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--resume_model', type=str, default=None,
                       help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--data_set', type=str, default="flowers",
                       help='Dat set: MS-COCO, flowers')

    parser.add_argument('--save_dir', type=str, required=True,
                       help='Save dir')

    args = parser.parse_args()
    model_options = {
        'z_dim' : args.z_dim,
        't_dim' : args.t_dim,
        'batch_size' : args.batch_size,
        'image_size' : args.image_size,
        'gf_dim' : args.gf_dim,
        'df_dim' : args.df_dim,
        'gfc_dim' : args.gfc_dim,
        'caption_vector_length' : args.caption_vector_length,
        'num_class' : args.num_class
    }


    gan = model.GAN(model_options)
    input_tensors, variables, loss, outputs, checks = gan.build_model()

    d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1 = args.beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=5)
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    data_loader, loaded_data = load_training_data(args.data_dir, args.data_set)

    writer = tf.summary.FileWriter(args.save_dir + "/logs/", sess.graph)

    for i in range(args.epochs):
        batch_no = 0
        while batch_no*args.batch_size < loaded_data['data_length']:
            # DISCR UPDATE
            for _ in range(0, 1):
                feed_dict = get_feed_dict(data_loader, batch_no, args, input_tensors, loaded_data)
                check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
                _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts, feed_dict=feed_dict)

            print("d1", d1)
            print("d2", d2)
            print("d3", d3)
            print("D", d_loss)
            sys.stdout.flush()

            # GEN UPDATE TWICE or more, to make sure d_loss does not go to 0
            for _ in range(0, 5):
                feed_dict = get_feed_dict(data_loader, batch_no, args, input_tensors, loaded_data)
                _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']], feed_dict)


            print("LOSSES", d_loss, g_loss, batch_no, i, loaded_data['data_length']// args.batch_size)
            batch_no += 1

            # Output summary
            if (batch_no % args.save_every) == 0:
                print("Output to tensorboard")
                result = sess.run(loss['summary'], feed_dict=feed_dict)
                writer.add_summary(result, batch_no + i * loaded_data['data_length']// args.batch_size)

            #if (batch_no % args.save_every) == 0:
            #    print("Saving Images, Model")
            #    if args.data_set != 'flowers_myself':
            #        save_for_vis(args.data_dir, real_images, gen, image_files)
            #    save_path = saver.save(sess, args.save_dir + "/latest_model_{}_temp.ckpt".format(args.data_set))
        if i%50 == 0:
            save_path = saver.save(sess, args.save_dir + "/model_after_{}_epoch_{}.ckpt".format(args.data_set, i))

        loaded_data = data_loader.shuffle(loaded_data)

def load_training_data(data_dir, data_set):
    print("Loading dataset {}".format(data_set))

    if data_set == 'flowers':
        dataloader = flowers_dataloader.FlowersDataloader()
        return dataloader, dataloader.load_training_data(data_dir, data_set)

    elif data_set == 'mscoco':
        with open(join(data_dir, 'meta_train.pkl')) as f:
            meta_data = pickle.load(f)
        # No preloading for MS-COCO
        return meta_data

    elif data_set == 'MSR-VTT':
        dataloader = msrvtt_dataloader.MsrvttDataloader()
        return dataloader, dataloader.load_training_data

    elif data_set == 'soundnet':
        dataloader = soundnet_dataloader.SoundnetDataloader(data_dir, data_set)
        return dataloader, dataloader.load_training_data()

    elif data_set == 'soundnet_mix_onehot':
        dataloader = soundnet_mix_onehot_dataloader.SoundnetMixOnehotDataloader()
        return dataloader, dataloader.load_training_data()

    elif data_set == 'flowers_myself':
        dataloader = flowers_myself_dataloader.FlowersMyselfDataloader()
        return dataloader, dataloader.load_training_data()

    elif data_set == 'greatest_hits':
        dataloader = greatest_hits_dataloader.GreatestHitsDataloader()
        return dataloader, dataloader.load_training_data()

    elif data_set == 'soundnet_mix_multi':
        dataloader = soundnet_mix_multi_dataloader.SoundnetMixMultiDataloader(add_noise=False)
        return dataloader, dataloader.load_training_data()

    elif data_set == 'soundnet_mix_multi_noise':
        dataloader = soundnet_mix_multi_dataloader.SoundnetMixMultiDataloader(add_noise=True)
        return dataloader, dataloader.load_training_data()

    elif data_set == 'imagenet':
        dataloader = imagenet_dataloader.ImageNetDataloader()
        return dataloader, dataloader.load_training_data()

    elif data_set == 'soundnet_mix_multi_auxiliary_classifier':
        dataloader = soundnet_mix_multi_auxiliary_classifier_dataloader.SoundnetMixMultiAuxiliaryClassifierDataloader(add_noise=False)
        return dataloader, dataloader.load_training_data()

    elif data_set == 'soundnet_mix_multi_auxiliary_classifier_noise':
        dataloader = soundnet_mix_multi_auxiliary_classifier_dataloader.SoundnetMixMultiAuxiliaryClassifierDataloader(add_noise=True)
        return dataloader, dataloader.load_training_data()

    elif data_set == 'soundnet_mix_multi_auxiliary_classifier_one_to_many':
        dataloader = soundnet_mix_multi_auxiliary_classifier_one_to_many_dataloader.SoundnetMixMultiAuxiliaryClassifierOneToManyDataloader(add_noise=False)
        return dataloader, dataloader.load_training_data()

    elif data_set == 'soundnet_mix_multi_auxiliary_classifier_one_to_many_noise':
        dataloader = soundnet_mix_multi_auxiliary_classifier_one_to_many_dataloader.SoundnetMixMultiAuxiliaryClassifierOneToManyDataloader(add_noise=True)
        return dataloader, dataloader.load_training_data()

def get_feed_dict(data_loader, batch_no, args, input_tensors, loaded_data=None):
    if "auxiliary_classifier" in args.data_set:
        real_images, wrong_images, caption_vectors, z_noise, class_label, image_files = data_loader.get_training_batch(batch_no, args.batch_size,
            args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)
        feed_dict = {
                input_tensors['t_real_image'] : real_images,
                input_tensors['t_wrong_image'] : wrong_images,
                input_tensors['t_real_caption'] : caption_vectors,
                input_tensors['t_z'] : z_noise,
                input_tensors['t_class_label'] : class_label
            }

    else:
        real_images, wrong_images, caption_vectors, z_noise, image_files = data_loader.get_training_batch(batch_no, args.batch_size,
            args.image_size, args.z_dim, args.caption_vector_length, 'train', args.data_dir, args.data_set, loaded_data)
        feed_dict = {
                input_tensors['t_real_image'] : real_images,
                input_tensors['t_wrong_image'] : wrong_images,
                input_tensors['t_real_caption'] : caption_vectors,
                input_tensors['t_z'] : z_noise
            }

    return feed_dict

def save_for_vis(data_dir, real_images, generated_images, image_files):

    shutil.rmtree( join(data_dir, 'samples') )
    os.makedirs( join(data_dir, 'samples') )

    for i in range(0, real_images.shape[0]):
        real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
        real_images_255 = (real_images[i,:,:,:])
        scipy.misc.imsave( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

        fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
        fake_images_255 = (generated_images[i,:,:,:])
        scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)

# Leave this deprecated function for dataset mscoco
def get_training_batch(batch_no, batch_size, image_size, z_dim,
    caption_vector_length, split, data_dir, data_set, loaded_data = None):
    if data_set == 'mscoco':
        with h5py.File( join(data_dir, 'tvs/'+split + '_tvs_' + str(batch_no))) as hf:
            caption_vectors = np.array(hf.get('tv'))
            caption_vectors = caption_vectors[:,0:caption_vector_length]
        with h5py.File( join(data_dir, 'tvs/'+split + '_tv_image_id_' + str(batch_no))) as hf:
            image_ids = np.array(hf.get('tv'))

        real_images = np.zeros((batch_size, 64, 64, 3))
        wrong_images = np.zeros((batch_size, 64, 64, 3))

        image_files = []
        for idx, image_id in enumerate(image_ids):
            image_file = join(data_dir, '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, image_id) )
            image_array = image_processing.load_image_array(image_file, image_size)
            real_images[idx,:,:,:] = image_array
            image_files.append(image_file)

        # TODO>> As of Now, wrong images are just shuffled real images.
        first_image = real_images[0,:,:,:]
        for i in range(0, batch_size):
            if i < batch_size - 1:
                wrong_images[i,:,:,:] = real_images[i+1,:,:,:]
            else:
                wrong_images[i,:,:,:] = first_image

        z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

        return real_images, wrong_images, caption_vectors, z_noise, image_files

if __name__ == '__main__':
    main()
