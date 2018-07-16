import tensorflow as tf
from Utils import ops

class GAN:
    '''
    OPTIONS
    z_dim : Noise dimension 100
    t_dim : Text feature dimension 256
    image_size : Image Dimension 64
    gf_dim : Number of conv in the first layer generator 64
    df_dim : Number of conv in the first layer discriminator 64
    gfc_dim : Dimension of gen untis for for fully connected layer 1024
    caption_vector_length : Caption Vector Length 2400
    batch_size : Batch Size 64
    '''
    def __init__(self, options):
        self.options = options

        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')

        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')
        self.d_bn3 = ops.batch_norm(name='d_bn3')
        self.d_bn4 = ops.batch_norm(name='d_bn4')


    def build_model(self):
        img_size = self.options['image_size']
        t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
        t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
        t_class_label = tf.placeholder('float32', [self.options['batch_size'], self.options['num_class']], name = 'class_label')

        fake_image = self.generator(t_z, t_real_caption)

        disc_real_image, disc_real_image_logits, disc_real_class_logit   = self.discriminator(t_real_image, t_real_caption)
        disc_fake_image, disc_fake_image_logits, disc_fake_class_logit   = self.discriminator(fake_image, t_real_caption)

        # Generator loss and discriminator loss
        g_loss_image = -1 * tf.reduce_mean(disc_fake_image)
        g_loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t_class_label, logits=disc_fake_class_logit))
        g_loss = g_loss_image + g_loss_class
        d_loss_image = tf.reduce_mean(tf.maximum(tf.zeros_like(disc_real_image), 1 - disc_real_image)) + tf.reduce_mean(tf.maximum(tf.zeros_like(disc_fake_image), 1 + disc_fake_image))
        d_loss_class = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t_class_label, logits=disc_real_class_logit))
        d_loss = d_loss_image + d_loss_class
        d_loss1 = d_loss2 = d_loss3 = d_loss

        # For summary
        tf.summary.scalar("Generator loss : image", g_loss_image)
        tf.summary.scalar("Generator loss : classifier", g_loss_class)
        tf.summary.scalar("Generator loss : all", g_loss)
        tf.summary.scalar("Discriminator loss : image", d_loss_image)
        tf.summary.scalar("Discriminator loss : classifier", d_loss_class)
        tf.summary.scalar("Discriminator loss : all", d_loss)
        tf.summary.image("Real image", t_real_image)
        tf.summary.image("Fake image", fake_image)
        merged_summary = tf.summary.merge_all()

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        input_tensors = {
            't_real_image' : t_real_image,
            't_wrong_image' : t_wrong_image,
            't_real_caption' : t_real_caption,
            't_z' : t_z,
            't_class_label' : t_class_label
        }

        variables = {
            'd_vars' : d_vars,
            'g_vars' : g_vars
        }

        loss = {
            'g_loss' : g_loss,
            'd_loss' : d_loss,
            'summary' : merged_summary
        }

        outputs = {
            'generator' : fake_image
        }

        checks = {
            'd_loss1': d_loss1,
            'd_loss2': d_loss2,
            'd_loss3' : d_loss3,
            'disc_real_image_logits' : disc_real_image_logits,
            'disc_fake_image_logits' : disc_fake_image_logits
        }

        return input_tensors, variables, loss, outputs, checks

    def build_generator(self):
        img_size = self.options['image_size']
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
        fake_image = self.sampler(t_z, t_real_caption)

        input_tensors = {
            't_real_caption' : t_real_caption,
            't_z' : t_z
        }

        outputs = {
            'generator' : fake_image
        }

        return input_tensors, outputs

    # Sample Images for a text embedding
    def sampler(self, t_z, t_text_embedding):
        tf.get_variable_scope().reuse_variables()

        s = self.options['image_size']
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
        #z_concat = tf.concat(1, [t_z, reduced_text_embedding])
        z_concat = t_text_embedding
        z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train = False))

        h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train = False))

        h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train = False))

        h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train = False))

        h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

        return (tf.tanh(h4)/2. + 0.5)

    # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def generator(self, t_z, t_text_embedding):

        s = self.options['image_size']
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
        tf.summary.tensor_summary("Reduced voice embedding", reduced_text_embedding)
        #z_concat = tf.concat(1, [t_z, reduced_text_embedding])
        z_concat = t_text_embedding
        z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

        return (tf.tanh(h4)/2. + 0.5)

    # DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def discriminator(self, image, t_text_embedding):
        update_collection=tf.GraphKeys.UPDATE_OPS
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            h0 = ops.lrelu(ops.conv2d_sn(image, self.options['df_dim'], spectral_normed=True, update_collection=update_collection, name = 'd_h0_conv')) #32
            h1 = ops.lrelu( self.d_bn1(ops.conv2d_sn(h0, self.options['df_dim']*2, spectral_normed=True, update_collection=update_collection, name = 'd_h1_conv'))) #16
            h2 = ops.lrelu( self.d_bn2(ops.conv2d_sn(h1, self.options['df_dim']*4, spectral_normed=True, update_collection=update_collection, name = 'd_h2_conv'))) #8
            h3 = ops.lrelu( self.d_bn3(ops.conv2d_sn(h2, self.options['df_dim']*8, spectral_normed=True, update_collection=update_collection, name = 'd_h3_conv'))) #4
            h3_new = ops.lrelu( self.d_bn4(ops.conv2d_sn(h3, self.options['df_dim']*8, 1,1,1,1, spectral_normed=True, update_collection=update_collection, name = 'd_h3_conv_new'))) #4
            h3_new = tf.reshape(h3_new, [self.options['batch_size'], -1])
            image_embedding = ops.linear(h3_new, self.options['t_dim'], 'd_h3_embedding')

            # Auxiliary classifier
            class_logit = ops.linear(image_embedding, self.options['num_class'], 'd_image_classifier')

            # Embedding matrix of condition
            reduced_text_embeddings = ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding')

            # Scalar output function
            h4 = ops.linear(image_embedding, 1, 'd_scalar_output')

            discriminator_output_logit = tf.reduce_sum(tf.multiply(reduced_text_embeddings, image_embedding), 1, keepdims=True) + h4

        return discriminator_output_logit, discriminator_output_logit, class_logit
