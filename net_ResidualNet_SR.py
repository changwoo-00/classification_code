
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import os
import shutil

class ResidualNet_SR(object):
    def __init__(self, image_size, layers_num=32, features_size=64, input_channels=3, output_channels=3, scale=2, scope='SR', verbose=False):
        print('Building Residual Net Super Resolution...')
        self.image_size = image_size
        self.output_channels = output_channels

        with tf.variable_scope(scope) as sc:
            #Placeholder for image inputs
            self.input = x = tf.placeholder(tf.float32,
                                            [None,None,None,input_channels],
                                            name='image_input')
            #placeholde for image ground-truth
            self.target = y = tf.placeholder(tf.float32,
                                            [None,None,None,output_channels],
                                            name='image_gt')

            #self.is_training = isTrain = tf.placeholder_with_default(False, (), name='is_training')

            # preprocessing 
            mean_x = 127
            image_input = x - mean_x
            mean_y = 127
            image_target = y - mean_y

            with slim.arg_scope([slim.conv2d], padding='SAME',
                                weights_regularizer=slim.l1_regularizer(0.0005),
                                activation_fn=None):

                #One convolution before res blocks and to convert to required feature depth
                x = slim.conv2d(image_input, features_size, [3,3])

                #Store the output of the first convolution to add later
                conv_1 = x

                scaling_factor = 0.1

                for i in range(layers_num):
                    x = self._residual_block_EDSR(x, features_size, [3, 3], scale=scaling_factor, scope='residual_block_EDSR_%d' % i)
                
                #One more convolution, and then we add the output of our first conv layer
                x = slim.conv2d(x,features_size,[3,3])
                x += conv_1

		        #Upsample output of the convolution		
                x = self._upsample(x,scale,features_size,None)

		        #One final convolution on the upsampling output
                output = x#slim.conv2d(x,output_channels,[3,3])
                self.out = tf.clip_by_value(output+mean_x,0.0,255.0, name='op_out') 
                self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target, output))

                ##Calculating Peak Signal-to-noise-ratio
		        ##Using equations from here: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
                #mse = tf.reduce_mean(tf.squared_difference(image_target,output))	
                #PSNR = tf.constant(255**2,dtype=tf.float32)/mse
                #PSNR = tf.constant(10,dtype=tf.float32)*utils.log10(PSNR)

        optimizer = tf.train.AdamOptimizer()
		#This is the train operation for our objective
        self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(tf.global_variables())

        print('Done building!')

    def _residual_block_EDSR(self, x, out_channels=64, kernel_size=[3,3], scale=1, scope='residual_block_EDSR'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=slim.l2_regularizer(0.0005),
                activation_fn=None):

                residual = slim.conv2d(x, out_channels, kernel_size, scope='conv1')
                residual = tf.nn.relu(residual)
                residual = slim.conv2d(residual, out_channels, kernel_size, scope='conv2')
                output = residual * scale
                return x + output

            
    def _upsample(self, x,scale=2,features=64,activation=tf.nn.relu):
        assert scale in [2,3,4]
        x = slim.conv2d(x,features,[3,3],activation_fn=activation)
        if scale == 2:
            ps_features = 3*(scale**2)
            x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		    #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
            x = self._PS(x,2,color=True)
        elif scale == 3:
            ps_features =3*(scale**2)
            x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		    #x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
            x = self._PS(x,3,color=True)
        elif scale == 4:
            ps_features = 3*(2**2)
            for i in range(2):
                x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
			    #x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
                x = self._PS(x,2,color=True)
        return x

    def _phase_shift(self, I, r):
        return tf.depth_to_space(I, r)

    def _PS(self, X, r, color=False):
        if color:
            Xc = tf.split(X, 3, 3)
            X = tf.concat([self._phase_shift(x, r) for x in Xc],3)
        else:
            X = self._phase_shift(X, r)
        return X

    def restore_fn(self, sess, checkpoint_file = None):
        if not checkpoint_file:
            checkpoint_file = self.path_pretrain
        return self.saver.restore(sess, checkpoint_file)

    def save_checkpoint(self, sess, path, save_name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(sess, '{}\\{}'.format(path, save_name))

    def train_step(self, sess, images, labels):
        loss, out_images, _ = sess.run([self.loss, self.out, self.train_op], feed_dict={self.input:images, self.target:labels})
        return loss, None, None, out_images
    
    def validation_step(self, sess, images, labels):
        out_images = sess.run([self.out], feed_dict={self.input:images, self.target:labels})
        return 0.0, None, out_images[0]

    def save_image(self, utils, save_path, input_image, label_image, output_image, class_names, predicted_class, save_name):
        # input
        input_merge_images = utils.mergeimage(input_image)
        utils.saveimage(input_merge_images, save_path, '{}_Input'.format(save_name))
        # result
        result_images = utils.mergeimage(output_image)
        utils.saveimage(result_images, save_path, '{}_Result'.format(save_name))