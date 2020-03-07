
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import os
import shutil
import utils

class ResidualNet_DN(object):
    def __init__(self, image_size, layers_num=32, features_size=64, input_channels=3, output_channels=3, scope='Denoise', verbose=False):
        print('Building Residual Net Denoise...')
        self.image_size = image_size
        self.output_channels = output_channels

        with tf.variable_scope(scope) as sc:
            #Placeholder for image inputs
            self.input = x = tf.placeholder(tf.float32, [None,None,None], name='image_input')
            x = tf.expand_dims(x, axis=3)
            #placeholde for image ground-truth
            self.target = y = tf.placeholder(tf.float32, [None,None,None], name='image_gt')
            y = tf.expand_dims(y, axis=3)

            # preprocessing 
            mean_x = 127
            image_input = x - mean_x
            mean_y = 127
            image_target = y - mean_y

            with slim.arg_scope([slim.conv2d], padding='SAME',
                                #weights_initialize=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=slim.l1_regularizer(0.0005),
                                activation_fn=None):

                x = slim.conv2d(image_input, features_size, [3, 3], scope='conv_start')
                x = tf.nn.relu(x)

                for i in range(layers_num):
                    x = slim.conv2d(x, features_size, [3, 3], scope='conv_%d'%(i+1))
                    x = tf.nn.relu(x)

                tf.add_to_collection(sc.name + '/conv_end', x)

                x = slim.conv2d(x, output_channels, [3, 3], scope='conv_end')
                        
            output = image_input + x
            self.out = tf.clip_by_value(output + mean_x, 0.0, 255.0, name='op_out')
            #self.loss = loss = tf.reduce_mean(tf.losses.absolute_difference(image_target, output))
            #L2 loss
            self.loss = loss = tf.losses.mean_squared_error(image_target, output)

            print('%s Done building!'% scope)

    def save(self, save_dir):
        self.saver.save(self.sess, save_dir + '\model')
        print('\n> Saved the checkpoint.(path:{})'.format(save_dir))

    def resume(self, save_dir):
        self.saver.restore(self.sess,tf.train.latest_checkpoint(save_dir))
        print('> Restored!')
    