
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import os
import shutil
import utils

class ResidualNet_DM(object):
    def __init__(self, image_size, layers_num=32, features_size=64, input_channels=3, output_channels=3, scope='DMCNN', verbose=False):
        print('Building Residual Net Demosaicing...')
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
            self.is_training = isTrain = tf.placeholder_with_default(False, (), name='is_training')
            
            # preprocessing 
            mean_x = 127
            image_input = x - mean_x
            mean_y = 127
            image_target = y - mean_y

            with slim.arg_scope([slim.conv2d], padding='SAME',
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.001),
                                #weights_initializer=tf.contrib.layers.xavier_initializer(),                        
                                activation_fn=None):

                x = slim.conv2d(image_input, features_size, [3,3])
                x = slim.batch_norm(x, is_training=isTrain)
                x = tf.nn.selu(x)
                conv_1 = x

                for i in range(layers_num):
                    x = utils.residual_block_DMCNN(x, features_size, [3, 3], isTrain, scope='residual_block_DM_%d'%i)
                
                x = slim.conv2d(x,features_size,[3,3])
                x += conv_1
                x = slim.conv2d(x,output_channels,[3,3])

                output = image_input + x
                self.out = tf.clip_by_value(output+mean_x,0.0,255.0, name='op_out') 
                self.loss = tf.reduce_mean(tf.squared_difference(image_target, output))

            print('Done building!')

    def save(self, save_dir):
        self.saver.save(self.sess, save_dir + '\model')
        print('\n> Saved the checkpoint.(path:{})'.format(save_dir))

    def resume(self, save_dir):
        self.saver.restore(self.sess,tf.train.latest_checkpoint(save_dir))
        print('> Restored!')
    