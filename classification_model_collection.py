import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from Inception_resnet import inception_resnet_v2, inception_resnet_v2_arg_scope
from resnet import resnet_v2_50
from resnet_utils import resnet_arg_scope
from mobilenet_v2 import mobilenet


class ClassificationModelCollection(object):
    def __init__(self, model, image_size, image_num, class_num, batch_size, root_pretrain, init_learning_rate, learning_rate_decay_factor):
        print('Building classification model...')
        self.image_size = image_size

        logits, end_points = self._select_model(model, class_num, root_pretrain)

        # restore 할 변수
        variables_to_restore = slim.get_variables_to_restore(exclude = self.exclude)

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = learning_rate = tf.train.exponential_decay(learning_rate = init_learning_rate,
                                                                        global_step = self.global_step,
                                                                        decay_steps = 1000,      # hcw decay_steps = decay_steps,
                                                                        decay_rate = learning_rate_decay_factor,
                                                                        staircase = False)

        self.predictions = end_points['Predictions']
        
        #one_hot_labels = slim.one_hot_encoding(self.labels, class_num) # hcw, cutmix
        one_hot_labels = self.labels # hcw, cutmix
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        #self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits, label_smoothing=0.1) #hcw
        self.total_loss = total_loss = tf.losses.get_total_loss()

        self.accuracy = tf.metrics.accuracy(tf.argmax(one_hot_labels,1), tf.argmax(self.predictions,1))


        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = slim.learning.create_train_op(total_loss, optimizer, global_step = self.global_step)
        
        self.predicted_class = predicted_class = tf.argmax(self.predictions, 1)

        
        #self.my_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, predicted_class), dtype=tf.float32)) # hcw, cutmix

        self.saver_restore = tf.train.Saver(variables_to_restore) # hcw
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000) # hcw

    def _select_model(self, model, class_num, root_pretrain):
        self.input = tf.placeholder(tf.float32, [None, self.image_size[1], self.image_size[0], 3], name='image_input')
        #self.labels = tf.placeholder(tf.int64, [None]) # hcw cutmix
        self.labels = tf.placeholder(tf.float32, [None, None])  # hcw cutmix
        self.is_training = tf.placeholder(tf.bool, name='is_training')  #hcw
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')   #hcw

        if model == 'inception_resnet_v2':
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(self.input, class_num, is_training = self.is_training, dropout_keep_prob = self.keep_prob) #hcw
                self.exclude = ['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits'] 
                self.last_layer_name = 'Predictions'
                self.path_pretrain = root_pretrain + 'inception_resnet_v2.ckpt'
        elif model == 'resnet_v2_50':
            with slim.arg_scope(resnet_arg_scope()):
                logits, end_points = resnet_v2_50(self.input, class_num, is_training = self.is_training)
                self.exclude = ['resnet_v2_50/logits']
                self.last_layer_name = 'predictions'
                self.path_pretrain = root_pretrain + 'resnet_v2_50.ckpt'
        elif model == 'mobilenet_v2':
            logits, end_points = mobilenet(self.input, class_num, is_training = self.is_training, depth_multiplier=0.5, finegrain_classification_mode=True)
            self.exclude = ['MobilenetV2/Logits']
            self.last_layer_name = 'Predictions'
            self.path_pretrain = root_pretrain + 'mobilenet_v2_0.5_128.ckpt'
            # Wrappers for mobilenet v2 with depth-multipliers. Be noticed that
            # 'finegrain_classification_mode' is set to True, which means the embedding
            # layer will not be shrinked when given a depth-multiplier < 1.0.
        else:
            raise ValueError('Error: the model is not available.')

        return logits, end_points

    def restore_fn(self, sess, checkpoint_file = None):
        if not checkpoint_file:
            checkpoint_file = self.path_pretrain
        return self.saver_restore.restore(sess, checkpoint_file)

    def save_checkpoint(self, sess, path, save_name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(sess, '{}\\{}'.format(path, save_name))

    def train_step(self, sess, images, labels):
        
        loss, predicted_class, learning_rate, acc = sess.run([self.train_op, self.predicted_class, self.learning_rate, self.accuracy], feed_dict={self.input:images, self.labels:labels, self.keep_prob:0.5, self.is_training:True})    #hcw

        return loss, predicted_class, learning_rate, acc

    def validation_step(self, sess, images, labels):
        loss, predicted_class, accuracy = sess.run([self.total_loss, self.predicted_class, self.accuracy], feed_dict={self.input:images, self.labels:labels, self.keep_prob:1.0, self.is_training:False})  #hcw
        
        return loss, predicted_class, accuracy # hcw, cutmix

    def save_image(self, utils, save_path, input_image, label_image, output_image, class_names, predicted_class, save_name):
        # input
        input_merge_images = utils.mergeimage(input_image)
        utils.saveimage(input_merge_images, save_path, '{}_Input'.format(save_name))
        # result
        utils.imagePredictLabel(input_image, label_image, class_names, predicted_class)
        result_images = utils.mergeimage(input_image)
        utils.saveimage(result_images, save_path, '{}_Result'.format(save_name))