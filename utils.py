
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import freeze_graph

import matplotlib.pyplot as plt
import numpy as np
import os
import errno
import shutil
import cv2
import math

def subsample(x, factor, scope=None):
    if factor == 1:
        return x
    return slim.max_pool2d(x, [1, 1], factor, scope=scope)

def residual_block(x, bottleneck_depth, out_depth, stride=1, scope='residual_block'):
    in_depth=x.get_shape().as_list()[-1]
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], padding='SAME',
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005),
            activation_fn=None):

            if in_depth == out_depth:
                shortcut = subsample(x, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(x, out_depth, [1,1], stride=stride, scope='shortcut')

            residual = slim.conv2d(x, bottleneck_depth, [1,1], stride=stride, scope='conv1')
            residual = tf.nn.relu(residual)
            residual = slim.conv2d(residual, bottleneck_depth, 3, stride=stride, scope='conv2')
            residual = tf.nn.relu(residual)
            residual = slim.conv2d(residual, out_depth, [1,1], stride=1, scope='conv3')

            output = tf.nn.relu(shortcut + residual)

        return output

def residual_block_DMCNN(x, out_channels=64, kernel_size=[3,3], isTrain=False, scope='residual_block_DMCNN'):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], padding='SAME',
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=0.001),
            #weights_initializer=tf.contrib.layers.xavier_initializer(),
            activation_fn=None):

            residual = slim.conv2d(x, out_channels, kernel_size, scope='conv1')
            residual = slim.batch_norm(residual, is_training=isTrain)
            output = tf.nn.selu(residual)
            return x + output

"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def save_checkpoint(session, saver, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename + '.ckpt')
    tf.train.Saver.save(session, filepath)
    return filepath

def save_as_pb(session, directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    ckpt_filepath = os.path.join(directory, filename)

    # Save check point for graph frozen later
    pbtxt_filename = filename + '.pbtxt'
    pbtxt_filepath = os.path.join(directory, pbtxt_filename)
    pb_filepath = os.path.join(directory, filename + '.pb')
    # This will only save the graph but the variables will not be saved.
    # You have to freeze your model first.
    tf.train.write_graph(graph_or_graph_def=session.graph_def, 
                         logdir=directory, 
                         name=pbtxt_filename, 
                         as_text=True)

    # Freeze graph
    freeze_graph.freeze_graph(input_graph=pbtxt_filepath, 
                              input_saver='', 
                              input_binary=False, 
                              input_checkpoint=ckpt_filepath, 
                              #output_node_names='InceptionResnetV2/Logits/Predictions', ### Here!!
                              output_node_names='MobilenetV2/Predictions/Reshape_1', ### Here!!
                              restore_op_name='save/restore_all', 
                              filename_tensor_name='save/Const:0', 
                              output_graph=pb_filepath, 
                              clear_devices=True, 
                              initializer_nodes='')
        

PLOT_DIR = '../Weight/plots'

def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

def plot_conv_output(conv_img, org_img, result_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]), figsize=(10, 10), dpi=100)

    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        #_, h, w, c = conv_img.shape
        #w, h = get_ax_size(ax, fig)

        # put it on the grid
        #ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys', aspect='auto')
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', aspect='equal')

        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
    # save figure
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')
    cv2.imwrite(os.path.join(plot_dir, '{}_org.png'.format(name)), org_img)
    cv2.imwrite(os.path.join(plot_dir, '{}_result.png'.format(name)), result_img)

def get_ax_size(ax, fig):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

def get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def empty_dir(path):
    """
    Delete all files and folders in a directory
    :param path: string, path to directory
    :return: nothing
    """
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Warning: {}').format(e)


def create_dir(path):
    """
    Creates a directory
    :param path: string
    :return: nothing
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def prepare_dir(path, empty=False):
    """
    Creates a directory if it soes not exist
    :param path: string, path to desired directory
    :param empty: boolean, delete all directory content if it exists
    :return: nothing
    """
    if not os.path.exists(path):
        create_dir(path)

    if empty:
        empty_dir(path)


#----------------------------------------------------------------------------------------------
def mergeimage(images, blank=True):
    if images.ndim == 3:
        images = np.expand_dims(images, axis=3) 

    d, h, w, c = np.shape(images)
    size = round(math.sqrt(d)+0.5)
    img = np.zeros((h * size, (w * size) + int(blank), c))
        
    for idx, image in enumerate(images):
        i = idx % size
        j = idx // size
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img

def saveimage(image, save_path, file_name, file_extention='png'):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    path = save_path + '\\%s.%s' % (file_name, file_extention)
    cv2.imwrite(path, image)

def _imageText(image, label, point, color_text, color_bg):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_thickness = 1
    box_thickness = 1
    text_size, base = cv2.getTextSize(label, font_face, font_scale, text_thickness)
    offset = 3
    cv2.rectangle(image, (point[0]-1, point[1]+offset), (point[0]+text_size[0]+offset, point[1]-text_size[1]-offset), color_bg, cv2.FILLED)
    cv2.putText(image, label, point, font_face, font_scale, color_text)

def imagePredictLabel(images, labels, class_names, predicted_class):
    for idx, image in enumerate(images):
        p_class_idx = predicted_class[idx]
        gt_class_idx = labels[idx]
        gt_class_name = class_names[gt_class_idx]
        p_class_name = class_names[p_class_idx]
        _imageText(image, "Ground Truth : {}".format(gt_class_name), (10, 20), (255, 255, 255), (0, 0, 0))
        _imageText(image, "Predicted    : {}".format(p_class_name), (10, 40), (255, 255, 255), (0, 0, 0))

        if not gt_class_name == p_class_name:
            d, h, w, c = np.shape(images)
            cv2.rectangle(image, (1, 1), (w-3, h-3), (0,0,255), 3)

def psnr_and_ssim(output, target):
    """Compute the PSNR and SSIM.
    Args:
    output: 4-D Tensor, shape=(num_frames, height, width, num_channels)
    target: 4-D Tensor, shape=(num_frames, height, width, num_channels)
    Returns:
    psnr: 1-D Tensor, shape=(num_frames,)
    ssim: 1-D Tensor, shape=(num_frames,)
    """
    output = tf.cast(output, dtype=tf.int32)
    target = tf.cast(target, dtype=tf.int32)
    psnr = tf.image.psnr(output, target, max_val=255)
    ssim = tf.image.ssim(output, target, max_val=255)
    return psnr, ssim