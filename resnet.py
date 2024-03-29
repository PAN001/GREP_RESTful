import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

from config import Config

import datetime
import numpy as np
import os
import time

MOVING_AVERAGE_DECAY =0# 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00001  #0.00001
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004  #0.00004
FC_WEIGHT_STDDEV = 0.01
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

tf.app.flags.DEFINE_integer('input_size', 32, "input image size")


activation = tf.nn.relu
feature=None

def inference(x, is_training,
              num_classes=1000,
              num_blocks=[2,2,2,2],  # defaults to 50-layer network
              use_bias=False, # defaults to using batch norm
              bottleneck=True):
    c = Config()
    c['bottleneck'] = bottleneck
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2

    with tf.variable_scope('1'):
        c['conv_filters_out'] = 64
        c['ksize'] = 7
        c['stride'] = 2
        x = conv(x, c,True)
        x = bn(x, c)
        x = activation(x)

    with tf.variable_scope('2'):
        x = _max_pool(x, ksize=3, stride=2) # max pooling
        c['num_blocks'] = num_blocks[0]
        c['stack_stride'] = 1
        c['block_filters_internal'] = 64
        x = stack(x, c)

    with tf.variable_scope('3'):
        c['num_blocks'] = num_blocks[1]
        c['block_filters_internal'] = 128
        assert c['stack_stride'] == 2
        x = stack(x, c)

    with tf.variable_scope('4'):
        c['num_blocks'] = num_blocks[2]
        c['block_filters_internal'] = 256
        x = stack(x, c)

    with tf.variable_scope('5'):
        c['num_blocks'] = num_blocks[3]
        c['block_filters_internal'] = 512
        x = stack(x, c)
    
    # post-net
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    before_x=x
    if num_classes != None:
        with tf.variable_scope('fc'):
            x = fc(x, c)

    return before_x,x


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x,
                    is_training,
                    num_blocks=3, # 6n+2 total weight layers will be used.
                    use_bias=False, # defaults to using batch norm
                    num_classes=10):
    c = Config()
    c['is_training'] =is_training #tf.convert_to_tensor(is_training,
                                           # dtype='bool',
                                           # name='is_training')
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['num_classes'] = num_classes
    return inference_small_config(x, c)

def inference_small_config(x, c):
    c['bottleneck'] = False
    c['ksize'] = 3
    c['stride'] = 1
    with tf.variable_scope('scale1'):
        c['conv_filters_out'] = 16
        c['block_filters_internal'] = 16
        c['stack_stride'] = 1
        x = conv(x, c)
        x = bn(x, c)
        x = activation(x)
        x = stack(x, c)
    with tf.variable_scope('scale2'):
        c['block_filters_internal'] = 32
        c['stack_stride'] = 2
        x = stack(x, c)

    with tf.variable_scope('scale3'):
        c['block_filters_internal'] = 64
        c['stack_stride'] = 2
        x = stack(x, c)

    
    x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")
    feature=x 
    if c['num_classes'] != None:
        with tf.variable_scope('fc'):
            x = fc(x, c)

    return feature,x


def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(3, 3, rgb * 255.0)
    bgr = tf.concat(3, [blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr


def loss(logits, labels,scope_name):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels= labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope=scope_name)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    
    tf.summary.scalar('loss', loss_)


    return loss_

def regression_loss(logits,labels,scope_name):
    l2loss=tf.pow(tf.subtract(logits,labels),2)
    l2loss_mean=tf.reduce_mean(l2loss)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope=scope_name)

    loss_ = tf.add_n([l2loss_mean] + regularization_losses)
    return l2loss_mean
def softmax_regression_loss(logits,labels,scope_name):
    prediction=tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,scope=scope_name)
    loss_ = tf.add_n([cross_entropy] + regularization_losses)
    return loss_

def stack(x, c):
    alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q'
             'r','s','t','u','v','w','x','y','z']
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1
        c['block_stride'] = s
        with tf.variable_scope(alphabet[n]):
            x = block(x, c)
    return x


def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']
    
    if c['bottleneck']:
        with tf.variable_scope('branch2a'):
            c['ksize'] = 2
            c['stride'] = c['block_stride']
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('branch2b'):
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('branch2c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 2
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            assert c['ksize'] == 3
            x = conv(x, c)
            x = bn(x, c)
            x = activation(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            assert c['ksize'] == 3
            assert c['stride'] == 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('branch1'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 2
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)
    
    return activation(x + shortcut)


def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    if c['use_bias']:
        bias = _get_variable('bias', params_shape,
                             initializer=tf.zeros_initializer())
        return x + bias

    
    axis = list(range(len(x_shape) - 1))

    with tf.variable_scope('bn'):
        beta = _get_variable('offset',#beta
                         params_shape,
                         initializer=tf.zeros_initializer())
        gamma = _get_variable('scale',#gamma
                          params_shape,
                          initializer=tf.ones_initializer())

        moving_mean = _get_variable('mean', #moving_mean
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
        moving_variance = _get_variable('variance',#moving_variance
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    global feature
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        c['is_training'], lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??
    return x


def fc(x, c):
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer =tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x


def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def conv(x, c,bias_term=False):
    ksize = c['ksize']
    stride = c['stride']
    
    filters_out = c['conv_filters_out']
    if stride==2:
        print ksize,stride,filters_out
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    with tf.variable_scope('res'):
        weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
        out=tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
        if bias_term:
            bias=_get_variable('biases',shape=[filters_out],dtype='float',initializer=tf.zeros_initializer())
            out=tf.nn.bias_add(out,bias)
   
    return out


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
