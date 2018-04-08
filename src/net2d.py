""" Common utility for CNN2D. 

Author: Chi Li
Date: June 26, 2017

Wrap up conv+BN+Relu with careful BN parameter setting.

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import pdb

from six.moves import urllib
import tensorflow as tf

# Constants describing the network

def var_summaries(var, name):
  #Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  with tf.name_scope(name + '_summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.histogram('histogram', var)

def _var_init(name, shape, init_method):
  # return float32 variable initialized with init_method
  return tf.get_variable(name, shape, initializer=init_method, dtype=tf.float32)

def deconv(name, input_, size, out_num, stride, is_training):
  vec = input_.get_shape().as_list()  
  n = vec[0]
  dim1 = vec[1]
  dim2 = vec[2]
  in_num = vec[3]
  xavier = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  kernel = _var_init(name, shape=[size, size, out_num, in_num], init_method=xavier)
  
  dconv = tf.nn.conv2d_transpose(input_, kernel, [n, dim1 * stride, dim2 * stride, out_num], 
      [1, stride, stride, 1], padding='SAME')
  return dconv

def modern_conv(name, input_, size, num, stride, is_training, dropout):
  # convolutional layer 
  in_num = input_.get_shape()[-1].value
  out_num = num

  xavier = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  constant = tf.constant_initializer(0.0)
  with tf.variable_scope(name):
    kernel = _var_init('weights', shape=[size, size, in_num, out_num], init_method=xavier)
    conv = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding='SAME')
    biases = _var_init('biases', [out_num], init_method=constant)
    pre_bn = tf.nn.bias_add(conv, biases)

    pre_relu = tf.contrib.layers.batch_norm(pre_bn, center=True, scale=True, 
        is_training=is_training, decay=0.99, zero_debias_moving_mean=False, 
        fused=True, scope='bn')     
    
    final_conv = tf.nn.relu(pre_relu, name='relu')
    if is_training is True:
      final_conv = tf.nn.dropout(final_conv, keep_prob=dropout, name='dropout')

    return final_conv

def GAP(name, input_):
  with tf.variable_scope(name):
    gap = tf.reduce_mean(input_, [1, 2])
    return gap

def GAP3D(name, input_):
  with tf.variable_scope(name):
    gap = tf.reduce_mean(input_, [1, 2, 3])
    return gap

def FC(name, input_, size):
  in_num = input_.get_shape()[-1].value
  out_num = size
  xavier = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  constant = tf.constant_initializer(0.0)
  with tf.variable_scope(name) as scope:
    kernel = _var_init('weights', shape=[in_num, out_num], init_method=xavier)
    biases = _var_init('biases', [out_num], init_method=constant)
    fc = tf.add(tf.matmul(input_, kernel), biases, name=scope.name)      

    return fc

def iFC(name, input_, size, is_training, dropout):
  
  in_num = input_.get_shape()[-1].value
  out_num = size
  xavier = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  constant = tf.constant_initializer(0.0)
  with tf.variable_scope(name) as scope:
    kernel = _var_init('weights', shape=[in_num, out_num], init_method=xavier)
    biases = _var_init('biases', [out_num], init_method=constant)
    fc = tf.add(tf.matmul(input_, kernel), biases, name=scope.name)      
    #fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True, 
    #    is_training=is_training, decay=0.9, zero_debias_moving_mean=False, 
    #    fused=True, scope='bn')     
    fc = tf.nn.relu(fc, name='relu')
    if is_training is True and dropout < 1.0:
      fc = tf.nn.dropout(fc, keep_prob=dropout, name='dropout')

    return fc

def stack_modern_conv(prefix, ls, le, input_, size, num, 
    is_training, dropout, down_first=True): 
  
  # stack the convolutional layers with same resolution and filter number 
  pre = input_
  for ll in range(ls, le):
    pivot = prefix + str(ll)
    cur_stride = 2 if ll == ls and down_first else 1
    cur_drop = dropout if ll == le else 1.0

    pre = modern_conv(pivot, pre, size, num, cur_stride, 
        is_training=is_training, dropout=cur_drop)

  return pre   		

def modern_conv3d(name, input_, size, num, stride, is_training, dropout):
  # convolutional layer 
  in_num = input_.get_shape()[-1].value
  out_num = num

  xavier = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
  constant = tf.constant_initializer(0.0)
  with tf.variable_scope(name):
    kernel = _var_init('weights', shape=[size, size, size, in_num, out_num], init_method=xavier)
    conv3d = tf.nn.conv3d(input_, kernel, [1, stride, stride, stride, 1], padding='SAME')
    biases = _var_init('biases', [out_num], init_method=constant)
    pre_bn = tf.nn.bias_add(conv3d, biases)

    pre_relu = tf.contrib.layers.batch_norm(pre_bn, center=True, scale=True, 
        is_training=is_training, decay=0.99, zero_debias_moving_mean=False, 
        fused=False, scope='bn')     
    
    final_conv3d = tf.nn.relu(pre_relu, name='relu')
    if is_training is True:
      final_conv3d = tf.nn.dropout(final_conv3d, keep_prob=dropout, name='dropout')

    return final_conv3d

def stack_modern_conv3d(prefix, ls, le, input_, size, num, 
    is_training, dropout, down_first=True): 
  
  # stack the convolutional layers with same resolution and filter number 
  pre = input_
  for ll in range(ls, le):
    pivot = prefix + str(ll)
    cur_stride = 2 if ll == ls and down_first else 1
    cur_drop = dropout if ll == le else 1.0

    pre = modern_conv3d(pivot, pre, size, num, cur_stride, 
        is_training=is_training, dropout=cur_drop)

  return pre   		
