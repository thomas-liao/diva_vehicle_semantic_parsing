'''
Multi-View Network for 3D bbox and orientation regression

Author: Chi Li
Date: June 26, 2017

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import pdb

import net2d
from six.moves import urllib
import tensorflow as tf
import math

def small_keynn_64(image, pose_dim, reuse_flag, tp):
  
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 64, 
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 128, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 256, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 15, 20, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)

    vout = net2d.GAP('vGAP', inet)
    pred_pose = net2d.FC('pose', vout, pose_dim) 
    
    return pred_pose

def keynn_64(image, pose_dim, reuse_flag, tp):
  
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 64, 
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 128, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 256, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 15, 20, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 20, 25, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=False)

    vout = net2d.GAP('vGAP', inet)
    pred_pose = net2d.FC('pose', vout, pose_dim) 
    
    return pred_pose

def keynn_128(image, pose_dim, reuse_flag, tp):
  
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 64, 
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 128, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 256, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 15, 20, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 20, 25, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    
    vout = net2d.GAP('vGAP', inet)
    pred_pose = net2d.FC('pose', vout, pose_dim) 
    
    return pred_pose

def large_keynn_64(image, pose_dim, reuse_flag, tp):
  
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 128, 
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 256, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 15, 20, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 20, 25, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=False)

    vout = net2d.GAP('vGAP', inet)
    pred_pose = net2d.FC('pose', vout, pose_dim) 
    
    return pred_pose

def infer_key(images, key_dim, tp, reuse_=False):
  with tf.variable_scope('Single_VNN', reuse=reuse_):
    #return small_keynn_64(images, key_dim, reuse_, tp)
    #return large_keynn_64(images, key_dim, reuse_, tp)
    return keynn_64(images, key_dim, reuse_, tp)
    #return keynn_128(images, key_dim, reuse_, tp)

def L2_loss_key(pred_key, gt_key, weight_decay):
  loss_group = 'losses'
  n = pred_key.get_shape().as_list()[0]
  assert n == gt_key.get_shape().as_list()[0], "prediction number == gt number"

  key_loss = tf.nn.l2_loss(pred_key - gt_key) / n
  tf.add_to_collection(loss_group, key_loss)
  tf.summary.scalar('key_loss', key_loss)

  # add weight decay loss on all trainable variables
  if weight_decay:
    for ix, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
      wd_reg = tf.multiply(tf.nn.l2_loss(var), weight_decay, 
          name=str(var.name[:-2] + '/weight_decay'))
      tf.add_to_collection(loss_group, wd_reg)

  total_loss = tf.add_n(tf.get_collection(loss_group), name='total_loss')
  
  return total_loss, key_loss

def key23d_64(image, k2d_dim, k3d_dim, reuse_flag, tp):
  
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 64, 
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 128, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 256, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 15, 20, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    
    out_3d = net2d.GAP('vGAP', inet)
    pred3d = net2d.FC('3d', out_3d, k3d_dim) 
    
    inet = net2d.stack_modern_conv(prefix, 20, 25, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=False)

    out_2d = net2d.GAP('vGAP', inet)
    pred2d = net2d.FC('2d', out_2d, k2d_dim) 
    
    return pred2d, pred3d

def infer_23d(images, k2d_dim, k3d_dim, tp, reuse_=False):
  with tf.variable_scope('Single_VNN', reuse=reuse_):
    return key23d_64(images, k2d_dim, k3d_dim, reuse_, tp)

def L2_loss_23d(pred_key, gt_key, weight_decay):
  loss_group = 'losses'
  n = pred_key[0].get_shape().as_list()[0]
  assert n == gt_key[0].get_shape().as_list()[0], "prediction number == gt number"

  key2d_loss = tf.nn.l2_loss(pred_key[0] - gt_key[0]) / n
  tf.add_to_collection(loss_group, key2d_loss)
  tf.summary.scalar('key2d_loss', key2d_loss)
  key3d_loss = tf.nn.l2_loss(pred_key[1] - gt_key[1]) / n
  tf.add_to_collection(loss_group, key3d_loss)
  tf.summary.scalar('key3d_loss', key3d_loss)

  # add weight decay loss on all trainable variables
  if weight_decay:
    for ix, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
      wd_reg = tf.multiply(tf.nn.l2_loss(var), weight_decay, 
          name=str(var.name[:-2] + '/weight_decay'))
      tf.add_to_collection(loss_group, wd_reg)

  total_loss = tf.add_n(tf.get_collection(loss_group), name='total_loss')
  
  return total_loss, key2d_loss

def keyos_64(image, key_dim, reuse_flag, tp):
  
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 64, 
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 128, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 256, 
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 15, 18, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    
    out_os = net2d.GAP('osGAP', inet)
    pred_os = net2d.FC('os', out_os, key_dim) 
    
    inet = net2d.stack_modern_conv(prefix, 18, 21, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=True)
    
    out_3d = net2d.GAP('3dGAP', inet)
    pred3d = net2d.FC('3d', out_3d, key_dim * 3) 
    
    inet = net2d.stack_modern_conv(prefix, 21, 25, inet, 3, 512, 
        is_training=tp, dropout=0.7, down_first=False)

    out_2d = net2d.GAP('2dGAP', inet)
    pred2d = net2d.FC('2d', out_2d, key_dim * 2) 
    
    return pred2d, pred3d, pred_os

def infer_os(images, key_dim, tp, reuse_=False):
  with tf.variable_scope('Single_VNN', reuse=reuse_):
    return keyos_64(images, key_dim, reuse_, tp)

def L2_loss_os(pred_key, gt_key, weight_decay):
  loss_group = 'losses'
  n = pred_key[0].get_shape().as_list()[0]
  assert n == gt_key[0].get_shape().as_list()[0], "prediction number == gt number"

  key2d_loss = tf.nn.l2_loss(pred_key[0] - gt_key[0]) / n
  tf.add_to_collection(loss_group, key2d_loss)
  tf.summary.scalar('key2d_loss', key2d_loss)
  key3d_loss = tf.nn.l2_loss(pred_key[1] - gt_key[1]) / n
  tf.add_to_collection(loss_group, key3d_loss)
  tf.summary.scalar('key3d_loss', key3d_loss)
  os_loss = tf.nn.l2_loss(pred_key[2] - gt_key[2]) / n
  tf.add_to_collection(loss_group, os_loss)
  tf.summary.scalar('os_loss', os_loss)

  # add weight decay loss on all trainable variables
  if weight_decay:
    for ix, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
      wd_reg = tf.multiply(tf.nn.l2_loss(var), weight_decay, 
          name=str(var.name[:-2] + '/weight_decay'))
      tf.add_to_collection(loss_group, wd_reg)

  total_loss = tf.add_n(tf.get_collection(loss_group), name='total_loss')
  return total_loss, key2d_loss


def base_64_key_vnn(image, depth, key_dim, reuse_flag, tp):
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 64,
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 128,
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 256,
        is_training=tp, dropout=0.7, down_first=True)

    dnet = net2d.stack_modern_conv(prefix + '_depth', 1, 5, depth, 3, 64,
        is_training=tp, dropout=0.7, down_first=False)
    dnet = net2d.stack_modern_conv(prefix + '_depth', 5, 10, dnet, 3, 128,
        is_training=tp, dropout=0.7, down_first=True)
    dnet = net2d.stack_modern_conv(prefix + '_depth', 10, 15, dnet, 3, 128,
        is_training=tp, dropout=0.7, down_first=True)

    idnet = tf.concat([inet, dnet], axis=3)

    idnet = net2d.stack_modern_conv(prefix, 15, 22, idnet, 3, 512,
        is_training=tp, dropout=0.7, down_first=True)

    key_out = net2d.GAP('kGAP', idnet)
    pred_key = net2d.FC('kout', key_out, key_dim)

    return pred_key

def infer_key_depth_plain(images, depths, key_dim, tp, reuse_=False):
  with tf.variable_scope('Single_VNN', reuse=reuse_):
    return base_64_key_vnn(images, depths, key_dim, reuse_, tp)

def base_64_ds_key_vnn(image, depth, pose_dim, mask_dim, key_dim, reuse_flag, tp):
  n = image.get_shape().as_list()[0]
  prefix = 'conv'
  with tf.variable_scope('Base_CNN', reuse=reuse_flag):
    inet = net2d.stack_modern_conv(prefix, 1, 5, image, 3, 64,
        is_training=tp, dropout=0.7, down_first=False)
    inet = net2d.stack_modern_conv(prefix, 5, 10, inet, 3, 128,
        is_training=tp, dropout=0.7, down_first=True)
    inet = net2d.stack_modern_conv(prefix, 10, 15, inet, 3, 256,
        is_training=tp, dropout=0.7, down_first=True)

    dnet = net2d.stack_modern_conv(prefix + '_depth', 1, 5, depth, 3, 64,
        is_training=tp, dropout=0.7, down_first=False)
    dnet = net2d.stack_modern_conv(prefix + '_depth', 5, 10, dnet, 3, 128,
        is_training=tp, dropout=0.7, down_first=True)
    dnet = net2d.stack_modern_conv(prefix + '_depth', 10, 15, dnet, 3, 128,
        is_training=tp, dropout=0.7, down_first=True)

    idnet = tf.concat([inet, dnet], axis=3)

    mask_out = tf.reshape(idnet, [n, -1])
    pred_mask = net2d.FC('mout', mask_out, mask_dim * 2)

    idnet = net2d.stack_modern_conv(prefix, 15, 18, idnet, 3, 512,
        is_training=tp, dropout=0.7, down_first=True)
    
    vout = net2d.GAP('vGAP', idnet)
    pred_bin = net2d.FC('bin', vout, pose_dim)
 
    idnet = net2d.stack_modern_conv(prefix, 18, 22, idnet, 3, 512,
        is_training=tp, dropout=0.7, down_first=False)

    key_out = net2d.GAP('kGAP', idnet)
    pred_key = net2d.FC('kout', key_out, key_dim)

    return pred_bin, pred_mask, pred_key

def infer_key_depth_ds(images, depths, bin_dim, mask_dim, key_dim, tp, reuse_=False):
  with tf.variable_scope('Single_VNN', reuse=reuse_):
    return base_64_ds_key_vnn(images, depths, bin_dim, mask_dim, key_dim, reuse_, tp)

def L2_loss_key_ds(pred_bin, gt_bin, pred_mask, gt_mask, pred_key, gt_key, weight_decay):
  loss_group = 'losses'
  n = pred_key.get_shape().as_list()[0]
  assert n == gt_key.get_shape().as_list()[0], "prediction number == gt number"

  key_loss = tf.nn.l2_loss(pred_key - gt_key) / n
  tf.add_to_collection(loss_group, key_loss)
  tf.summary.scalar('key_loss', key_loss)
  
  #bin_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_bin,
  #   labels=gt_bin, name='bin_softmax')
  #bin_loss = tf.reduce_mean(bin_loss, name='theta_ce')
  #tf.add_to_collection(loss_group, bin_loss)
  #tf.summary.scalar('bin_loss', bin_loss)
  
  pm = tf.reshape(pred_mask, [-1, 2])
  gm = tf.reshape(gt_mask, [-1])
  mask_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=gm, logits=pm, name='mask_ce')
  mask_loss = tf.reduce_mean(mask_loss, name='mask_ce_mean')
  tf.add_to_collection(loss_group, mask_loss)
  tf.summary.scalar('mask_loss', mask_loss)

  # add weight decay loss on all trainable variables
  if weight_decay:
    for ix, var in enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
      wd_reg = tf.multiply(tf.nn.l2_loss(var), weight_decay, 
          name=str(var.name[:-2] + '/weight_decay'))
      tf.add_to_collection(loss_group, wd_reg)

  total_loss = tf.add_n(tf.get_collection(loss_group), name='total_loss')
  
  return total_loss, key_loss
