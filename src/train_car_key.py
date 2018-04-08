# Python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import os.path as osp

import tensorflow as tf
import numpy as np
import random
import pdb
import math

import single_key_net as sk_net 
import time
import datetime
from utils import preprocess
from utils import ndata_tfrecords 
from utils import optimizer 
from utils import vis_keypoints 
from tran3D import quaternion_matrix
from tran3D import polar_to_axis
from tran3D import quaternion_about_axis

from scipy.linalg import logm
import cv2

def read_one_datum(fqueue, dim, key_num=36):
  reader = tf.TFRecordReader()
  key, value = reader.read(fqueue)
  basics = tf.parse_single_example(value, features={
    'key2d': tf.FixedLenFeature([key_num * 2], tf.float32),
    'image': tf.FixedLenFeature([], tf.string)})

  image = basics['image']
  image = tf.decode_raw(image, tf.uint8)
  image.set_shape([3 * dim * dim])
  image = tf.reshape(image, [dim, dim, 3])

  key = basics['key2d']

  return image, key 
    
def create_bb_pip(tfr_pool, nepoch, sbatch, mean, shuffle=True):
  if len(tfr_pool) == 3:
    ebs = [int(sbatch * 0.5), int(sbatch * 0.3), sbatch - int(sbatch * 0.5) - int(sbatch * 0.3)]
  elif len(tfr_pool) == 1:
    ebs = [sbatch]
  else:
    print("Input Format is not recognized")
    return

  data_pool = []

  for ix, tfr in enumerate(tfr_pool): 
    cur_ebs = ebs[ix]
    tokens = tfr.split('/')[-1].split('_')
    dim = int(tokens[-1].split('.')[0][1:])
    tf_mean = tf.constant(mean, dtype=tf.float32)
    tf_mean = tf.reshape(tf_mean, [1, 1, 1, 3])
    
    fqueue = tf.train.string_input_producer([tfr], num_epochs=nepoch)
    image, gt_key = read_one_datum(fqueue, dim)

    if shuffle:
      data = tf.train.shuffle_batch([image, gt_key], batch_size=cur_ebs, 
        num_threads=12, capacity=sbatch * 6, min_after_dequeue=cur_ebs * 3)
    else:
      data = tf.train.batch([image, gt_key], batch_size=cur_ebs,
          num_threads=12, capacity=cur_ebs * 5)
    
    # preprocess input images 
    data[0] = preprocess(data[0], tf_mean)
    if ix == 0:
      for j in range(len(data)):
        data_pool.append([data[j]])
    else:
      for j in range(len(data)):
        data_pool[j].append(data[j])
    
  combined_data = []
  for dd in data_pool:
    combined_data.append(tf.concat(dd, axis=0))

  return combined_data

def vis2d_one_output(image, pred_key, gt_key):
  image += 128
  image.astype(np.uint8)
  dim = image.shape[0]		# height == width by default
  
  left = np.copy(image)
  right = np.copy(image)
  
  pk = np.reshape(pred_key, (36, 2)) * dim
  gk = np.reshape(gt_key, (36, 2)) * dim
  pk.astype(np.int32)
  gk.astype(np.int32)

  for pp in pk:
    cv2.circle(left, (pp[0], pp[1]), 2, (0, 0, 255), -1)

  for pp in gk:
    cv2.circle(right, (pp[0], pp[1]), 2, (0, 0, 255), -1)

  final_out = np.hstack([left, right])

  outputs = np.zeros((1, final_out.shape[0], final_out.shape[1], 3), dtype=np.uint8)
  outputs[0] = final_out

  return outputs 

def add_bb_summary(images, pred_keys, gt_keys, name_prefix, max_out=1):
  n = images.get_shape().as_list()[0]

  for i in range(np.min([n, max_out])):
    pred = tf.gather(pred_keys, i)
    gt = tf.gather(gt_keys, i)
    image = tf.gather(images, i)
    
    result = tf.py_func(vis2d_one_output, [image, pred, gt], tf.uint8)

    tf.summary.image(name_prefix + '_result_' + str(i), result, 1)

def eval_one_epoch(sess, val_loss, niters):
  total_loss = .0
  for i in xrange(niters):
    cur_loss = sess.run(val_loss)
    total_loss += cur_loss 
  return total_loss / niters


def train(input_tfr_pool, val_tfr_pool, out_dir, log_dir, mean, sbatch, wd):
  """Train Multi-View Network for a number of steps."""
  log_freq = 100
  val_freq = 1000
  model_save_freq = 3000
  tf.logging.set_verbosity(tf.logging.ERROR)

  # maximum epochs
  total_iters = 200000 
  lrs = [0.01, 0.001, 0.0001]
  steps = [int(total_iters * 0.5), int(total_iters * 0.4), int(total_iters * 0.1)]
   
  # set config file 
  config = tf.ConfigProto(log_device_placement=False)
  with tf.Graph().as_default():
    sys.stderr.write("Building Network ... \n")
    global_step = tf.contrib.framework.get_or_create_global_step()
    
    images, gt_key = create_bb_pip(input_tfr_pool, 1000, sbatch, mean, shuffle=True)

    # inference model
    key_dim = gt_key.get_shape().as_list()[1]
    pred_key = sk_net.infer_key(images, key_dim, tp=True)
    
    # Calculate loss
    total_loss, data_loss = sk_net.L2_loss_key(pred_key, gt_key, weight_decay=wd)
    train_op = optimizer(total_loss, global_step, lrs, steps)
    sys.stderr.write("Train Graph Done ... \n")
    add_bb_summary(images, pred_key, gt_key, 'train', max_out=3)
    
    if val_tfr_pool:
      val_pool = []
      val_iters = []
      for ix, val_tfr in enumerate(val_tfr_pool):
        total_val_num = ndata_tfrecords(val_tfr)
        total_val_iters = int(float(total_val_num) / sbatch)
        val_iters.append(total_val_iters)
        val_images, val_gt_key = create_bb_pip([val_tfr], 
            1000, sbatch, mean, shuffle=False)
        
        val_pred_key = sk_net.infer_key(val_images, key_dim, tp=False, reuse_=True)
        _, val_data_loss = sk_net.L2_loss_key(val_pred_key, val_gt_key, None)
        val_pool.append(val_data_loss)
        add_bb_summary(val_images, val_pred_key, val_gt_key, 'val_c' + str(ix), max_out=3)
      sys.stderr.write("Validation Graph Done ... \n")

    # merge all summaries
    merged = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(),
        tf.local_variables_initializer())

    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      model_saver = tf.train.Saver()
      
      sys.stderr.write("Initializing ... \n")
      # initialize graph
      sess.run(init_op)

      # initialize the queue threads to start to shovel data
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
      
      model_prefix = os.path.join(out_dir, 'single_key')
      timer = 0
      timer_count = 0
      
      sys.stderr.write("Start Training --- OUT DIM: %d\n" % (key_dim))
      for i in xrange(total_iters):
        ts = time.time()
        if i > 0 and i % log_freq == 0:
          key_loss, _, summary = sess.run([data_loss, train_op, merged])

          summary_writer.add_summary(summary, i)
          summary_writer.flush()
          
          sys.stderr.write('Training %d (%fs) --- Key L2 Loss: %f\n'
              % (i, timer / timer_count, key_loss))
          timer = 0
          timer_count = 0
        else:
          sess.run([train_op])
          timer += time.time() - ts
          timer_count += 1
        
        if val_tfr and i > 0 and i % val_freq == 0:
          sys.stderr.write('Validation %d\n' % i)
          for cid, v_dl in enumerate(val_pool):
            val_key_loss = eval_one_epoch(sess, v_dl, val_iters[cid])
            sys.stderr.write('Class %d --- Key L2 Loss: %f\n' % (cid, val_key_loss))

        if i > 0 and i % model_save_freq == 0:
          model_saver.save(sess, model_prefix, global_step=i)
          
      model_saver.save(sess, model_prefix, global_step=i)
      
      summary_writer.close() 
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=5)
   

def main(FLAGS):
  assert tf.gfile.Exists(FLAGS.input)
  mean = [int(m) for m in FLAGS.mean.split(',')] 
  if tf.gfile.Exists(FLAGS.out_dir) is False:
    tf.gfile.MakeDirs(FLAGS.out_dir)

  with open(osp.join(FLAGS.out_dir, 'meta.txt'), 'w') as fp:
    dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    fp.write('train_single_key.py --- %s\n' % dt)
    fp.write('input: %s\n' % FLAGS.input)
    fp.write('weight decay: %f\n' % FLAGS.wd)
    fp.write('batch: %d\n' % FLAGS.batch)
    fp.write('mean: %s\n' % FLAGS.mean)

  log_dir = osp.join(FLAGS.out_dir, 'log')
  if tf.gfile.Exists(log_dir) is False:
    tf.gfile.MakeDirs(log_dir)
  else:
    for ff in os.listdir(log_dir):
      os.unlink(osp.join(log_dir, ff))

  model_dir = osp.join(FLAGS.out_dir, 'model')
  if tf.gfile.Exists(model_dir) is False:
    tf.gfile.MakeDirs(model_dir)

  train_files = ['syn_car_full_train_d64.tfrecord', 'syn_car_crop_train_d64.tfrecord', 'syn_car_multi_train_d64.tfrecord'] 
  val_files = ['syn_car_full_val_d64.tfrecord', 'syn_car_crop_val_d64.tfrecord', 'syn_car_multi_val_d64.tfrecord'] 
  train_files = [osp.join(FLAGS.input, tt) for tt in train_files] 
  val_files = [osp.join(FLAGS.input, tt) for tt in val_files] 

  train(train_files, val_files, model_dir, log_dir, mean, FLAGS.batch, FLAGS.wd)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--out_dir',
      type=str,
      default='log',
      help='Directory of output training and log files'
  )
  parser.add_argument(
      '--input',
      type=str,
      default='/home/chi/syn_dataset/tfrecord/car/v0',
      help='Directory of input directory'
  )
  parser.add_argument(
      '--mean',
      type=str,
      default='128,128,128',
      help='Directory of input directory'
  )
  parser.add_argument(
      '--wd', 
      type=float, 
      default=0, 
      help='Weight decay of the variables in network.'
  )
  parser.add_argument(
      '--batch', 
      type=int, 
      default=100, 
      help='batch size.'
  )
  parser.add_argument('--debug', action='store_true', help='debug mode')

  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS)
