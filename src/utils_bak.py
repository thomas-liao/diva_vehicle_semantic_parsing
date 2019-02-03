# Python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

import argparse
import os
import sys
import os.path as osp
import numpy as np
import pdb
import math

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import socket
from sklearn.neighbors import KDTree

hostname = socket.gethostname()
if hostname == 'supermicro':
  import cv2

colors = [(1.0, 0, 0),
          (0, 1.0, 0),
          (0, 0, 1.0),
          (0.75, 0.5, 0),
          (1.0, 0, 1.0),
          (0, 1.0, 1.0),
          (0.5, 0.5, 0.5),
          (0.5, 1.0, 0),
          (0, 0.5, 1.0)]

def ndata_tfrecords(filename):
  if osp.exists(filename + '.num'):
    with open(filename + '.num', 'r') as fp:
      return int(fp.readline())
  else:
    c = 0
    for record in tf.python_io.tf_record_iterator(filename):
      c += 1
    with open(filename + '.num', 'w') as fp:
      fp.write("%d" % c)
    return c

def preprocess(images, mean):
  new_images = tf.cast(images, tf.float32)
  return tf.subtract(new_images, mean)

# try norm /255....
def preprocess_norm(images):
  new_images = tf.cast(images,tf.float32)
  new_images /= 255
  return new_images

def optimizer(total_loss, step_counter, lr_values, lr_steps, moving_decay=0.9):
  
  # Decay the learning rate at fixed intervals.
  lr_bound = [lr_steps[0]]
  for i in range(1, len(lr_steps) - 1):
    lr_bound.append(lr_bound[i - 1] + lr_steps[i])
  lr_bound = [np.int64(lb) for lb in lr_bound]

  lr = tf.train.piecewise_constant(step_counter, lr_bound, lr_values, 
      name='learning_rate')
  tf.summary.scalar('learning_rate', lr)

  momentum = 0.9
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    opt = tf.train.MomentumOptimizer(lr, momentum)  
    grads = opt.compute_gradients(total_loss)
    #for gg in grads[-12:]:
    #  tf.summary.histogram(gg[1].name + '/gradients', gg[0])
      
    sys.stderr.write("Number of Grads: %d\n" % len(grads))
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=step_counter)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(moving_decay, 
      step_counter)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op, variable_averages    

def vis_keypoints(key):
  '''
  Visualize keypoints and return output image 
    Args:
      bb: A keypoint vector, each row is (x1, y1, z1, ...)
  
  '''

  key = np.reshape(key, (-1, 3))
  key = np.transpose(key)

  fig = Figure()
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111, projection='3d')
  ax.view_init(elev=-90, azim=270)

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  ax.scatter(key[0], key[1], key[2], c='b', marker='.', s=20)

  v_range = 0.5
  ax.set_xlim(-v_range, v_range, auto=False)
  ax.set_ylim(-v_range, v_range, auto=False)

  canvas.draw()

  data = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(canvas.get_width_height()[::-1] + (3,))
  
  h = data.shape[0]
  w = data.shape[1]
  data = data[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8), :]

  outputs = np.zeros((1, data.shape[0], data.shape[1], 3), dtype=np.uint8)
  outputs[0] = data

  return outputs

def best_fit_transform(A, B):
  '''
  Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
  Input:
    A: Nxm numpy array of corresponding points
    B: Nxm numpy array of corresponding points
  Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
  '''

  assert A.shape == B.shape

  # get number of dimensions
  m = A.shape[1]

  # translate points to their centroids
  centroid_A = np.mean(A, axis=0)
  centroid_B = np.mean(B, axis=0)
  AA = A - centroid_A
  BB = B - centroid_B

  # rotation matrix
  H = np.dot(AA.T, BB)
  U, S, Vt = np.linalg.svd(H)
  R = np.dot(Vt.T, U.T)

  # special reflection case
  if np.linalg.det(R) < 0:
     Vt[m - 1, :] *= -1
     R = np.dot(Vt.T, U.T)

  # translation
  t = centroid_B.T - np.dot(R, centroid_A.T)

  # homogeneous transformation
  T = np.identity(m + 1)
  T[:m, :m] = R
  T[:m, m] = t

  return T

def icp_refine(src, dst, max_iters=15, debug=False):
  '''
    src: the complete object mesh, Nx3
    dst: the visible part predicted by the network, Mx3

  '''
  if len(dst) < 100:
    return src

  outlier_th = 0.02
  stop_th = 0.01

  kdt = KDTree(src, metric='euclidean')
  all_RT = np.eye(4)

  tmpc = np.copy(dst)
  for i in range(max_iters):
    dist, ind = kdt.query(tmpc, k=1)
    inliers = dist < outlier_th
    if len(inliers) < 7 or np.max(dist) < stop_th:
      break
    if debug:
      print(np.mean(dist), np.sum(inliers))
    RT = best_fit_transform(tmpc[inliers.flatten()], src[ind[inliers]])
    if np.sum(np.isfinite(RT)) < np.product(RT.shape):
      break

    all_RT = np.matmul(RT, all_RT)
    tmpc = np.transpose(np.matmul(RT[:3, :3], np.transpose(tmpc))) + RT[:3, 3]

  icp_RT = np.linalg.inv(all_RT)
  return np.transpose(np.matmul(icp_RT[:3, :3], np.transpose(src))) + icp_RT[:3, 3]
