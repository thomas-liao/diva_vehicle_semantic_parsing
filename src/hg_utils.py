"""Thomas Liao
Additional utilty functions for stacked hourglass model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import pdb

from six.moves import urllib
import tensorflow as tf
import numpy as np

# TODO: write docstring, according to PEP8 coding style - https://www.python.org/dev/peps/pep-0008/
def _np_makeGaussian(height, width, fwhm = 3, center=None):
		""" Make a square gaussian kernel.
		size is the length of a side of the square
		fwhm is full-width-half-maximum, which
		can be thought of as an effective radius.
		"""
		x = np.arange(0, width, 1, float)
		y = np.arange(0, height, 1, float)[:, np.newaxis]
		if center is None:
			x0 =  width // 2
			y0 = height // 2
		else:
			x0 = center[0]
			y0 = center[1]
		return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def _np_generate_hm(height, width, joints, maxlenght, weight):
		""" Generate a full Heap Map for every joints in an array
		Args:
			height			: Wanted Height for the Heat Map
			width			: Wanted Width for the Heat Map
			joints			: Array of Joints
			maxlenght		: Lenght of the Bounding Box
		"""
		num_joints = joints.shape[0]
		hm = np.zeros((height, width, num_joints), dtype = np.float32)
		for i in range(num_joints):
			# if not(np.array_equal(joints[i], [-1,-1])) and weight[i] == 1: # 7-9 doesn't need this for tf car keypoint project(at this moment at least)
				s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
				hm[:,:,i] = _np_makeGaussian(height, width, fwhm= s, center= (joints[i, 0], joints[i, 1]))
			# else:
			# 	hm[:,:,i] = np.zeros((height,width))
		return hm

# for tf graph
def _tf_makeGaussian(h, w, fwhm=3, center=None): # good
    """keypoint - > hm for single keypoint"""
    x = tf.range(0, w, 1, dtype=tf.float32)
    y = tf.range(0, h, 1, dtype=tf.float32)
    y = tf.reshape(y, shape=(h, 1))

    if center is None:
        x0 = w // 2
        y0 = h // 2
    else:
        x0 = center[0]
        y0 = center[1]

    # temp1 = tf.multiply(tf.L23d_pmc(tf.constant(2, tf.float32)), tf.constant(-4, dtype=tf.float32) )
    temp1 = tf.constant(-2.7725887, dtype=tf.float32) # save time
    temp2 = tf.divide((tf.square(x - x0) + tf.square(y - y0)), tf.constant(fwhm**2, tf.float32))

    return tf.exp(tf.multiply(temp1, temp2))


def _tf_generate_hm(h, w, keypoints, maxLen=64):  # good
    """maxLen: length of bounding box, for computing fwhm(i.e. full width of half maximum of gaussian distribution,
    maxLen=64 by default(for 64*64 hm)
    """
    res = []
    for i in range(keypoints.get_shape().as_list()[0]):
        #
        # fwhm = int(np.sqrt(maxLen) * maxLen * 10 / 4096) + 2 # res is 3 for 64 * 64 bbx, save time
        # print(fwhm)
        fwhm = 3 # computed as above equation.... just ave time here

        res.append(_tf_makeGaussian(h, w, fwhm=fwhm, center=keypoints[i, :]))

    return tf.stack(res, -1)

def _argmax(tensor): # return: tuple of max position, only works for 2D tensor
    resh = tf.reshape(tensor, [-1]) # flatten
    argmax = tf.arg_max(resh, 0)
    dim = tensor.get_shape().as_list()[0]
    # return (argmax % dim, argmax // dim) # fixed, now (x, y) instead of (i, j) # this order is for tf car keypoint
    return (argmax // dim, argmax % dim) # this is for human keypoitn - datagen - MPII dataset


def _bce_loss(logits, gtMaps, name='ce_loss', weighted=False):
    """BCE loss computation, weighted or not weighted.
    Args:
        logits: output, shape: [batch_size, nStack/Stage, hm_size, hm_size, outputDim]
    """
    with tf.name_scope(name=name):
        if weighted:
            # TODO: implement weighted loss.. yet to be tested
            print("function not implemented yet")
        else:
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=gtMaps))

def _hm2kp(gt_key_hm):
    """Convert single hm of single image to 2d keypoint, shape=(outDim, 2)
    There could be 2 different shapes, i.e. only input last nStack-1 : [64, 64, outDim] or all stacks [nStack, 64, 64, outDim]
    Returns:
        2d keypoint, shape:(36, 2)
    """
    shape = gt_key_hm.get_shape().as_list()

    outDim = shape[-1]
    temp = []
    for i in range(outDim):
        if len(shape) == 3:
            resh = tf.reshape(gt_key_hm[:, :, i], shape=[-1])
        elif len(shape) == 4:
            resh = tf.reshape(gt_key_hm[-1, :, :, i], shape=[-1])
        else:
            print("not a valid shape for heatmap")
        max_idx_flat = tf.arg_max(resh, 0)
        # pt = (max_idx_flat % shape[-2], max_idx_flat // shape[-2]) # fixed, for car keypoint
        pt = (max_idx_flat // shape[-2], max_idx_flat % shape[-2])
        # cast to uint8
        pt = tf.cast(pt, dtype=tf.uint8) # may need to change to others, e.g. uint64, if img w or h > 255
        temp.append(pt)
    return tf.stack(temp, axis=0)

def _hm2kp_batch(batch_gt_key_hm): # no sigmoid, no threshold.....
    """Convert batch_gt_key_hm  to a batch of 2d keypoint
     Args:
         batch_gt_key_hm: output of caroutput of car_key_stacked_hourglass model, shape:(batch_size, nStack, 64, 64, 36]
     Returns:
         batch of 2d keypoints, shape: (batch_size, 36, 2)
    """
    shape = batch_gt_key_hm.get_shape().as_list()
    if len(shape) != 5:
        print('not a valid shape of batch_gt_key_hm, i.e. shape:(batch_size, nStack, h_hm, w_hm, out_dim')
    batch_size = shape[0]
    temp = []
    for i in range(batch_size):
        pred = _hm2kp(batch_gt_key_hm[i, -1]) # pred: # last stack..
        temp.append(pred)
    return tf.stack(temp, axis=0) #


def _hm2kp_sigmoid_thresh(gt_key_hm_sigmoid, thresh=0.2, sess=None):
    """Convert single hm sigmoid result of single image to 2d keypoint, shape=(outDim, 2)
    There could be 2 different shapes, i.e. only input last nStack-1 : [64, 64, outDim] or all stacks [nStack, 64, 64, outDim]
    if sigmoid result of a specific kp is lower than thresh, return (-1, -1) for that joint, as unpredictable.
    Returns:
        2d keypoint, shape:(36, 2)
    """
    shape = gt_key_hm_sigmoid.get_shape().as_list()
    outDim = shape[-1]
    temp = []
    th = tf.constant(thresh, dtype=tf.float32)
    for i in range(outDim):
        if len(shape) == 3:
            hm = gt_key_hm_sigmoid[:, :, i]
            resh = tf.reshape(gt_key_hm_sigmoid[:, :, i], shape=[-1])
        elif len(shape) == 4:
            hm = gt_key_hm_sigmoid[-1, :, :, i]
            resh = tf.reshape(gt_key_hm_sigmoid[-1, :, :, i], shape=[-1])
        else:
            print("not a valid shape for heatmap")
        max_idx_flat = tf.arg_max(resh, 0)
        # pt = (max_idx_flat % shape[-2], max_idx_flat // shape[-2]) # fixed, for car keypoint
        pt = [max_idx_flat // shape[-2], max_idx_flat % shape[-2]]
        # cast to uint8
        pt = tf.cast(pt, dtype=tf.uint8) # may need to change to others, e.g. uint64, if img w or h > 255

        pt = tf.py_func(helper, [hm, pt, th], tf.int8)
        # print(sess.run(pt))
        temp.append(pt)

    return tf.stack(temp, axis=0)

def helper(hm, pt, thresh):
    """helper function for _hm2kp_sigmoid_thresh
    Args:
        hm: single keypoint heat map, shape: (64, 64), dtype = tf.float32 - > np.float32 (tf.py_func)
    Returns:
        if hm[pt[0], pt[1]] > thresh, return pt as is, else return (-1, -1)
    """
    if hm[pt[0], pt[1]] < thresh:
        # print(hm[0,0])
        pt = np.array([-1, -1], dtype=np.int8)
    else:
        pt = np.array(pt, dtype=np.int8)
    return pt


def _hm2kp_batch_thresh(batch_gt_key_hm_sigmoid, threshold=0.2):
    """Convert batch_gt_key_hm  to a batch of 2d keypoint
     Args:
         batch_gt_key_hm: output of caroutput of car_key_stacked_hourglass model, shape:(batch_size, nStack, 64, 64, 36]
     Returns:
         batch of 2d keypoints, shape: (batch_size, 36, 2)
    """
    shape = batch_gt_key_hm_sigmoid.get_shape().as_list()
    if len(shape) == 5:
        # by default, use nStack-1
        batch_gt_key_hm_sigmoid = batch_gt_key_hm_sigmoid[:, -1]
    else:
        if len(shape) != 4:
            raise ValueError('Invalid shape for batch_gt_key_hm_sigmoid')

    batch_size = shape[0]
    temp = []
    for i in range(batch_size):
        pred = _hm2kp_sigmoid_thresh(batch_gt_key_hm_sigmoid[i], thresh=threshold) # pred:
        temp.append(pred)
    return tf.stack(temp, axis=0) #

### some bug in this section...TODO: fix bugs
# # def _compute_err(hm1, hm2, diag_len=89.1, sess=None):
# def _compute_err(hm1, hm2, diag_len=89.1, sess=None): # fixed, good
#     """Computer err of two heat map.. input need to be 2D tensor
#     Args:
#         hm1, hm2: 2 input heat map(2D Tensor) with shape: (H, W)
#     Return:
#         (float): dist norm to [0, 1] (by diag len), as err metrics
#     """
#     hm1_x, hm1_y = _argmax(hm1)
#     hm2_x, hm2_y = _argmax(hm2)
#     # print(sess.run([hm1_x, hm1_y, hm2_x, hm2_y]))
#     # dist = tf.sqrt(tf.square(hm1_x - hm2_x) + tf.square(hm1_y - hm2_y)) # bug
#     dist = tf.sqrt(tf.square(tf.to_float(hm1_x - hm2_x)) + tf.square(tf.to_float(hm1_y - hm2_y)))
#     diag_len = tf.to_float(diag_len)
#     return tf.divide(dist, diag_len)
#
# # def _compute_err( u, v):
# #     """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
# #     Args:
# #         u		: 2D - Tensor (Height x Width : 64x64 )
# #         v		: 2D - Tensor (Height x Width : 64x64 )
# #     Returns:
# #         (float) : Distance (in [0,1])
# #     """
# #     u_x, u_y = _argmax(u)
# #     v_x, v_y = _argmax(v)
# #     return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(89.1))
#
#
# def _compute_avg_accuracy(pred, gt, batch_size=16, sess=None):
#     """Compute avg. accuracy w.r.t. a single keypoint for batch
#     Args:
#         pred, gt : pred batch, gt batch for a single keypoint, shape : (B, hm_dim, hm_dim), i.e. (16, 64, 64)
#     Returns:
#         (float) value for avg accuracy, computed as 1 - err
#     """
#     # err_sum = tf.constant(0, dtype=tf.float32)
#     err_sum = tf.to_float(0)
#     for i in range(batch_size):
#         # single keypoint, single img - 2D tensor
#         err_sum = tf.add(err_sum, _compute_err(pred[i], gt[i], sess=sess))
#         # print("err_sum: ", sess.run(err_sum))
#         # err_sum = tf.assign_add(err_sum, _compute_err(pred[i], gt[i]))
#         return tf.subtract(tf.to_float(1), err_sum / batch_size)
#
#
# # def _accuracy_computation(output, gtMaps, nStack=4, sess=None):
# #     """Compute avg accuracy w.r.t. each keypoint in a batch
# #     Args:
# #         output, gtMaps: [Batch_size, nStack, hm_dim, hm_dim, outDim] i.e. [16, 4, 64, 64, 36]
# #     """
# #     shape = output.get_shape().as_list()
# #     num_keypoints = shape[-1]
# #     batch_size = shape[0]
# #
# #     ret = []
# #     for i in range(num_keypoints):
# #         single_kp_avg_accur = _compute_avg_accuracy(output[:, nStack-1, :, :, i], gtMaps[:, nStack-1, :, :, i], sess=sess)
# #         ret.append(single_kp_avg_accur)
# #     return ret
#
# def _accuracy_computation(output, gtMaps, nStack=4, batchSize=2):
#     """ Computes accuracy tensor
#     """
#     joint_accur = []
#     for i in range(4):
#          joint_accur.append(
#              _compute_avg_accuracy( output[:,  nStack - 1, :, :, i],  gtMaps[:,  nStack - 1, :, :, i],
#                          batchSize))
#     return joint_accur


def _compute_err(u, v):
    """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
    Args:
        u		: 2D - Tensor (Height x Width : 64x64 )
        v		: 2D - Tensor (Height x Width : 64x64 )
    Returns:
        (float) : Distance (in [0,1])
    """
    u_x, u_y = _argmax(u)
    v_x, v_y = _argmax(v)
    return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(89.1))


def _accur( pred, gtMap, num_image):
    """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
    returns one minus the mean distance.
    Args:
        pred		: Prediction Batch (shape = num_image x 64 x 64)
        gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
        num_image 	: (int) Number of images in batch
    Returns:
        (float)
    """
    err = tf.to_float(0)
    for i in range(num_image):
        err = tf.add(err, _compute_err(pred[i], gtMap[i]))
    return tf.subtract(tf.to_float(1), err / num_image)


def _accuracy_computation(output, gtMaps, nStack=4, batchSize=16):
    """ Computes accuracy tensor
    """
    joint_accur = []
    num_kp = output.get_shape().as_list()[-1]
    for i in range(num_kp): # need to modify
         joint_accur.append(_accur(output[:,  nStack - 1, :, :, i],  gtMaps[:,  nStack - 1, :, :, i], batchSize))
    return joint_accur


######################################################################################################################
#######################################Sanity_Check_start#############################################################
######################################################################################################################
## for test
# test = np.arange(64*64*36).reshape(64, 64, 36)
# test = tf.convert_to_tensor(test)
#
# res = _hm2kp(test)
#
# # print(res)
# gather = tf.gather(res,0)
#
# with tf.Session() as sess:
#     print(sess.run(gather))


## test _tf_makeGaussian(*args)
# res = _tf_makeGaussian(2, 3, 1, center=[2,0])
#
# with tf.Session() as sess:
#     print(sess.run(res))

# test _tf_generate_hm
# joints = np.array([[0, 1], [1, 2]])
# joints_tf = tf.constant(joints, dtype=tf.float32)
#
# res = _tf_generate_hm(4, 4, joints_tf, 4)
#
# with tf.Session() as sess:
#     print(sess.run(res))


## test ce loss
# output_joints = np.array([[0, 1], [0, 2], [4, 5], [6, 8], [63, 0]], dtype=np.float32)
# # gt_joints = np.array([[9, 9], [9, 9], [9, 9], [9, 9], [9, 9]], dtype=np.float32)
# output_joints = np.arange(72, dtype=np.float32).reshape(36, 2)
# # print(output_joints)
# gt_joints = np.arange(72, dtype=np.float32).reshape(36, 2)
#
# output_joints = tf.constant(output_joints, dtype=tf.float32)
# gt_joints = tf.constant(gt_joints, dtype=tf.float32)
#
# output = _tf_generate_hm(80, 80, keypoints=output_joints, maxLen=80)
# gtMaps = _tf_generate_hm(64, 64, keypoints=gt_joints, maxLen=64)
# # res = _bce_loss(logits=output, gtMaps=gtMaps)
#
# # test hm2kp
# kp = _hm2kp(output)

#
#
# # res_test = _bce_loss(logits=np.arange(4,dtype=np.float32), gtMaps=np.arange(4, dtype=np.float32))
# with tf.Session() as sess:
#     print(sess.run(kp))
#     # print(sess.run(output))

## test _compute_err() - single keypoint hm
# a = np.zeros(shape=(64, 64), dtype=np.float32)
# a[0, 0] = 1   # np:[a, b], point: [b, a]
# b = np.zeros(shape=(64, 64), dtype=np.float32)
# b[63, 63] = 1 # np:[a, b], point: [b, a]
#
# a = tf.constant(a, dtype=tf.float32)
# b = tf.constant(b, dtype=tf.float32)
#
# with tf.Session() as sess:
#     res = _compute_err(a, b)
#
#     print(sess.run(res))
#     # print('')
#
# print(np.sqrt(63**2 + 63**2) / 89.1)


# ## test _accuracy_computation
# output = np.zeros((16, 4, 64, 64, 4), np.float32) # 4 kps, (0, 0), (1, 1), (2, 2), (3, 3)
# output[:, :, 0, 0, 0] = 1
# output[:, :, 1, 1, 1] = 1
# output[:, :, 2, 2, 2] = 1
# output[:, :, 3, 3, 3] = 1
#
#
# gtMaps = np.zeros((16, 4, 64, 64, 4), np.float32) # 4 kps
# gtMaps[:, :, 63, 63, 0] = 1
# gtMaps[:, :, 1, 1, 1] = 1
# gtMaps[:, :, 2, 2, 2] = 1
# gtMaps[:, :, 3, 3, 3] = 1
#
# # wrapper
# output = tf.constant(output, dtype=tf.float32)
# gtMaps = tf.constant(gtMaps, dtype=tf.float32)
#
#
#
#
# with tf.Session() as sess:
#     res = _accuracy_computation(output, gtMaps, batchSize=output.get_shape().as_list()[0])
#     print(sess.run(res))


# def _hm2kp_sigmoid_thresh(gt_key_hm_sigmoid, thresh=0.2):
#     """Convert single hm sigmoid result of single image to 2d keypoint, shape=(outDim, 2)
#     There could be 2 different shapes, i.e. only input last nStack-1 : [64, 64, outDim] or all stacks [nStack, 64, 64, outDim]
#     if sigmoid result of a specific kp is lower than thresh, return (-1, -1) for that joint, as unpredictable.
#     Returns:
#         2d keypoint, shape:(36, 2)
#     """
#     a = 1
#     return a


# test = np.random.randint(10, size=64*64*36).reshape(64, 64, 36)
# test = np.arange(16*64*64*36).reshape(16, 64, 64, 36)
# test = test.astype(np.float32)
# test = test / 60000000.0
# test = tf.convert_to_tensor(test, dtype=tf.float32)
# test = tf.nn.sigmoid(test)
#
#
#
# with tf.Session() as sess:
#     pt = _hm2kp_batch_thresh(test, threshold=0.2)
#     print(sess.run(pt))
#
#     # pt = _hm2kp_sigmoid_thresh(test, thresh=0, sess=sess)
#     # print(sess.run(pt))
#     # print(sess.run(test))



# res = _hm2kp_sigmoid_thresh()


######################################################################################################################
#######################################Sanity_Check_end###############################################################
######################################################################################################################