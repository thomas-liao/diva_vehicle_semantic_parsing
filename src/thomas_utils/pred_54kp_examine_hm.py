#!/usr/bin/env python
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
from PIL import Image
from multiprocessing import Pool

import single_key_net as sk_net
import socket
import time
import datetime
import cv2

from utils import preprocess
import hg_utils as ut
import net2d_hg_modified_v1 as hg

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

colors = [(241,242,224), (196,203,128), (136,150,0), (64,77,0),
				(201,230,200), (132,199,129), (71,160,67), (32,94,27),
				(130,224,255), (7,193,255), (0,160,255), (0,111,255),
				(220,216,207), (174,164,144), (139,125,96), (100,90,69),
				(252,229,179), (247,195,79), (229,155,3), (155,87,1),
				(231,190,225), (200,104,186), (176,39,156), (162,31,123),
				(210,205,255), (115,115,229), (80,83,239), (40,40,198),
             (70, 255, 255), (50, 132, 123), (132, 50, 132), (94, 32, 27),
             (56, 83,222), (230, 132, 129), (96, 138, 125), (80, 39, 100),
             ]

# sanity check
# colors = []
# for i in range(50):
#     colors.append((255, 255, 255))

order_54 = order = [
           u'0', u'1', u'2', u'3', u'4', u'7', u'8', u'9', u'10', u'11', u'12', u'13', u'14', u'15',

           u'16', u'17', u'0_flipped', u'1_flipped', u'2_flipped', u'3_flipped', u'4_flipped',

           u'7_flipped', u'8_flipped', u'9_flipped', u'10_flipped', u'11_flipped',

           u'12_flipped', u'13_flipped', u'14_flipped', u'15_flipped', u'16_flipped',

           u'17_flipped', u'FrontDoor1', u'FrontDoor2', u'FrontDoor3', u'FrontDoor4',

           u'BackDoor1', u'BackDoor2', u'BackDoor3', u'BackDoor4', u'Trunk2', u'Trunk3',

           u'Trunk4', u'FrontDoor1_flipped', u'FrontDoor2_flipped',

           u'FrontDoor3_flipped', u'FrontDoor4_flipped', u'BackDoor1_flipped',

           u'BackDoor2_flipped', u'BackDoor3_flipped', u'BackDoor4_flipped',

           u'Trunk2_flipped', u'Trunk3_flipped', u'Trunk4_flipped']

door1 = [u'FrontDoor1', u'FrontDoor2', u'FrontDoor3', u'FrontDoor4'] # right front
door2 = [u'FrontDoor1_flipped', u'FrontDoor2_flipped',
          u'FrontDoor3_flipped', u'FrontDoor4_flipped']
door3 = [u'BackDoor1', u'BackDoor2', u'BackDoor3', u'BackDoor4']
door4 = [u'BackDoor1_flipped',
           u'BackDoor2_flipped', u'BackDoor3_flipped', u'BackDoor4_flipped']
trunk = [u'Trunk2', u'Trunk3', u'Trunk4', u'Trunk2_flipped',u'Trunk4_flipped', u'Trunk3_flipped']

order_chi = [
           u'0', u'1', u'2', u'3', u'4', u'Trunk2', u'Trunk4', u'7', u'8', u'9', u'10', u'11', u'12', u'13', u'14', u'15',

           u'16', u'17',  u'0_flipped', u'1_flipped', u'2_flipped', u'3_flipped', u'4_flipped', u'Trunk2_flipped', u'Trunk4_flipped',

           u'7_flipped', u'8_flipped', u'9_flipped', u'10_flipped', u'11_flipped', u'12_flipped', u'13_flipped', u'14_flipped',

           u'15_flipped', u'16_flipped', u'17_flipped']

matching_idx = [order_54.index(h) for h in order_chi] # 54 - > 36 rigid key points



door1_idx = [order_54.index(h) for h in door1]


door2_idx = [order_54.index(h) for h in door2]
door3_idx = [order_54.index(h) for h in door3]
door4_idx = [order_54.index(h) for h in door4]
trunk_idx = [order_54.index(h) for h in trunk]
rigid_kp_idx = matching_idx



def_kp_idx = door1_idx + door2_idx + door3_idx + door4_idx + trunk_idx

FLAG_54_to_36 = True

def compute_xy_one_angle(one_bin, one_delta):
    pivot = np.argmax(one_bin)
    bin_size = 2.0 / one_bin.size
    return (pivot + 0.5) * bin_size + one_delta[pivot] - 1.0


def show_img_hms(img, img_ori, hms):
    """utility function to draw original image and combined heatmaps side-by-side, for clearer effect, resized to 512 by 512
    image: original single image
    hms: shape (64, 64, num_keypoints)
    """
    canvas_rigid = np.zeros((256, 256, 3))
    canvas_def = np.zeros((256, 256, 3))

    img, _ = noniso_warp(img, 256)
    img_ori, _ = noniso_warp(img_ori, 256)

    # rigid keypoints
    for i in range(len(rigid_kp_idx)):
        temp = hms[:, :, rigid_kp_idx[i]]
        temp = np.stack([temp,]*3, axis=-1)
        temp = cv2.resize(temp, (256, 256))
        # rescale to 0-255
        temp -= temp.min()
        temp /= temp.max()
        temp *= colors[i]
        canvas_rigid += temp

    # temp = hms[:, :, rigid_kp_idx[1]]
    # temp = np.stack([temp, ] * 3, axis=-1)
    # temp = cv2.resize(temp, (256, 256))
    # # rescale to 0-255
    # temp -= temp.min()
    # temp /= temp.max()
    # temp *= colors[1]
    # canvas_rigid += temp

    # # # def key points
    for i in range(len(def_kp_idx)):
        temp = hms[:, :, rigid_kp_idx[i]]
        temp = np.stack([temp, ] * 3, axis=-1)
        temp = cv2.resize(temp, (256, 256))
        # rescale to 0-255
        temp -= temp.min()
        temp /= temp.max()
        temp *= colors[i]
        canvas_rigid += temp

    # canvas_rigid += img_ori*0.1
    # canvas_def += img_ori*0.1


    # # pad and add text
    # img_ori  = np.pad(img_ori, [[0, 40], [0, 0],[0,0]], 'constant')
    # img_ori_shape = img_ori.shape
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (0 + 5, img_ori_shape[0] - 20)
    # fontScale = 0.5
    # fontColor = (255, 0, 0)
    # lineType = 1
    # cv2.putText(img_ori, 'img_ori', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    #
    # img_  = np.pad(img, [[0, 40], [0, 0],[0,0]], 'constant')
    # canvas_rigid  = np.pad(canvas_rigid, [[0, 40], [0, 0],[0,0]], 'constant')
    # canvas_def  = np.pad(canvas_def, [[0, 40], [0, 0],[0,0]], 'constant')
    #
    # canvas_rigid = 255 *canvas_rigid / canvas_rigid.max()
    # canvas_def = 255 * canvas_def / canvas_def.max()
    canvas_rigid = canvas_rigid.astype(np.uint8)
    canvas_def = canvas_def.astype(np.uint8)

    ret = np.hstack([img_ori, img, canvas_rigid, canvas_def])
    return ret



def show2dLandmarks(image, proj2d):
    proj_2d_transpose = np.array(proj2d)
    proj_2d_transpose = proj_2d_transpose.T
    door1_2d = proj_2d_transpose[door1_idx, :].astype(np.int32)
    door2_2d = proj_2d_transpose[door2_idx, :].astype(np.int32)
    door3_2d = (np.array(proj_2d_transpose)[door3_idx, :]).astype(np.int32)
    door4_2d = (np.array(proj_2d_transpose)[door4_idx, :]).astype(np.int32)
    trunk_2d = (np.array(proj_2d_transpose)[trunk_idx, :]).astype(np.int32)

    cv2.polylines(image, [door1_2d], True, (125, 255, 255))
    cv2.polylines(image, [door2_2d], True, (125, 255, 255))
    cv2.polylines(image, [door3_2d], True, (125, 255, 255))
    cv2.polylines(image, [door4_2d], True, (125, 255, 255))
    cv2.polylines(image, [trunk_2d], True, (255, 255, 255))



    if FLAG_54_to_36:
            proj2d = np.array(proj2d)[:, matching_idx]
            for idx in range(18):
                cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 0, 255), -1)

            for idx in range(18, 36):
                cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 255, 0), -1)

            for idx in range(15):
                cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])),
                         (int(proj2d[0][idx + 1]), int(proj2d[1][idx + 1])),
                         (255, 0, 0), 2)
            cv2.line(image, (int(proj2d[0][15]), int(proj2d[1][15])), (int(proj2d[0][0]), int(proj2d[1][0])),
                     (255, 0, 0),
                     2)

            for idx in range(18, 33):
                cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])),
                         (int(proj2d[0][idx + 1]), int(proj2d[1][idx + 1])),
                         (255, 0, 0), 2)
                cv2.line(image, (int(proj2d[0][33]), int(proj2d[1][33])), (int(proj2d[0][18]), int(proj2d[1][18])),
                         (255, 0, 0),
                         2)

            for idx in range(8):
                cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])),
                         (int(proj2d[0][idx + 18]), int(proj2d[1][idx + 18])), (255, 0, 0), 2)

            x18 = [2 * proj2d[0][16] - proj2d[0][13], 2 * proj2d[1][16] - proj2d[1][13]]
            x19 = [2 * proj2d[0][16] - proj2d[0][14], 2 * proj2d[1][16] - proj2d[1][14]]
            x20 = [2 * proj2d[0][17] - proj2d[0][9], 2 * proj2d[1][17] - proj2d[1][9]]
            x21 = [2 * proj2d[0][17] - proj2d[0][10], 2 * proj2d[1][17] - proj2d[1][10]]
            cv2.line(image, (int(proj2d[0][15]), int(proj2d[1][15])), (int(x18[0]), int(x18[1])), (255, 0, 0), 2)
            cv2.line(image, (int(proj2d[0][12]), int(proj2d[1][12])), (int(x19[0]), int(x19[1])), (255, 0, 0), 2)
            cv2.line(image, (int(x19[0]), int(x19[1])), (int(x18[0]), int(x18[1])), (255, 0, 0), 2)
            cv2.line(image, (int(proj2d[0][11]), int(proj2d[1][11])), (int(x20[0]), int(x20[1])), (255, 0, 0), 2)
            cv2.line(image, (int(proj2d[0][8]), int(proj2d[1][8])), (int(x21[0]), int(x21[1])), (255, 0, 0), 2)
            cv2.line(image, (int(x20[0]), int(x20[1])), (int(x21[0]), int(x21[1])), (255, 0, 0), 2)

            x22 = [2 * proj2d[0][16 + 18] - proj2d[0][13 + 18], 2 * proj2d[1][16 + 18] - proj2d[1][13 + 18]]
            x23 = [2 * proj2d[0][16 + 18] - proj2d[0][14 + 18], 2 * proj2d[1][16 + 18] - proj2d[1][14 + 18]]
            x24 = [2 * proj2d[0][17 + 18] - proj2d[0][9 + 18], 2 * proj2d[1][17 + 18] - proj2d[1][9 + 18]]
            x25 = [2 * proj2d[0][17 + 18] - proj2d[0][10 + 18], 2 * proj2d[1][17 + 18] - proj2d[1][10 + 18]]
            cv2.line(image, (int(proj2d[0][15 + 18]), int(proj2d[1][15 + 18])), (int(x22[0]), int(x22[1])), (255, 0, 0),
                     2)
            cv2.line(image, (int(proj2d[0][12 + 18]), int(proj2d[1][12 + 18])), (int(x23[0]), int(x23[1])), (255, 0, 0),
                     2)
            cv2.line(image, (int(x22[0]), int(x22[1])), (int(x23[0]), int(x23[1])), (255, 0, 0), 2)
            cv2.line(image, (int(proj2d[0][11 + 18]), int(proj2d[1][11 + 18])), (int(x24[0]), int(x24[1])), (255, 0, 0),
                     2)
            cv2.line(image, (int(proj2d[0][8 + 18]), int(proj2d[1][8 + 18])), (int(x25[0]), int(x25[1])), (255, 0, 0),
                     2)
            cv2.line(image, (int(x24[0]), int(x24[1])), (int(x25[0]), int(x25[1])), (255, 0, 0), 2)




    else:
        # print(proj2d)
        # print(proj2d)
        # proj2d = proj2d[::-1, :] # fix order issue i.e. (x, y) <-> (j, i)
        # print("sanity check", proj2d.shape)
        for idx in range(15):
            cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 0, 255), -1)
        for idx in range(15, 30):
            cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 255, 0), -1)
        for idx in range(30, 41):
            cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 255, 255), -1)
        for idx in range(41, 54):
            cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (255, 255, 0), -1)




        #
        # for idx in range(18, 36):
        #     cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 255, 0), -1)
        #
        # for idx in range(15):
        #     cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), (int(proj2d[0][idx + 1]), int(proj2d[1][idx + 1])),
        #              (255, 0, 0), 2)
        # cv2.line(image, (int(proj2d[0][15]), int(proj2d[1][15])), (int(proj2d[0][0]), int(proj2d[1][0])), (255, 0, 0),
        #          2)
        #
        # for idx in range(18, 33):
        #     cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), (int(proj2d[0][idx + 1]), int(proj2d[1][idx + 1])),
        #              (255, 0, 0), 2)
        #     cv2.line(image, (int(proj2d[0][33]), int(proj2d[1][33])), (int(proj2d[0][18]), int(proj2d[1][18])), (255, 0, 0),
        #              2)
        #
        # for idx in range(8):
        #     cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])),
        #              (int(proj2d[0][idx + 18]), int(proj2d[1][idx + 18])), (255, 0, 0), 2)
        #
        # x18 = [2 * proj2d[0][16] - proj2d[0][13], 2 * proj2d[1][16] - proj2d[1][13]]
        # x19 = [2 * proj2d[0][16] - proj2d[0][14], 2 * proj2d[1][16] - proj2d[1][14]]
        # x20 = [2 * proj2d[0][17] - proj2d[0][9], 2 * proj2d[1][17] - proj2d[1][9]]
        # x21 = [2 * proj2d[0][17] - proj2d[0][10], 2 * proj2d[1][17] - proj2d[1][10]]
        # cv2.line(image, (int(proj2d[0][15]), int(proj2d[1][15])), (int(x18[0]), int(x18[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(proj2d[0][12]), int(proj2d[1][12])), (int(x19[0]), int(x19[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(x19[0]), int(x19[1])), (int(x18[0]), int(x18[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(proj2d[0][11]), int(proj2d[1][11])), (int(x20[0]), int(x20[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(proj2d[0][8]), int(proj2d[1][8])), (int(x21[0]), int(x21[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(x20[0]), int(x20[1])), (int(x21[0]), int(x21[1])), (255, 0, 0), 2)
        #
        # x22 = [2 * proj2d[0][16 + 18] - proj2d[0][13 + 18], 2 * proj2d[1][16 + 18] - proj2d[1][13 + 18]]
        # x23 = [2 * proj2d[0][16 + 18] - proj2d[0][14 + 18], 2 * proj2d[1][16 + 18] - proj2d[1][14 + 18]]
        # x24 = [2 * proj2d[0][17 + 18] - proj2d[0][9 + 18], 2 * proj2d[1][17 + 18] - proj2d[1][9 + 18]]
        # x25 = [2 * proj2d[0][17 + 18] - proj2d[0][10 + 18], 2 * proj2d[1][17 + 18] - proj2d[1][10 + 18]]
        # cv2.line(image, (int(proj2d[0][15 + 18]), int(proj2d[1][15 + 18])), (int(x22[0]), int(x22[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(proj2d[0][12 + 18]), int(proj2d[1][12 + 18])), (int(x23[0]), int(x23[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(x22[0]), int(x22[1])), (int(x23[0]), int(x23[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(proj2d[0][11 + 18]), int(proj2d[1][11 + 18])), (int(x24[0]), int(x24[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(proj2d[0][8 + 18]), int(proj2d[1][8 + 18])), (int(x25[0]), int(x25[1])), (255, 0, 0), 2)
        # cv2.line(image, (int(x24[0]), int(x24[1])), (int(x25[0]), int(x25[1])), (255, 0, 0), 2)
    #

def read_single_image(fqueue, dim):
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)
    basics = tf.parse_single_example(value, features={
        'idx': tf.FixedLenFeature([], tf.int64),
        'bbx': tf.FixedLenFeature([4], tf.float32),
        'vimg': tf.FixedLenFeature([], tf.string)})

    image = basics['vimg']

    image = tf.decode_raw(image, tf.uint8)
    image.set_shape([3 * dim * dim])
    image = tf.reshape(image, [dim, dim, 3])
    # resize to 256 x 256
    # image = tf.image.resize_images(image, size=[256, 256])
    # image = tf.cast(image, dtype=tf.uint8)
    # print(image)
    idx = tf.cast(basics['idx'], tf.int64)
    # bbx = basics['bbx'] * 256 / 64
    bbx = basics['bbx']

    # print('idx', idx)
    # print('bbx  ', bbx) # Tensor("ParseSingleExample/Squeeze_bbx:0", shape=(4,), dtype=float32)

    return image, idx, bbx


def create_bb_pip(tfr, nepoch, sbatch, mean, shuffle=True):
    tf_mean = tf.constant(mean, dtype=tf.float32)

    tf_mean = tf.reshape(tf_mean, [1, 1, 1, 3])

    fqueue = tf.train.string_input_producer([tfr], num_epochs=nepoch * 10)
    image, idx, bbx = read_single_image(fqueue, 64)

    data = tf.train.batch([image, idx, bbx], batch_size=sbatch,
                          num_threads=1, capacity=sbatch * 3)

    # preprocess input images
    data[0] = preprocess(data[0], tf_mean)
    # print("check data", data)
    return data




def evaluate(input_tfr, model_dir, mean, in_dir, out_dir, png_list):
    """Evaluate Multi-View Network for a epoch."""
    tf.logging.set_verbosity(tf.logging.FATAL)

    # maximum epochs
    total_num = len(png_list)
    sbatch = 16
    niters = int(total_num / sbatch) + 1
    diff_on_last = int(total_num - sbatch * (niters - 1))

    # set config file
    config = tf.ConfigProto(log_device_placement=False)

    with tf.Graph().as_default():
        sys.stderr.write("Building Network ... \n")
        images, idx, bbxs = create_bb_pip(input_tfr, 10, sbatch, mean, shuffle=False)
        # print("check images", images)

        # inference model.
        # pred_key = sk_net.infer_key(images, 36 * 2, tp=False)
        # pred_2d, pred_3d = sk_net.infer_23d(images, 36 * 2, 36 * 3, tp=False)
        # print("check shape - pred2d", pred_2d.shape) # 100, 72


        # pred_2d, pred_3d, _ = sk_net.infer_os(images, 36, tp=False)
        out_dim = 54
        # pred_keys_hm = hg._graph_hourglass(input=images,  outDim=out_dim, tiny=False, modif=False,
        #                                    is_training=False)
        # pred_2d = ut._hm2kp_batch(pred_keys_hm)

        hg_input, prep_3d = sk_net.modified_hg_preprocessing_with_3d_info_v2(images, 54 * 3, reuse_=False, tp=False)
        vars_avg = tf.train.ExponentialMovingAverage(0.9)
        vars_to_restore = vars_avg.variables_to_restore()
        model_saver_3d_prep = tf.train.Saver(vars_to_restore)  # when you write the model_saver matters... it will restore up to this point

        r3 = tf.image.resize_nearest_neighbor(hg_input, size=[64, 64]) # shape=(16, 64, 64, 256), dtype=float32)
        pred_keys_hm = hg._graph_hourglass_modified_v1(input=r3, outDim=out_dim, tiny=False, modif=False, is_training=False)  # shape=(16, 4, 64, 64, 36), dtype=float32)
        pred_keys_hm_last = pred_keys_hm[:, -1, :, :]
        model_saver_2d_pred = tf.train.Saver()
        pred_2d = ut._hm2kp_batch(pred_keys_hm)



        # pred_sigmoid = tf.nn.sigmoid(pred_keys_hm[:, self.params['nstacks']-1]) # play around this thing.... #TODO: implement sigmoid - thresh - pred - re-assess uncertain keypoints

        # vars_avg = tf.train.ExponentialMovingAverage(0.9)
        # vars_to_restore = vars_avg.variables_to_restore()
        # saver = tf.train.Saver(vars_to_restore)
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session(config=config) as sess:
            sys.stderr.write("Initializing ... \n")
            # initialize graph
            sess.run(init_op)

            # ckpt = tf.train.get_checkpoint_state(model_dir)
            # if ckpt and ckpt.model_checkpoint_path:
            #     mcp = ckpt.model_checkpoint_path
            #     mcp = osp.join(model_dir, mcp.split('/')[-1])
            #     print('Loading Model File %s' % mcp)
            # #     saver.restore(sess, mcp)
            # else:
            #     print('No checkpoint file found')
            #     return
            # print('restoring 3d preprocessing')
            # model_saver_3d_prep.restore(sess, 'L23d_non_iso/single_key-144000')
            # print("Successfully restored - 3d")

            print('restoring 2d prediction')
            model_saver_2d_pred.restore(sess, '/home/tliao4/tf_car_keypoint_def-9-28/src/log_hg_def_54kp_v2bugfixed_oct15/model/def_23d_4s_hg_23d_v2_r0.1_hgdropout0.5_wd0.9_v2bugfixed_oct_15-120000')
            print('Successfully restored - 2d')


            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sys.stderr.write("Start Testing\n")
            count = 0
            for i in xrange(niters):
                print('[%s]: Parsing batch %d/%d...' % (datetime.datetime.now(), i + 1, niters))
                # img_idx_pool, key2d, key3d, bbx_pool = sess.run([idx, pred_2d, pred_3d, bbxs])
                # img_idx_pool, key2d, bbx_pool = sess.run([idx, pred_2d, bbxs]) # pred_keys_hm_last
                img_idx_pool, key2d, hm_last, bbx_pool = sess.run([idx, pred_2d,pred_keys_hm_last, bbxs])
                # print("key2d sanity check", key2d[0])
                # change key2d shape from (16, 36, 2) to (16, 72) note and change i, j - > x, y
                key2d = key2d[:, :, ::-1]
                key2d = key2d.flatten()
                key2d = key2d.reshape((16, 54*2))


                if i == niters - 1: # last batch yayyy
                    img_idx_pool = img_idx_pool[:diff_on_last]
                    key2d = key2d[:diff_on_last]
                    # key3d = key3d[:diff_on_last]

                # for img_idx, k2d, k3d, bb in zip(img_idx_pool, key2d, key3d, bbx_pool):
                for img_idx, k2d, hm, bb in zip(img_idx_pool, key2d, hm_last, bbx_pool):

                    img_path = png_list[img_idx]
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    cur_out_path = img_path.replace(in_dir, out_dir)
                    if cur_out_path.endswith('.png'):
                        cur_out_path = cur_out_path.replace('.png', '')
                    elif cur_out_path.endswith('.jpg'):
                        cur_out_path = cur_out_path.replace('.jpg', '')

                    k2d_tmp = np.reshape(k2d, (54, 2))
                    k2d_tmp = k2d_tmp * 1.0
                    # print(type(k2d_tmp))
                    # k2d_tmp *= 64

                    # *4 : 256 / 64
                    # k2d_tmp = k2d_tmp * 4.0 # no need... plain2d: 256*256*3 input, here: maintain 64*64*3 image input
                    # k2d_tmp = k2d
                    # print("bb-sanity check", bb)
                    # print('sanity check', k2d_tmp)
                    k2d_tmp[:, 0] = 1.0*(k2d_tmp[:, 0] - bb[0]) / (bb[2] - bb[0])
                    k2d_tmp[:, 1] = 1.0*(k2d_tmp[:, 1] - bb[1]) / (bb[3] - bb[1])
                    k2d = k2d_tmp.flatten()
                    # print('k2d sanity check', k2d)
                    # cur_flen = len(cur_out_path.split('/')[-1])
                    # if osp.exists(cur_out_path[:-cur_flen]) is False:
                    #   os.makedirs(cur_out_path[:-cur_flen])
                    cur_dir = os.path.dirname(cur_out_path)
                    if not os.path.exists(cur_dir):
                        os.makedirs(cur_dir)

                    np.savetxt(cur_out_path + '_2d.txt', k2d)
                    # np.savetxt(cur_out_path + '_3d.txt', k3d)

                    proj2d = np.transpose(np.reshape(k2d, (54, 2)))
                    # print('image shape:', img.shape)
                    proj2d[0] *= img.shape[1]
                    proj2d[1] *= img.shape[0]
                    img_ori = img.copy()
                    show2dLandmarks(img, proj2d)
                    img_plus_combinedHm = show_img_hms(img, img_ori, hm)

                    # cv2.imwrite(cur_out_path + '_2d.jpg', img)
                    cv2.imwrite(cur_out_path + '_hmOri_combined.jpg', img_plus_combinedHm)
                    count += 1

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=5)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def noniso_warp(img, dim, method=cv2.INTER_CUBIC):
    h = img.shape[0]
    w = img.shape[1]

    rt = float(dim) / max(h, w)
    new_h = int(h * rt) if w > h else dim
    new_w = int(w * rt) if h > w else dim

    itype = img.dtype
    pimg = cv2.resize(img, (new_w, new_h), interpolation=method)
    if w > h:
        pad_w = dim
        pad_h_up = int((dim - new_h) / 2)
        pad_h_down = dim - new_h - pad_h_up
        if len(img.shape) == 3:
            pad_up = np.ones((pad_h_up, pad_w, 3), dtype=itype) * 128
            pad_down = np.ones((pad_h_down, pad_w, 3), dtype=itype) * 128
        else:
            pad_up = np.zeros((pad_h_up, pad_w), dtype=itype)
            pad_down = np.zeros((pad_h_down, pad_w), dtype=itype)

        pimg = np.concatenate((pad_up, pimg, pad_down), axis=0)
        bbx = [0, pad_h_up, dim, dim - pad_h_down]
    else:
        pad_h = dim
        pad_w_left = int((dim - new_w) / 2)
        pad_w_right = dim - new_w - pad_w_left
        if len(img.shape) == 3:
            pad_left = np.ones((pad_h, pad_w_left, 3), dtype=itype) * 128
            pad_right = np.ones((pad_h, pad_w_right, 3), dtype=itype) * 128
        else:
            pad_left = np.zeros((pad_h, pad_w_left), dtype=itype)
            pad_right = np.zeros((pad_h, pad_w_right), dtype=itype)
        pimg = np.concatenate((pad_left, pimg, pad_right), axis=1)
        bbx = [pad_w_left, 0, dim - pad_w_right, dim]

    return pimg, bbx


def process_noniso_img(tuple_):
    filename = tuple_[0]
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    dim = 64

    pimg, bbx = noniso_warp(img, dim)
    # change RGB to BGR
    pimg = pimg[:, :, ::-1]
    return pimg, tuple_[1], bbx


def gen_tfrecords(input_dir, out_filename):
    batch_N = 500
    thread_num = 6
    p = Pool(thread_num)

    writer = tf.python_io.TFRecordWriter(out_filename)
    png_list = []
    out_list = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                img_path = os.path.join(root, filename)
                png_list.append(img_path)
    # png_list: list of images for test..
    N = len(png_list)
    for ix in xrange(N):
        if ix % batch_N == 0:
            print('[%s]: Generating tfrecord %d/%d...' % (datetime.datetime.now(), ix + 1, N))
            batch_data = [(png_list[k + ix], k + ix)
                          for k in range(min(batch_N, N - ix))]

            batch_datums = p.map(process_noniso_img, batch_data)
            # print("sanity check - batch_data", batch_data) [(img_path, idx)]
        iso_img = batch_datums[ix % batch_N][0]
        index = batch_datums[ix % batch_N][1]
        bbx = batch_datums[ix % batch_N][2]
        # print('bbx', bbx)

        cur_feature = {}
        cur_feature['vimg'] = _bytes_feature(iso_img.tobytes())
        cur_feature['idx'] = _int64_feature(int(index))
        cur_feature['bbx'] = _float_feature(bbx)

        example = tf.train.Example(features=tf.train.Features(feature=cur_feature))
        writer.write(example.SerializeToString())

    writer.close()
    return png_list


def main(FLAGS):
    assert tf.gfile.Exists(FLAGS.in_dir)
    model_dir = osp.join(FLAGS.model_dir, 'L23d_pmc', 'model')
    assert tf.gfile.Exists(FLAGS.model_dir)

    mean = [128, 128, 128]

    out_filename = '/tmp/val_raw.tfrecords'
    png_list = gen_tfrecords(FLAGS.in_dir, out_filename)

    if tf.gfile.Exists(FLAGS.out_dir) is False:
        tf.gfile.MakeDirs(FLAGS.out_dir)

    # import ipdb; ipdb.set_trace(context=21)
    evaluate(out_filename, model_dir, mean, FLAGS.in_dir, FLAGS.out_dir, png_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/home/tliao4/Desktop/',
        # default=''
        help='Directory of output training and L23d_pmc files'
    )
    parser.add_argument(
        '--in_dir',
        type=str,
        #default='/home/tliao4/Desktop/tf_car_keypoint_from_8GPU/demo/train_car_full_mini',
        # default='/home/tliao4/Desktop/val_mini',
        # default='/home/tliao4/Desktop/high_res_cars',
        # default='/home/tliao4/Desktop/one_test',
        #default='/home/tliao4/Desktop/multi_car_miniset', # miniset of multi car case
        # default = '/media/tliao4/671073B1329C337D/activity_crop/validate/vehicle_turning_left/VIRAT_S_000007.mp4_61/car',
        # default='/media/tliao4/671073B1329C337D1/benchmark_seq/input',
        # default='/work_12t/tliao4/def_data_single_2nd_rand_bkgd_54_val',
        default='/home/tliao4/tf_car_keypoint_def-9-28/src/temp_singleImg_sanity',
        help='Directory of input directory'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        #default='/home/tliao4/Desktop/tf_car_keypoint_from_8GPU/demo/output_full_mini_190ksteps',
        # default='/home/tliao4/Desktop/new_tf_car_keypoint_from_8GPU/tf_car_keypoint/demo/high_res_car_test_hg_plain2d_190k',
        # default='/home/tliao4/Desktop/temp_/DEF_PRETRAIN_RES',
        # default='temp_delete_validation_HM',
        # default='temp_delete_sanity_check_hm',
        default='temp_delete_single_img_sanity',

        help='Directory of output files'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)




