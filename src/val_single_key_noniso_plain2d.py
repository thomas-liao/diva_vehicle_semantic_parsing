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


def compute_xy_one_angle(one_bin, one_delta):
    pivot = np.argmax(one_bin)
    bin_size = 2.0 / one_bin.size
    return (pivot + 0.5) * bin_size + one_delta[pivot] - 1.0


def show2dLandmarks(image, proj2d):
    for idx in range(18):
        cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 0, 255), -1)

    for idx in range(18, 36):
        cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0, 255, 0), -1)

    for idx in range(15):
        cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), (int(proj2d[0][idx + 1]), int(proj2d[1][idx + 1])),
                 (255, 0, 0), 2)
        cv2.line(image, (int(proj2d[0][15]), int(proj2d[1][15])), (int(proj2d[0][0]), int(proj2d[1][0])), (255, 0, 0),
                 2)

    for idx in range(18, 33):
        cv2.line(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), (int(proj2d[0][idx + 1]), int(proj2d[1][idx + 1])),
                 (255, 0, 0), 2)
        cv2.line(image, (int(proj2d[0][33]), int(proj2d[1][33])), (int(proj2d[0][18]), int(proj2d[1][18])), (255, 0, 0),
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
    cv2.line(image, (int(proj2d[0][15 + 18]), int(proj2d[1][15 + 18])), (int(x22[0]), int(x22[1])), (255, 0, 0), 2)
    cv2.line(image, (int(proj2d[0][12 + 18]), int(proj2d[1][12 + 18])), (int(x23[0]), int(x23[1])), (255, 0, 0), 2)
    cv2.line(image, (int(x22[0]), int(x22[1])), (int(x23[0]), int(x23[1])), (255, 0, 0), 2)
    cv2.line(image, (int(proj2d[0][11 + 18]), int(proj2d[1][11 + 18])), (int(x24[0]), int(x24[1])), (255, 0, 0), 2)
    cv2.line(image, (int(proj2d[0][8 + 18]), int(proj2d[1][8 + 18])), (int(x25[0]), int(x25[1])), (255, 0, 0), 2)
    cv2.line(image, (int(x24[0]), int(x24[1])), (int(x25[0]), int(x25[1])), (255, 0, 0), 2)


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

    idx = tf.cast(basics['idx'], tf.int64)
    bbx = basics['bbx']

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
    return data


def evaluate(input_tfr, model_dir, mean, in_dir, out_dir, png_list):
    """Evaluate Multi-View Network for a epoch."""
    tf.logging.set_verbosity(tf.logging.FATAL)

    # maximum epochs
    total_num = len(png_list)
    sbatch = 100
    niters = int(total_num / sbatch) + 1
    diff_on_last = int(total_num - sbatch * (niters - 1))

    # set config file
    config = tf.ConfigProto(log_device_placement=False)

    with tf.Graph().as_default():
        sys.stderr.write("Building Network ... \n")
        images, idx, bbxs = create_bb_pip(input_tfr, 10, sbatch, mean, shuffle=False)

        # inference model.
        pred_2d = sk_net.infer_key(images, 36 * 2, tp=False)
        # pred_2d, pred_3d = sk_net.infer_23d(images, 36 * 2, 36 * 3, tp=False)
        # pred_2d, pred_3d, _ = sk_net.infer_os(images, 36, tp=False)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        vars_avg = tf.train.ExponentialMovingAverage(0.9)
        vars_to_restore = vars_avg.variables_to_restore()
        saver = tf.train.Saver(vars_to_restore)

        with tf.Session(config=config) as sess:
            sys.stderr.write("Initializing ... \n")
            # initialize graph
            sess.run(init_op)

            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                mcp = ckpt.model_checkpoint_path
                mcp = osp.join(model_dir, mcp.split('/')[-1])
                print('Loading Model File %s' % mcp)
                saver.restore(sess, mcp)
            else:
                print('No checkpoint file found')
                return

            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sys.stderr.write("Start Testing\n")
            count = 0
            for i in xrange(niters):
                print('[%s]: Parsing batch %d/%d...' % (datetime.datetime.now(), i + 1, niters))
                # img_idx_pool, key2d, key3d, bbx_pool = sess.run([idx, pred_2d, pred_3d, bbxs])
                img_idx_pool, key2d, bbx_pool = sess.run([idx, pred_2d, bbxs])

                if i == niters - 1:
                    img_idx_pool = img_idx_pool[:diff_on_last]
                    key2d = key2d[:diff_on_last]
                    # key3d = key3d[:diff_on_last]

                # for img_idx, k2d, k3d, bb in zip(img_idx_pool, key2d, key3d, bbx_pool):
                for img_idx, k2d, bb in zip(img_idx_pool, key2d, bbx_pool):

                    img_path = png_list[img_idx]
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    cur_out_path = img_path.replace(in_dir, out_dir)
                    if cur_out_path.endswith('.png'):
                        cur_out_path = cur_out_path.replace('.png', '')
                    elif cur_out_path.endswith('.jpg'):
                        cur_out_path = cur_out_path.replace('.jpg', '')

                    k2d_tmp = np.reshape(k2d, (36, 2))
                    k2d_tmp *= 64
                    k2d_tmp[:, 0] = (k2d_tmp[:, 0] - bb[0]) / (bb[2] - bb[0])
                    k2d_tmp[:, 1] = (k2d_tmp[:, 1] - bb[1]) / (bb[3] - bb[1])
                    k2d = k2d_tmp.flatten()

                    # cur_flen = len(cur_out_path.split('/')[-1])
                    # if osp.exists(cur_out_path[:-cur_flen]) is False:
                    #   os.makedirs(cur_out_path[:-cur_flen])
                    cur_dir = os.path.dirname(cur_out_path)
                    if not os.path.exists(cur_dir):
                        os.makedirs(cur_dir)

                    np.savetxt(cur_out_path + '_2d.txt', k2d)
                    # np.savetxt(cur_out_path + '_3d.txt', k3d)

                    proj2d = np.transpose(np.reshape(k2d, (36, 2)))
                    proj2d[0] *= img.shape[1]
                    proj2d[1] *= img.shape[0]
                    show2dLandmarks(img, proj2d)

                    cv2.imwrite(cur_out_path + '_2d.jpg', img)
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

    N = len(png_list)
    for ix in xrange(N):
        if ix % batch_N == 0:
            print('[%s]: Generating tfrecord %d/%d...' % (datetime.datetime.now(), ix + 1, N))
            batch_data = [(png_list[k + ix], k + ix)
                          for k in range(min(batch_N, N - ix))]

            batch_datums = p.map(process_noniso_img, batch_data)

        iso_img = batch_datums[ix % batch_N][0]
        index = batch_datums[ix % batch_N][1]
        bbx = batch_datums[ix % batch_N][2]

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
        default='/media/tliao4/671073B1329C337D/all_models/model/diva/L2_plain2d',
        help='Directory of output training and L23d_pmc files'
    )
    parser.add_argument(
        '--in_dir',
        type=str,
        # default='/home/tliao4/Desktop/new_tf_car_keypoint_from_8GPU/tf_car_keypoint/demo/input',

        # sanity check
        #default='/home/tliao4/Desktop/tf_car_keypoint_from_8GPU/demo/train_car_full_mini',
        # default='/home/tliao4/Desktop/val_mini',
        default='/home/tliao4/Desktop/test_car_multi_mini',
        help='Directory of input directory'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='/home/tliao4/Desktop/aaaa_multi_mini',
        help='Directory of output files'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)