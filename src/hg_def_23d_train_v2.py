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
from utils import preprocess_norm

from utils import ndata_tfrecords
from utils import optimizer
from utils import vis_keypoints
from tran3D import quaternion_matrix
from tran3D import polar_to_axis
from tran3D import quaternion_about_axis

from scipy.linalg import logm
import cv2

import hg_utils as ut
import net2d_hg_modified_v1 as hg

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
nStack=4
# v2: doesn't fix pre-trained value(or has no pre-train value at all)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('train_def_rand_bkgd_v2data_9-7.L23d_pmc')  #@
logger.addHandler(file_handler)

def read_one_datum(fqueue, dim, key_num=54, hm_dim=64, nStack=nStack):
  reader = tf.TFRecordReader()
  key, value = reader.read(fqueue)
  basics = tf.parse_single_example(value, features={
    'key2d': tf.FixedLenFeature([key_num * 2], tf.float32),
    'key3d': tf.FixedLenFeature([key_num * 3], tf.float32),
    'image': tf.FixedLenFeature([], tf.string)})

  image = basics['image']
  image = tf.decode_raw(image, tf.uint8)
  image.set_shape([3 * dim * dim])
  image = tf.reshape(image, [dim, dim, 3])

  k2d = basics['key2d']
  k3d = basics['key3d']

  # convert k2d to HMs(4stacks for sHG)
  key_resh = tf.reshape(k2d, shape=(key_num, 2))
  key_resh *= hm_dim  # to match hm_dimension

  key_hm = ut._tf_generate_hm(hm_dim, hm_dim, key_resh)  # tensor, (64, 64, 52), dtype=tf.float32
  temp = [key_hm for _ in range(nStack)]
  key_hm = tf.stack(temp, axis=0)

  return image, key_hm, k3d


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
        # dim = int(tokens[-1].split('.')[0][1:])
        dim = 64
        tf_mean = tf.constant(mean, dtype=tf.float32)
        tf_mean = tf.reshape(tf_mean, [1, 1, 1, 3])

        fqueue = tf.train.string_input_producer([tfr], num_epochs=nepoch)
        image, gt_key, gt_3d = read_one_datum(fqueue, dim)

        if shuffle:
            data = tf.train.shuffle_batch([image, gt_key, gt_3d], batch_size=cur_ebs,
                                          num_threads=12, capacity=sbatch * 6, min_after_dequeue=cur_ebs * 3)
        else:
            data = tf.train.batch([image, gt_key, gt_3d], batch_size=cur_ebs,
                                  num_threads=12, capacity=cur_ebs * 5)

        # preprocess input images

        # print("data0]", data[0])
        data[0] = preprocess(data[0], tf_mean) #
        # data[0] = preprocess_norm(data[0]) #

        if ix == 0:
            for j in range(len(data)):
                data_pool.append([data[j]])
        else:
            for j in range(len(data)):
                data_pool[j].append(data[j])

    combined_data = []
    for dd in data_pool:
        combined_data.append(tf.concat(dd, axis=0))
    # print("sanity check : combined_data", combined_data)

    return combined_data


# def vis2d_one_output(image, pred_key, gt_key):
#     image += 128
#     image.astype(np.uint8)
#     dim = image.shape[0]  # height == width by default
#
#     left = np.copy(image)
#     right = np.copy(image)
#
#     pk = np.reshape(pred_key, (36, 2)) * dim
#     gk = np.reshape(gt_key, (36, 2)) * dim
#     pk.astype(np.int32)
#     gk.astype(np.int32)
#
#     for pp in pk:
#         cv2.circle(left, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     for pp in gk:
#         cv2.circle(right, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     final_out = np.hstack([left, right])
#
#     outputs = np.zeros((1, final_out.shape[0], final_out.shape[1], 3), dtype=np.uint8)
#     outputs[0] = final_out
#
#     return outputs

# def vis2d_one_output_hm(image, pred_key, gt_key):
#     # de-norm to 0~255
#     image *= 255
#     # resize to 64*64, to match hm size
#     image = tf.image.resize_images(image, size=(64, 64))
#     image.astype(np.uint8)
#     # print(image)
#     dim = image.shape[0]  # height == width by default,  64 by default
#     # print("dim", dim)
#
#     left = np.copy(image)
#     right = np.copy(image)
#
#     pk = pred_key
#     gk = gt_key
#     pk.astype(np.int32)
#     gk.astype(np.int32)
#
#     for pp in pk:
#         cv2.circle(left, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     for pp in gk:
#         cv2.circle(right, (pp[0], pp[1]), 2, (0, 0, 255), -1)
#
#     final_out = np.hstack([left, right])
#
#     outputs = np.zeros((1, final_out.shape[0], final_out.shape[1], 3), dtype=np.uint8)
#     outputs[0] = final_out
#
#     return outputs

def vis2d_one_output_hm(image, pred_key, gt_key):
    return image


# def add_bb_summary(images, pred_keys, gt_keys, name_prefix, max_out=1):
#     n = images.get_shape().as_list()[0]
#
#     for i in range(np.min([n, max_out])):
#         pred = tf.gather(pred_keys, i)
#         gt = tf.gather(gt_keys, i)
#         image = tf.gather(images, i)
#
#         result = tf.py_func(vis2d_one_output_hm, [image, pred, gt], tf.uint8)
#
#         tf.summary.image(name_prefix + '_result_' + str(i), result, 1)

def add_bb_summary_hm(images, pred_keys_hm, gt_keys_hm, name_prefix, max_out=1):
    n = images.get_shape().as_list()[0]

    for i in range(np.min([n, max_out])):
        # collect from batch
        pred_key_hm = tf.gather(pred_keys_hm, i)
        gt_key_hm = tf.gather(gt_keys_hm, i)
        image = tf.gather(images, i)

        # convert hm to 2d keypoints
        pred_key = ut._hm2kp(pred_key_hm)
        gt_key = ut._hm2kp(gt_key_hm)
        # print("pred_key", pred_key)
        # print("gt_key", gt_key)

        result = tf.py_func(vis2d_one_output_hm, [image, pred_key, gt_key], tf.uint8)

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
    val_freq = 4000 #@ 8-15 currently unknown bug.... getting stucked during validation
    model_save_freq = 25000 #@
    tf.logging.set_verbosity(tf.logging.ERROR)

    # maximum epochs
    total_iters = 250000 # smaller test... #@
    # total_iters = 200000 # batch_size 100
    # total_iters = 1250000 # batchsize = 16
    # lrs = [0.01, 0.001, 0.0001]

    # steps = [int(total_iters * 0.5), int(total_iters * 0.4), int(total_iters * 0.1)]

    # set config file
    config = tf.ConfigProto(log_device_placement=False)
    with tf.Graph().as_default():
        sys.stderr.write("Building Network ... \n")
        # global_step = tf.contrib.framework.get_or_create_global_step() # THIS IS REALLY MESSEED UP WHEN LOADING MODELS..

        # images, gt_key = create_bb_pip(input_tfr_pool, 1000, sbatch, mean, shuffle=True)
        images, gt_keys_hm, gt_3d = create_bb_pip(input_tfr_pool, None, sbatch, mean, shuffle=True)

        # print(gt_key.get_shape().as_list()) # key_hm: [B, nStack, h, w, #key_points], i.e. [16, 4, 64, 64, 36]
        # inference model
        #
        # key_dim = gt_key.get_shape().as_list()[1]
        # pred_key = sk_net.infer_key(images, key_dim, tp=True)

        # out_dim = gt_keys_hm.get_shape().as_list()[-1]
        out_dim = 54
        # test_out = sk_net.modified_key23d_64_breaking(images)
        # pred_keys_hm = hg._graph_hourglass(input=images, dropout_rate=0.2, outDim=out_dim, tiny=False, modif=False, is_training=True)

        #preparation with 3d intermediate supervision...
        hg_input, pred_3d = sk_net.modified_hg_preprocessing_with_3d_info_v2(images,  54 * 3, reuse_=False, tp=True) # fix prep part  #@

        vars_avg = tf.train.ExponentialMovingAverage(0.9)
        vars_to_restore = vars_avg.variables_to_restore()
        # print(vars_to_restore)
        model_saver = tf.train.Saver(vars_to_restore) # when you write the model_saver matters... it will restore up to this point


        r3 = tf.image.resize_nearest_neighbor(hg_input, size=[64, 64]) # shape=(16, 64, 64, 256), dtype=float32)
        pred_keys_hm = hg._graph_hourglass_modified_v1(input=r3, dropout_rate=0.5, outDim=out_dim, nStack=nStack, tiny=False, modif=False, is_training=True) # shape=(16, 4, 64, 64, 36), dtype=float32) #@

        # Calculate loss
        # total_loss, data_loss = sk_net.L2_loss_key(pred_key, gt_key, weight_decay=wd)
        # train_op, _ = optimizer(total_loss, global_step, lrs, steps)

        k2d_hm_loss = ut._bce_loss(logits=pred_keys_hm, gtMaps=gt_keys_hm, name='ce_loss', weighted=False) # 4 stacks / 4.... not dividing by 4, just to keep it consistent with what i've done before
        k3d_loss = 0.1*tf.nn.l2_loss(pred_3d - gt_3d) #@
        total_loss = tf.add_n([k2d_hm_loss, k3d_loss]) #@
        # total_loss = k2d_hm_loss
        init_learning_rate = 2.5e-4 # to be deteremined
        # # exp decay: 125000 / 2000 = 625decays,   0.992658^625 ~=0.01, 0.99^625 ~= 0.00187
        model_saver_23d_v2 = tf.train.Saver(max_to_keep=10)

        train_step = tf.Variable(0, name='train_steps', trainable=False) # you need to move train_step here in order to avoid being loaded

        # lr_hg = tf.train.exponential_decay(init_learning_rate, global_step=train_step, decay_rate=0.96, decay_steps=2000, staircase=True, name="learning_rate") #@
        lr_hg = tf.train.exponential_decay(init_learning_rate, global_step=train_step, decay_rate=0.96, decay_steps=2000, staircase=True, name="learning_rate")

        #

        rmsprop_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_hg)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        ## weight decay version
        # with tf.control_dependencies(update_ops): #@ modified 8-11 with EMA
        #     grads = rmsprop_optimizer.compute_gradients(total_loss)
        #     apply_grad_op = rmsprop_optimizer.apply_gradients(grads, global_step=train_step)
        #     # train_op_hg = rmsprop_optimizer.minimize(total_loss, train_step)
        # var_avg_ema = tf.train.ExponentialMovingAverage(0.9, train_step)
        # var_avg_ema_op = var_avg_ema.apply(tf.trainable_variables())
        #
        # with tf.control_dependencies([apply_grad_op, var_avg_ema_op]): # EMA enabled
        #     train_op_hg = tf.no_op(name='Train')

        ## normal
        #def compute_grad():
        #    grads = rmsprop_optimizer.compute_gradients(total_loss)
         #   apply_grad_op = rmsprop_optimizer.apply_gradients(grads, global_step=train_step)
         #   return apply_grad_op
        def compute_grad():
            apply_grad_op = rmsprop_optimizer.minimize(total_loss, global_step=train_step)
            return apply_grad_op

        def skip_grad():
            grads = rmsprop_optimizer.compute_gradients(total_loss)
            grads_zero = tf.zeros_like(grads)
            apply_grad_op_skipped = rmsprop_optimizer.apply_gradients(grads_zero, global_step=train_step)
            return apply_grad_op_skipped

        def no_op():
            return tf.no_op()


        with tf.control_dependencies(update_ops):
            p = tf.less(total_loss, tf.constant((100.0), dtype=tf.float32))
            train_op_hg = tf.cond(p, compute_grad, no_op)

            # train_op_hg = tf.cond(p, compute_grad, skip_grad)

        sys.stderr.write("Train Graph Done ... \n")
        #
        # # add_bb_summary_hm(images, pred_keys_hm, gt_keys_hm, 'train', max_out=3) # TODO: enable it
        if val_tfr_pool:
            val_pool = []
            val_iters = []
            accur_pool = []
            for ix, val_tfr in enumerate(val_tfr_pool):
                # total_val_num = ndata_tfrecords(val_tfr)
                total_val_num = 89
                total_val_iters = int(float(total_val_num) / sbatch) # num of batches, iters / epoch
                val_iters.append(total_val_iters)
                # val_images, val_gt_key = create_bb_pip([val_tfr],
                #                                        1000, sbatch, mean, shuffle=False)
                val_images, val_gt_keys_hm, val_gt_3d = create_bb_pip([val_tfr],
                                                                      None, sbatch, mean, shuffle=False)

                val_r3, _ = sk_net.modified_hg_preprocessing_with_3d_info_v2(val_images, 54 * 3, reuse_=True, tp=False)

                val_r3 = tf.image.resize_nearest_neighbor(val_r3, size=[64, 64])  # shape=(16, 64, 64, 256), dtype=float32)
                # val_pred_key = sk_net.infer_key(val_images, key_dim, tp=False, reuse_=True)

                # val_pred_key = sk_net.infer_key(val_images, key_dim, tp=False, reuse_=True)
                val_pred_keys_hm = hg._graph_hourglass_modified_v1(input=val_r3, outDim=out_dim,is_training=False, tiny=False, nStack=nStack, modif=False, reuse=True)

                # _, val_data_loss = sk_net.L2_loss_key(val_pred_key, val_gt_key, None)
                val_train_loss_hg = ut._bce_loss(logits=val_pred_keys_hm, gtMaps=val_gt_keys_hm, name="val_ce_loss")
                # val_pool.append(val_data_loss)
                val_accur = ut._accuracy_computation(output=val_pred_keys_hm, gtMaps=val_gt_keys_hm, nStack=nStack,
                                                       batchSize=16)

                val_pool.append(val_train_loss_hg)
                accur_pool.append(val_accur)
        #
        #         # add_bb_summary(val_images, val_pred_key, val_gt_key, 'val_c' + str(ix), max_out=3)
        #         # add_bb_summary_hm(val_images, val_pred_keys_hm, val_gt_keys_hm, 'val_c' + str(ix), max_out=3) # TODO: argmax pred, draw
            sys.stderr.write("Validation Graph Done ... \n")
        #
        # # merge all summaries
        # # merged = tf.summary.merge_all()
        merged = tf.constant(0)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sys.stderr.write("Initializing ... \n")
            # initialize graph
            sess.run(init_op)

            #########################################################################################################
            ###### disable/enable pre-trained weight loading
            # print('restoring')
            # # model_saver.restore(sess, '/home/tliao4/Desktop/new_tf_car_keypoint/tf_car_keypoint/src/log_hg_s4_256/L23d_pmc/model/single_key_4s_hg-85000') # 85k steps
            # model_saver.restore(sess, 'L23d_non_iso/single_key-144000') #@
            # print("Successfully restored 3d preprocessing")
            #########################################################################################################

            # print('restoring -v3') # v3 refers to: no-pretrain vgg part(preprocessing, then fix it and train hourglass only)
            # model_saver_23d_v1.restore(sess, '/home/tliao4/Desktop/temp_/v2_0.1_no_pretrain_model/single_key_4s_hg_23d_v2-200000')
            # print('restored successfully - v3')
            print('Load pre-trained model (trained on non-rand bkgd')
            model_saver_23d_v2.restore(sess, 'log_hg_def_s4_256_23d_v2_r0.1_hgdropout0.5_datav2/model/def_23d_4s_hg_23d_v2_r0.1_hgdropout0.5_datav2-160000') #@
            print('Succesfully loaded pre-trained model')


            print('initial-sanity check')
            print('init_step: ', sess.run(train_step))
            print('init_lr: ', sess.run(lr_hg))
            # check
            # initialize the queue threads to start to shovel data
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # for i in 4*range(10):
            #     print("check-img/n", (sess.run(images[1, 60+i, 60+i,:])))
            # print(images)

            model_prefix = os.path.join(out_dir, 'def_randbkgd_54kp') #@
            timer = 0
            timer_count = 0
            sys.stderr.write("Start Training --- OUT DIM: %d\n" % (out_dim))
            logger.info("Start Training --- OUT DIM: %d\n" % (out_dim))
            for iter in xrange(total_iters):
                ts = time.time()
                if iter > 0 and iter % log_freq == 0:
                    # print('lr', sess.run(lr_hg))
                    # print('global_step', sess.run(train_step))
                    # key_loss, _, summary = sess.run([data_loss, train_op, merged])
                    key_loss, _, summary = sess.run([total_loss,train_op_hg, merged])

                    # summary_writer.add_summary(summary, i)
                    # summary_writer.flush()

                    sys.stderr.write('Training %d (%fs) --- Combined 23d loss: %f\n'
                                     % (iter, timer / timer_count, key_loss))
                    logger.info(('Training %d (%fs) --- Combined23d loss: %f\n'
                                     % (iter, timer / timer_count, key_loss)))
                    timer = 0
                    timer_count = 0
                else:
                    # sess.run([train_op])
                    sess.run([train_op_hg, pred_3d])
                    timer += time.time() - ts
                    timer_count += 1

                if val_tfr and iter > 0 and iter % val_freq == 0:
                    cur_lr = lr_hg.eval()
                    print("lr: ", cur_lr)
                    logger.info('lr: {}'.format(cur_lr))

                    sys.stderr.write('Validation %d\n' % iter)
                    logger.info(('Validation %d\n' % iter))
                    # loss
                    for cid, v_dl in enumerate(val_pool):
                        val_key_loss = eval_one_epoch(sess, v_dl, val_iters[cid])
                        sys.stderr.write('Class %d --- Key HM CE Loss: %f\n' % (cid, val_key_loss))
                        logger.info('Class %d --- Key HM CE Loss: %f\n' % (cid, val_key_loss))
                    #
                    for cid, accur in enumerate(accur_pool):
                        rec=[]
                        for i in range(val_iters[cid]):
                             acc = sess.run(accur) # acc: [(float)*36]
                             rec.append(acc)
                        rec = np.array(rec)
                        rec = np.mean(rec, axis=0)
                        avg_accur = np.mean(rec)
                        temp_dict = {}
                        for k in range(54):
                            temp_dict['kp_'+str(iter)] = rec[k]
                        sys.stderr.write('Class %d -- Avg Accuracy : %f\n' %(cid, avg_accur))
                        sys.stderr.write('Classs {} -- All Accuracy:\n{}\n'.format(cid, rec))
                        logger.info('Class %d -- Avg Accuracy : %f\n' %(cid, avg_accur))
                        logger.info('Class {} -- All Accuracy:\n {}\n'.format(cid, rec))

                if iter > 0 and iter % model_save_freq == 0:
                    model_saver_23d_v2.save(sess, model_prefix, global_step=iter)

            model_saver_23d_v2.save(sess, model_prefix, global_step=iter)

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

    log_dir = osp.join(FLAGS.out_dir, 'L23d_pmc')
    if tf.gfile.Exists(log_dir) is False:
        tf.gfile.MakeDirs(log_dir)
    else:
        for ff in os.listdir(log_dir):
            os.unlink(osp.join(log_dir, ff))

    model_dir = osp.join(FLAGS.out_dir, 'model')
    if tf.gfile.Exists(model_dir) is False:
        tf.gfile.MakeDirs(model_dir)

    # train_files = ['syn_def_car_full_train_d64.tfrecord', 'syn_def_car_randshrinkcrop_train_d64.tfrecord',  ##@
    #                 'syn_car_randtranscrop_train_d64.tfrecord']
    #
    # val_files = ['syn_def_car_full_val_d64.tfrecord', 'syn_def_car_randshrinkcrop_val_d64.tfrecord',
    #                 'syn_car_randtranscrop_val_d64.tfrecord']
    train_files = ['def_data_2nd_rand_bkgd_64.tfrecord', 'def_data_2nd_rand_bkgd_64.tfrecord', 'def_data_2nd_rand_bkgd_64.tfrecord']
    val_files = ['def_data_2nd_rand_bkgd_val_64.tfrecord', 'def_data_2nd_rand_bkgd_val_64.tfrecord', 'def_data_2nd_rand_bkgd_val_64.tfrecord']

    train_files = [osp.join(FLAGS.input, tt) for tt in train_files]
    val_files = [osp.join(FLAGS.input, tt) for tt in val_files]

    train(train_files, val_files, model_dir, log_dir, mean, FLAGS.batch, FLAGS.wd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_dir',
        type=str,
        default='/media/tliao4/671073B1329C337D/def_car_rand_bkgd_tfr/def_data_2nd_multi_rand_bkgd_tfr/models',  #@
        help='Directory of output training and L23d_pmc files'
    )
    parser.add_argument(
        '--input',
        type=str,
        # default='/media/tliao4/671073B1329C337D/chi/syn_dataset/tfrecord/car/v1',
        default='/media/tliao4/671073B1329C337D/def_car_rand_bkgd_tfr/def_data_2nd_multi_rand_bkgd_tfr/',

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
        default=0, # not used.. hard-coded using 0.9
        help='Weight decay of the variables in network.'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='batch size.'
    )
    parser.add_argument('--debug', action='store_true', help='debug mode')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
