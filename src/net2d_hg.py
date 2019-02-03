# -*- coding: utf-8 -*-
"""
Thomas Liao  July 2018
working on stacked hourglass model on tf car keypoint project
This file contains building blocks and architecture of SHG model, alone with several utility functions.
"""
import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os


def _conv(input, filters, kernel_size=1, strides=1, pad='VALID',  name='conv', w_summary=False, p_fix=0):
    """ Spatial Convolution (CONV2D)
    Args:
        input			: Input Tensor (Data Type : NHWC)
        filters		: Number of filters (channels)
        kernel_size	: Size of kernel
        strides		: Stride
        pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
        name			: Name of the block
    Returns:
        conv			: Output Tensor (Convolved Input)
    """
    with tf.variable_scope(name):
        n = name + str(p_fix)
        # Kernel for convolution, Xavier Initialisation
        # kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
        #     [kernel_size, kernel_size, input.get_shape().as_list()[3], filters]), name=name) # fixed..... same as below, for reusability for val during training...
        xavier = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
        kernel = tf.get_variable(name=n, shape=[kernel_size, kernel_size, input.get_shape().as_list()[3], filters],
                                 initializer=xavier, dtype=tf.float32)
        conv = tf.nn.conv2d(input, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
        if w_summary:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weight'])
        return conv


def _conv_bn_relu(input, filters, kernel_size=1, strides=1, is_training=False, pad='VALID', name='conv_bn_relu', p_fix=0, w_summary=False):
    """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
    Args:
        input			: Input Tensor (Data Type : NHWC)
        filters		: Number of filters (channels)
        kernel_size	: Size of kernel
        strides		: Stride
        pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
        name			: Name of the block
    Returns:
        norm			: Output Tensor
    """
    with tf.variable_scope(name):
        n = name + str(p_fix)
        # kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
        #     [kernel_size, kernel_size, input.get_shape().as_list()[3], filters]), name=n)
        xavier = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
        kernel = tf.get_variable(name=n, shape=[kernel_size, kernel_size, input.get_shape().as_list()[3], filters], initializer=xavier, dtype=tf.float32)
        conv = tf.nn.conv2d(input, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
        norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                            is_training=is_training)
        if w_summary:
            with tf.device('/cpu:0'):
                tf.summary.histogram('weights_summary', kernel, collections=['weight'])
        return norm


def _conv_block(input, numOut, name='conv_block', p_fix=0, tiny=False, is_training=False):
    """ Convolutional Block
    Args:
        input	: Input Tensor
        numOut	: Desired output number of channel
        name	: Name of the block
    Returns:
        conv_3	: Output Tensor
    """
    n = name + str(p_fix)
    if tiny:
        with tf.variable_scope(name):

            norm = tf.contrib.layers.batch_norm(input, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=is_training)
            pad = tf.pad(norm, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')

            conv = _conv(pad, int(numOut), kernel_size=3, strides=1, pad='VALID', name=n)
            return conv
    else:
        with tf.variable_scope(name):
            with tf.variable_scope('norm_1'):
                norm_1 = tf.contrib.layers.batch_norm(input, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=is_training)
                conv_1 = _conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name=n)
            with tf.variable_scope('norm_2'):
                norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=is_training)
                pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv_2 = _conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name=name+str(p_fix+1))
            with tf.variable_scope('norm_3'):
                norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                      is_training=is_training)
                conv_3 = _conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name=name+str(p_fix+2))
            return conv_3


def _skip_layer(input, numOut, name='skip_layer', p_fix=0, weight_summary=False):
    """ Skip Layer
    Args:
        input	: Input Tensor
        numOut	: Desired output number of channel
        name	: Name of the bloc
    Returns:
        Tensor of shape (None, inputs.height, inputs.width, numOut)
    """
    with tf.variable_scope(name+str(p_fix)):
        if input.get_shape().as_list()[3] == numOut:
            return input
        else:
            conv = _conv(input, numOut, kernel_size=1, strides=1, name=name+str(p_fix), w_summary=weight_summary)
            return conv


def _residual(input, numOut, name='residual_block', modif=False, tiny=False, is_training=False, p_fix=0):
    """ Residual block - rb
    rb(residual block):
                       |--_skip_layer--------|
                       |                     +--->
                       |                     |
 ----------------------|--_conv_block--------|

    Args:
        input	: Input Tensor
        numOut	: Number of Output Features (channels)
        name	: Name of the block
    """
    with tf.variable_scope(name+str(p_fix)):
        convb = _conv_block(input, numOut, tiny=tiny, is_training=is_training, p_fix=p_fix)
        skipl = _skip_layer(input, numOut, weight_summary=False, p_fix=p_fix)
        if modif:
            # modif: apply relu to combined upper and lower path
            return tf.nn.relu(tf.add_n([convb, skipl], name='res_block'))
        else:
            return tf.add_n([convb, skipl], name='res_block')

def _hourglass(input, numOut, n=4, name='hourglass', modif=False, tiny=False, is_training=False, p_fix=0, p_fix_s=0):
    """ Single Hourglass module.
#########################################################################################################################################################################################################
Single HG module:
HG:

                (upper branch)
                                                        <h * w * n>
    <h * w * m> |----> [rb, m -> n] ----------------------------------------------------|   <h * w * n>
    ---input--->|                                                                      [+] ------out---->
                |----> /2 ---> [rb, m -> n] --->[rec*] --- > [rb, n - > n] ---> *2 -----|
                 <h/2 * w/2 * m>                   |         <h/2 * w/2 * n>       <h * w * n>
                                                   |
                (lower branch)                     |                                                                |
                                                   |
                                                   |
                      ############################################################
                      ## Recursion: if n_down_steps = 0, base case: [rb, n -> n]##
                      ##           else: HG with n_down_steps - 1               ##
                      ############################################################

#########################################################################################################################################################################################################

    Args:
        input	: Input Tensor
        n		: Number of downsampling step
        numOut	: Number of Output Features (channels)
        name	: Name of the block
    Returns:
        [batch_size, 64, 64, num_out]
    """
    # print("check point", input.get_shape().as_list())
    # p_fix: down_steps, p_fix_s: stacks
    with tf.variable_scope(name + 'down_'+str(p_fix) + 'stack_' + str(p_fix_s)):
        # Upper Branch
        up_1 = _residual(input, numOut, name='up_1', modif=modif, is_training=is_training)
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(input, [2, 2], [2, 2], padding='VALID')
        low_1 = _residual(low_, numOut, name='low_1', modif=modif, is_training=is_training)

        if n > 0:
            low_2 = _hourglass(input=low_1, n=n - 1, numOut=numOut, name='low_2', is_training=is_training, p_fix=p_fix+1, p_fix_s=p_fix_s)
        else:
            low_2 = _residual(input=low_1, numOut=numOut, name='low_2', modif=modif, is_training=is_training, p_fix=p_fix+1)

        low_3 = _residual(low_2, numOut, name='low_3', modif=modif, is_training=is_training)
        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
        if modif:
            # Use of RELU
            return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
        else:
            return tf.add_n([up_2, up_1], name='out_hg')

def _argmax(tensor):
    """ ArgMax
    Args:
        tensor	: 2D - Tensor (Height x Width : 64x64 )
    Returns:
        arg		: Tuple of max position
    """
    resh = tf.reshape(tensor, [-1])
    argmax = tf.arg_max(resh, 0)
    return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])


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
    return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
                     tf.to_float(91))


def _accur(pred, gtMap, num_image):
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


def _graph_hourglass(input, nFeat=256, nLow=4, outDim=36, dropout_rate=0.2, nStack=4, modif=False, tiny=False, w_summary=False, is_training=False, reuse=False):
    """Create the stacke hourglass network
########################################################################################################################################################################################################
########################################################################################################################################################################################################
SHG preprocessing:

                        (B, H, W, C)    (pad: h:33 w:33
                                         conv: ks: 6 stride: 2)        out_channels=128
                                         out_channels=64
                         ---input_------> (pad)conv1 ------------------------> r1 -----------------------pool1(/2)----------------> r2 -------------> r3 -----------> output of proproessing
                        (B,256,256,3)             (B,128,128,64)*                    (B, 128,128,128)              (B,64,64,128)     (B,64,64,128)       (B,64,64,256) (down from 512)

        * out_fm = 1 + (h(or w) - ks) /2,  h or w including padding

SHG complete graph:
                                                                r3
                 (pad1                    |----------------------------------------------------------------------------------------------------------------------------------------------|
                 conv1                    |                                                                                                                                              |
                 r1                       |                                                                                                                                              |
                 pool1                    |                                                                                                                                              |
     input       r2(optional)             |                                                                                                                                              |
                 r3)                  r3  |                        hg[0]                 drop[0]                 ll[0]                    out[0]                    out_[0]              |
 ------------preprocesing_part---------------> HG_Module(stage:1) ---------------->dropout------>_conv_bn_relu------------------->conv*----------------->conv----------------->----------+----------> (@BP)
  (B,256,256,3)                (B,64,64,256)                       (B,64,64,256)                               (B,64,64,256)   |        (B,64,64,outDim)         (B,64,64,256)           |
                                                                                                                               |               ll_[0]*                                   |
                                                                                                                               |_________________conv____________________________________|
conv*: if modif, _conv_bn_relu, else, _conv
ll_*: if not tiny, else just use ll.
                                       |----------------------------------------------------------------------------------------------------------------------------------------------|
                                       |                                                                                                                                              |
    # k=#stage - 1                     |                                                                                                                                              |
                                       |                                                                                                                                              |
                                       |                                                                                                                                              |
                     sum_[k]           |                          hg[stg]             drop[stg]               ll[stg]                    out[stg]                out_[stg]            |
 ------------(@BP)------------------------> HG_Module(stage:1) ---------------->dropout------>_conv_bn_relu------------------->conv*----------------->conv----------------->----------+----------> (@BP - recursion)
  (B,64,64,256)                (B,64,64,256)                       (B,64,64,256)                               (B,64,64,256)   |        (B,64,64,outDim)         (B,64,64,256)        |
                                                                                                                               |                                                      |
                                                                                                                               |               ll_[0]*                                |
                                                                                                                               |_________________conv_________________________________|
        (B,64,64,512)
#########################################################################################################################################################################################################

    Args:
        input : TF Tensor (placeholder) of shape (None, 256, 256, 3) # TODO : Create a parameter for customize size
    """
    with tf.variable_scope('sHG_model', reuse=reuse):
        with tf.variable_scope('preprocessing'):
            # Input Dim : nbImages x 256 x 256 x 3
            pad1 = tf.pad(input, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
            # Dim pad1 : nbImages x 260 x 260 x 3
            conv1 = _conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128', w_summary=w_summary, is_training=is_training)
            # Dim conv1 : nbImages x 128 x 128 x 64
            r1 = _residual(conv1, numOut=128, name='r1', modif=modif, tiny=tiny, is_training=is_training)
            # Dim pad1 : nbImages x 128 x 128 x 128
            pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
            # Dim pool1 : nbImages x 64 x 64 x 128
            if tiny:
                r3 = _residual(pool1, numOut=nFeat, name='r3', is_training=is_training)
            else:
                r2 = _residual(pool1, numOut=int(nFeat / 2), name='r2', modif=modif, tiny=tiny, is_training=is_training)
                r3 = _residual(r2, numOut=nFeat, name='r3', modif=modif, tiny=tiny, is_training=is_training)
        print("r3 sanity check", r3)
        # Storage Table
        hg = [None] * nStack
        ll = [None] * nStack
        ll_ = [None] * nStack
        drop = [None] * nStack
        out = [None] * nStack
        out_ = [None] * nStack
        sum_ = [None] * nStack
        if tiny:
            with tf.variable_scope('stacks'):
                with tf.variable_scope('stage_0'):
                    hg[0] = _hourglass(r3, n=nLow, numOut=nFeat,  name='hourglass', tiny=tiny, modif=modif, is_training=is_training, p_fix_s=0)
                    drop[0] = tf.layers.dropout(hg[0], rate=dropout_rate, training=is_training, name='dropout')
                    ll[0] = _conv_bn_relu(input=drop[0], filters=nFeat, kernel_size=1, strides=1, name='ll', is_training=is_training)
                    if modif:
                        # TEST OF BATCH RELU
                        out[0] = _conv_bn_relu(ll[0], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out', is_training=is_training)
                    else:
                        out[0] = _conv(input=ll[0], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out')
                    out_[0] = _conv(input=out[0], filters=nFeat, kernel_size=1, strides=1, pad='VALID', name='out_')
                    sum_[0] = tf.add_n([out_[0], ll[0], r3], name='merge')
                for i in range(1, nStack - 1):
                    with tf.variable_scope('stage_' + str(i)):
                        hg[i] = _hourglass(input=sum_[i - 1], n=nLow, numOut=nFeat, name='hourglass', is_training=is_training, tiny=tiny, modif=modif)
                        drop[i] = tf.layers.dropout(hg[i], rate=dropout_rate, training=is_training, name='dropout')
                        ll[i] = _conv_bn_relu(input=drop[i], filters=nFeat, kernel_size=1, strides=1, pad='VALID', name='ll', is_training=is_training)
                        if modif:
                            # TEST OF BATCH RELU
                            out[i] = _conv_bn_relu(input=ll[i], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out', is_training=is_training)
                        else:
                            out[i] = _conv(ll[i], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out')
                        out_[i] = _conv(input=out[i], filters=nFeat, kernel_size=1, strides=1, pad='VALID', name='out_')
                        sum_[i] = tf.add_n([out_[i], ll[i], sum_[i - 1]], name='merge')
                with tf.variable_scope('stage_' + str(nStack - 1)):
                    hg[nStack - 1] = _hourglass(niputs=sum_[nStack - 2],n=nLow, numOut=nFeat, name='hourglass', tiny=tiny, is_training=is_training, modif=modif)
                    drop[nStack - 1] = tf.layers.dropout(inputs=hg[nStack - 1], rate=dropout_rate,
                                                              training=is_training, name='dropout')
                    ll[nStack - 1] = _conv_bn_relu(input=drop[nStack - 1], filters=nFeat, kernel_size=1, strides=1, padding='VALID',
                                                   name='conv', is_training=is_training)
                    if modif:
                        out[nStack - 1] = _conv_bn_relu(ll[nStack - 1], outDim, 1, 1, is_training=is_training, pad='VALID',
                                                                  name='out')
                    else:
                        out[nStack - 1] = _conv(input=ll[nStack - 1], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out')
            if modif:
                return tf.nn.sigmoid(tf.stack(out, axis=1, name='stack_output'), name='final_output')
            else:
                return tf.stack(out, axis=1, name='final_output')
        else: # not-tiny version
            with tf.variable_scope('stacks'):
                with tf.variable_scope('stage_0'):
                    hg[0] = _hourglass(input=r3, n=nLow, numOut=nFeat, name='hourglass',modif=modif, tiny=tiny, is_training=is_training)
                    drop[0] = tf.layers.dropout(inputs=hg[0], rate=dropout_rate, training=is_training,
                                                name='dropout')
                    ll[0] = _conv_bn_relu(input=drop[0], filters=outDim, kernel_size=1, strides=1, is_training=is_training, pad='VALID', name='conv')
                    ll_[0] = _conv(input=ll[0], filters=nFeat, kernel_size=1, strides=1, pad='VALID', name='ll')
                    if modif:
                        # TEST OF BATCH RELU
                        out[0] = _conv_bn_relu(input=ll[0], filters=outDim, kernel_size=1, strides=1, is_training=is_training, pad='VALID', name='out')
                    else:
                        out[0] = _conv(input=ll[0], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out')
                    out_[0] = _conv(input=out[0], filters=nFeat, kernel_size=1, strides=1, pad='VALID', name='out_')
                    sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
                for i in range(1, nStack - 1):
                    with tf.variable_scope('stage_' + str(i)):
                        hg[i] = _hourglass(input=sum_[i - 1], n=nLow, numOut=nFeat, modif=modif, tiny=tiny, is_training=is_training, name='hourglass')
                        drop[i] = tf.layers.dropout(inputs=hg[i], rate=dropout_rate, training=is_training, name='dropout')
                        ll[i] = _conv_bn_relu(input=drop[i], filters=nFeat, kernel_size=1, strides=1, is_training=is_training, pad='VALID', name='conv')
                        ll_[i] = _conv(input=ll[i], filters=nFeat, kernel_size=1, strides=1, pad='VALID', name='ll')
                        if modif:
                            out[i] = _conv_bn_relu(input=ll[i], filters=outDim, kernel_size=1, strides=1, is_training=is_training, pad='VALID', name='out')
                        else:
                            out[i] = _conv(input=ll[i], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out')
                        out_[i] = _conv(input=out[i], filters=nFeat, kernel_size=1, strides=1, pad='VALID', name='out_')
                        sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')
                with tf.variable_scope('stage_' + str(nStack - 1)):
                    hg[nStack - 1] = _hourglass(input=sum_[nStack - 2], n=nLow, numOut=nFeat, name='hourglass', modif=modif, tiny=tiny, is_training=is_training)
                    drop[nStack - 1] = tf.layers.dropout(inputs=hg[nStack - 1], rate=dropout_rate, training=is_training, name='dropout')
                    ll[nStack - 1] = _conv_bn_relu(input=drop[nStack - 1], filters=nFeat, kernel_size=1, strides=1, pad='VALID', is_training=is_training, name='conv')
                    if modif:
                        out[nStack - 1] = _conv_bn_relu(input=ll[nStack - 1], filters=outDim, kernel_size=1, strides=1, is_training=is_training, pad='VALID', name='out')
                    else:
                        out[nStack - 1] = _conv(input=ll[nStack - 1], filters=outDim, kernel_size=1, strides=1, pad='VALID', name='out')
            if modif:
                return tf.nn.sigmoid(tf.stack(out, axis=1, name='stack_output'), name='final_output')
            else:
                # stack res and return, axis=1: nStack/Stage of HG model, i.e. BHWC - > B(new_axis = N_stack)HWC
                return tf.stack(out, axis=1, name='final_output')



