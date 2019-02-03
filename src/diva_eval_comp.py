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

import net2d_hg as hg
import hg_utils as ut
import cv2

left_input_list = []
right_input_list = []
output_list = []

# chi's result
input_dir_left = '/media/tliao4/671073B1329C337D/tk_val_chi'
# hg_3d-2d
input_dir_right = '/media/tliao4/671073B1329C337D/tk_val_hg23d_v3_0.1_105k'

out_dir_root = '/media/tliao4/671073B1329C337D/diva_eval_comp'

# left
for act_class in os.listdir(input_dir_left):
    act_path = osp.join(input_dir_left, act_class)
    for six, seq in enumerate(os.listdir(act_path)):
        seq_path = osp.join(act_path, seq)
        for fi in os.listdir(seq_path):
            # files
            if fi.endswith('.png') or fi.endswith('.jpg'):
                filename = osp.join(seq_path, fi)
                left_input_list.append(filename)
                output_list.append(osp.join(act_class, seq, fi[:-4]))
# right
for act_class in os.listdir(input_dir_right):
    act_path = osp.join(input_dir_right, act_class)
    for six, seq in enumerate(os.listdir(act_path)):
        seq_path = osp.join(act_path, seq)
        for fi in os.listdir(seq_path):
            # files
            if fi.endswith('.png') or fi.endswith('.jpg'):
                filename = osp.join(seq_path, fi)
                right_input_list.append(filename)

for left_img_path, right_img_path, out_rel_path in zip(left_input_list, right_input_list, output_list):
    # left
    left = cv2.imread(left_img_path)
    left = np.pad(left, [[0, 40], [0, 0], [0, 0]], 'constant')
    l_shape = left.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0 + 5, l_shape[0] - 20)
    fontScale = 0.5
    fontColor = (255, 0, 0)
    lineType = 1
    cv2.putText(left, 'chi_L23', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    # right
    right = cv2.imread(right_img_path)
    right = np.pad(right, [[0, 40], [0, 0], [0, 0]], 'constant')
    r_shape = right.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0 + 5, r_shape[0] - 20)
    fontScale = 0.5
    fontColor = (0, 0, 255)
    lineType = 1
    cv2.putText(right, 'v3_23d', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    mid = np.zeros((left.shape[0], 100,3))
    try:
        final_out = np.hstack([left, mid, right])
        out_path = osp.join(out_dir_root, out_rel_path)
        cur_flen = len(out_path.split('/')[-1])
        if not osp.exists(out_path[:-cur_flen]):
            os.makedirs(out_path[:-cur_flen])

        cv2.imwrite(osp.join(out_path + '_2d.jpg'), final_out)
    except:
        continue



