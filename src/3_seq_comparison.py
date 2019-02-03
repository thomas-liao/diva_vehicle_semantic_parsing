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
img_list_left_left = []
img_list_left = []
img_list_right = []
img_list_right_right = []

name_list = []



for root, _, files in os.walk('/home/tliao4/Desktop/hg_23d_models_v1/chi_23d_result/output'):
    for filename in files:
        img_path = os.path.join(root, filename)
        img_list_left_left.append(img_path)

for root, _, files in os.walk('/home/tliao4/Desktop/hg_23d_models_v1/hg_23d_seq_120k'):
    for filename in files:

        img_path = os.path.join(root, filename)
        img_list_left.append(img_path)

for root, _, files in os.walk('/home/tliao4/Desktop/hg_23d_models_v1/hg_23d_seq_200k'):
    for filename in files:

        img_path = os.path.join(root, filename)
        img_list_right.append(img_path)
        name_list.append(filename)

for root, _, files in os.walk('/home/tliao4/Desktop/temp_/aa'):
    for filename in files:

        img_path = os.path.join(root, filename)
        img_list_right_right.append(img_path)

for i in range(len(img_list_right)):
    leftleft = cv2.imread(img_list_left_left[i])
    leftleft = np.pad(leftleft, [[0, 40], [0, 0], [0, 0]], 'constant')

    ll_shape = leftleft.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0 + 5, ll_shape[0] - 20)
    fontScale = 0.3
    fontColor = (255, 0, 0)
    lineType = 1
    cv2.putText(leftleft, 'Chi_23d', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


    left = cv2.imread(img_list_left[i])
    left = np.pad(left,[[0, 40], [0, 0],[0,0]], 'constant')


    l_shape = left.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0+5,l_shape[0]-20)
    fontScale = 0.3
    fontColor = (255, 0, 0)
    lineType = 1
    cv2.putText(left, 'v1_120k', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


    right = cv2.imread(img_list_right[i])
    right = np.pad(right,[[0, 40], [0, 0],[0,0]], 'constant')

    r_shape = right.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0 + 5, r_shape[0] - 20)
    fontScale = 0.3
    fontColor = (0, 0, 255)
    lineType = 1
    cv2.putText(right, 'v1_200k', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


    rightright = cv2.imread(img_list_right_right[i])
    rightright = np.pad(rightright, [[0, 40], [0, 0], [0, 0]], 'constant')

    rr_shape = rightright.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0 + 5, r_shape[0] - 20)
    fontScale = 0.3
    fontColor = (0, 0, 255)
    lineType = 1
    cv2.putText(rightright, 'v3_100k', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    mid = np.zeros((left.shape[0], 100,3))
    final_out = np.hstack([leftleft, mid, left, mid, right, mid, rightright])

    # write text





    cv2.imwrite('/home/tliao4/Desktop/chi_vs_v1_v3/{}'.format(name_list[i]), final_out)



# img = cv2.imread('/home/tliao4/Desktop/tt/03646_2d.jpg')
# img2 = cv2.imread('/home/tliao4/Desktop/tt/03646_2d.jpg')
# final_out = np.hstack([img, img2])
#
# cv2.imshow('img', final_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




