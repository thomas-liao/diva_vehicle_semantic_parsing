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

img_list_left = []
img_list_right = []
name_list = []
for root, _, files in os.walk('/media/tliao4/671073B1329C337D1/DEF_ALL/def_pretrain_36kp_res_250ksteps'):
    for filename in files:
        img_path = os.path.join(root, filename)
        img_list_left.append(img_path)

for root, _, files in os.walk('/media/tliao4/671073B1329C337D1/DEF_ALL/def_pretrain_36kp_res_wd09_250ksteps'):
    for filename in files:
        img_path = os.path.join(root, filename)
        img_list_right.append(img_path)
        name_list.append(filename)
print("sanity check")
print(len(img_list_right))
print(len(img_list_left))


for i in range(len(img_list_right)):
    left = cv2.imread(img_list_left[i])

    left = np.pad(left, [[0, 40], [0, 0],[0,0]], 'constant')

    l_shape = left.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0+5,l_shape[0]-20)
    fontScale = 0.5
    fontColor = (255, 0, 0)
    lineType = 1
    cv2.putText(left, 'no_wd', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)


    right = cv2.imread(img_list_right[i])
    right = np.pad(right,[[0, 40], [0, 0],[0,0]], 'constant')

    r_shape = right.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (0 + 5, r_shape[0] - 20)
    fontScale = 0.5
    fontColor = (0, 0, 255)
    lineType = 1
    cv2.putText(right, 'wd_0.9', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)



    mid = np.zeros((left.shape[0], 100,3))
    final_out = np.hstack([left, mid, right])

    # write text





    cv2.imwrite('/media/tliao4/671073B1329C337D1/DEF_ALL/comp_wdvsnowd_250k/{}'.format(name_list[i]), final_out)



# img = cv2.imread('/home/tliao4/Desktop/tt/03646_2d.jpg')
# img2 = cv2.imread('/home/tliao4/Desktop/tt/03646_2d.jpg')
# final_out = np.hstack([img, img2])
#
# cv2.imshow('img', final_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




