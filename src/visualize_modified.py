#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pdb
import cv2
import numpy as np
from math import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob

def show2dLandmarks(proj2d, image):
	proj2d = proj2d.reshape(2, 36)
	for idx in range(proj2d.shape[1]/2):
		if proj2d[1][idx] >= image.shape[0] or proj2d[0][idx] >= image.shape[1] or proj2d[0][idx] < 0 or proj2d[1][idx] < 0:
			continue
		
		cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0,0,255), -1)

	for idx in range(proj2d.shape[1]/2,proj2d.shape[1]):
		if proj2d[1][idx] >= image.shape[0] or proj2d[0][idx] >= image.shape[1] or proj2d[0][idx] < 0 or proj2d[1][idx] < 0:
			continue
		
		cv2.circle(image, (int(proj2d[0][idx]), int(proj2d[1][idx])), 4, (0,255,0), -1)

def visualize_car(x3d, ax):
	ax.clear()
	x3d = x3d.reshape(36, 3).transpose()[[0, 2, 1], :]
	ax.scatter(x3d[0, 0:16], x3d[1, 0:16], x3d[2, 0:16], c='r', marker='.', s=40)
	ax.scatter(x3d[0, 16:18], x3d[1, 16:18], x3d[2, 16:18], c='r', marker='.', s=800)
	ax.scatter(x3d[0, 18:34], x3d[1, 18:34], x3d[2, 18:34], c='g', marker='.', s=40)
	ax.scatter(x3d[0, 34:36], x3d[1, 34:36], x3d[2, 34:36], c='g', marker='.', s=800)
	ax.plot(x3d[0, 0:16], x3d[1, 0:16], x3d[2, 0:16], c='r')
	ax.plot([x3d[0,15], x3d[0,0]], [x3d[1,15],x3d[1,0]], [x3d[2,15],x3d[2,0]], c='r')
	ax.plot(x3d[0, 18:34], x3d[1, 18:34], x3d[2, 18:34], c='g')
	ax.plot([x3d[0,33], x3d[0,18]], [x3d[1,33],x3d[1,18]], [x3d[2,33],x3d[2,18]], c='g')
	for i in range(8):
		ax.plot([x3d[0,i], x3d[0,i+18]], [x3d[1,i],x3d[1,i+18]], [x3d[2,i],x3d[2,i+18]], c=(0,0,0))
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		for direction in (-1, 1):
			for point in np.diag(direction * np.array([0.5,0.3,0.3])):
				ax.plot([point[0]], [point[1]], [point[2]], 'w')

	ax.view_init(elev=105, azim=270)
	plt.draw()
	plt.waitforbuttonpress()
 
def main(argv):
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--res_path', type=str, default='/home/tliao4/tliao4/demo/output_hg_os23d', help='path to val results')
	args = parser.parse_args()

	# img_filenames = [line.rstrip() for line in open(args.file, 'r')]
	img_filenames = [f[:-4] for f in glob.glob(os.path.join(args.res_path, '*.jpg'))]
	avail_3d = False

    #if args.file[-5] == '0':
	if True:
		avail_3d = True
		fig = plt.figure(1)
		ax = fig.add_subplot(111, projection='3d')

	for base_file in img_filenames:
		img = cv2.imread(base_file + '.jpg')
		proj2d = np.loadtxt(base_file+'.txt', delimiter=' ')
		# if a keypoint is not visible, its 2d (x,y) are both negative
		resized_img = cv2.resize(img, (img.shape[1]*3, img.shape[0]*3))
		show2dLandmarks(proj2d, resized_img)
		cv2.imshow('pred', resized_img)
		cv2.waitKey(100)
		
		# press any button to proceed to next data
		if avail_3d:
			model = np.loadtxt(base_file[:-3]+'_3d.txt', delimiter=' ') # N*3 matrix, N is the number of keypoints
			visualize_car(model,ax)
		else:
			cv2.waitKey()

if __name__ == '__main__':
	main(sys.argv)



