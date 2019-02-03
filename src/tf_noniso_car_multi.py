#!/usr/bin/python

import os
import sys
import tensorflow as tf
import argparse
import numpy as np
import pdb
import cv2
import random
import datetime
from multiprocessing import Pool

tran_ratio = 0.1

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def path2label(path):
	parts = os.path.basename(path).split('_')
	azimuth = int(parts[2][1:]) % 360
	elevation = int(parts[3][1:]) % 360
	tilt = int(parts[4][1:]) % 360
	return (azimuth, elevation, tilt)

def rescale3d(x3d):
	norm = np.sqrt(np.sum(np.square(np.array([x3d[0,0] - x3d[0,13], x3d[1,0] - x3d[1,13], x3d[2,0] - x3d[2,13]]))))
	return x3d / norm

def normalize3d(x3d):
	c3d = np.copy(x3d)
	c3d[0,:] -= np.mean(x3d[0,:])
	c3d[1,:] -= np.mean(x3d[1,:])
	c3d[2,:] -= np.mean(x3d[2,:])

	c3d = rescale3d(c3d)
	return c3d

def noniso_warp(img, key2d, dim, method=cv2.INTER_CUBIC):
  h = img.shape[0]
  w = img.shape[1]

  rt = float(dim) / max(h, w)
  new_h = int(h * rt) if w > h else dim
  new_w = int(w * rt) if h > w else dim

  itype = img.dtype
  pimg = cv2.resize(img, (new_w, new_h), interpolation=method)

  new_k2d = np.copy(key2d)
  new_k2d[:, 0] *= new_w
  new_k2d[:, 1] *= new_h

  if w > h:
    pad_w = dim
    pad_h_up = int((dim - new_h) / 2)
    pad_h_down = dim - new_h - pad_h_up
    new_k2d[:, 1] += pad_h_down # again, i think it is a bug..  pad_h_up

    if len(img.shape) == 3:
      pad_up = np.ones((pad_h_up, pad_w, 3), dtype=itype) * 128
      pad_down = np.ones((pad_h_down, pad_w, 3), dtype=itype) * 128
    else:
      pad_up = np.zeros((pad_h_up, pad_w), dtype=itype)
      pad_down = np.zeros((pad_h_down, pad_w), dtype=itype)

    pimg = np.concatenate((pad_up, pimg, pad_down), axis=0)
  else:
    pad_h = dim
    pad_w_left = int((dim - new_w) / 2)
    pad_w_right = dim - new_w - pad_w_left
    new_k2d[:, 0] += pad_w_left
    if len(img.shape) == 3:
      pad_left = np.ones((pad_h, pad_w_left, 3), dtype=itype) * 128
      pad_right = np.ones((pad_h, pad_w_right, 3), dtype=itype) * 128
    else:
      pad_left = np.zeros((pad_h, pad_w_left), dtype=itype)
      pad_right = np.zeros((pad_h, pad_w_right), dtype=itype)
    pimg = np.concatenate((pad_left, pimg, pad_right), axis=1 )

  new_k2d /= dim
  pimg = pimg[:,:,::-1]

  return pimg, new_k2d

def combine2datum(tuple_):
	data, idx = tuple_
	base_file = data[0]
	dim = data[1]
	method = data[2]

	im = cv2.imread(base_file+'.png', -1)
	im = np.array(im)
	# convert gray to color
	if len(np.shape(im)) == 2: # gray image
		im = im[:,:,np.newaxis]
		im = np.tile(im, [1,1,3])
	
	angle_labels = path2label(base_file)
	
	# crop image 
	bbx = np.loadtxt(base_file+'.bbx', delimiter=' ')
	bbx = [int(round(bb)) for bb in bbx]

	ori_height = bbx[3] - bbx[1] + 1
	ori_width = bbx[2] - bbx[0] + 1


	# TL - add / del / modified data pipeline - 5 data augmentation methods
	if method == 0:
		left = bbx[0]
		bottom = bbx[1]
		right = bbx[2]
		top = bbx[3]

	elif method == 1:
		# shrinking crop
		left = int(bbx[0] + ori_width*random.uniform(0,tran_ratio))
		bottom = int(bbx[1] + ori_height*random.uniform(0,tran_ratio))
		right = int(bbx[2] - ori_width*random.uniform(0,tran_ratio))
		top = int(bbx[3] - ori_height*random.uniform(0,tran_ratio))
		if bottom < bbx[1] or top > bbx[3] or left < bbx[0] or right > bbx[2]:
					print 'ERROR1!'

	elif method == 2:
		new_height = int(ori_height - ori_height * tran_ratio)
		new_width  = int(ori_width - ori_width * tran_ratio)
		left = random.randint(int(bbx[0]), int(bbx[2]-new_width))
		bottom = random.randint(int(bbx[1]), int(bbx[3]-new_height))
		right = int(left + new_width)
		top = int(bottom + new_height)
	
	elif method == 3:
		switch = random.uniform(0,1.0)
		if switch > 0.75:   #left-bottom
			bottom = bbx[1]
			right = bbx[2]
			left = bbx[0]+int(round(abs(ori_width * random.uniform(0.05, tran_ratio))))
			top = int(round(bbx[3] - abs(ori_height * random.uniform(0.05, tran_ratio))))
		elif switch > 0.5: #left
			bottom = bbx[1]
			right = bbx[2]
			left = bbx[0] + int(round(abs(ori_width * random.uniform(0.05, tran_ratio))))
			top = bbx[3]
		elif switch > 0.25: #right-bottom
			bottom = bbx[1]
			right = int(round(bbx[2] - abs(ori_width * random.uniform(0.05, tran_ratio))))
			left = bbx[0]
			top = int(round(bbx[3] - abs(ori_height * random.uniform(0.05, tran_ratio))))
		else:   	#right
			bottom = bbx[1]
			right = int(round(bbx[2] - abs(ori_width * random.uniform(0.05, tran_ratio))))
			left = bbx[0]
			top = bbx[3]

	cur_height = top - bottom + 1
	cur_width = right - left + 1
	
	im = im[bottom:top, left:right]
	output_vec = []
	
	key3d = np.loadtxt(base_file+'.3d', delimiter=' ')
	key3d = np.transpose(key3d)	

	datum2d = np.loadtxt(base_file+'.2d', delimiter=' ')
	new_datum2d = np.copy(datum2d[0:2, :])
	new_datum2d[0, :] = (new_datum2d[0, :] - left) / cur_width
	new_datum2d[1, :] = (new_datum2d[1, :] - bottom) / cur_height
	key2d = np.transpose(new_datum2d)
	
	cur_im, key2d = noniso_warp(im, key2d, dim)

	keyocc = np.loadtxt(base_file+'.occflag', delimiter=' ')
	for ix in range(new_datum2d.shape[1]):
		if new_datum2d[0, ix] < 0 or new_datum2d[0, ix] > 1 or new_datum2d[1, ix] < 0 or new_datum2d[1, ix] > 1:
			keyocc[ix] = 1 

	yaw = angle_labels[0]
	if yaw < 0:
		yaw += 360
	elif yaw >= 360:
		yaw -= 360
	
	return cur_im, key3d, key2d, keyocc, yaw 

def write_one_data(writer, data):
  cur_feature = {}
  cur_feature['image'] = _bytes_feature(data[0].tobytes())
  cur_feature['key3d'] = _float_feature(data[1].flatten())
  cur_feature['key2d'] = _float_feature(data[2].flatten())
  cur_feature['occ'] = _float_feature(data[3])
  cur_feature['yaw'] = _float_feature([data[4]])

  example = tf.train.Example(features=tf.train.Features(feature=cur_feature))
  writer.write(example.SerializeToString())
  return

def generate_tfrecord(filelist, out_filename, im_size, method, debug):
  batch_N = 1000
  thread_num = 24 
  p = Pool(thread_num)

  count = 0
  N = len(filelist)
  writer = tf.python_io.TFRecordWriter(out_filename)

  for ix in xrange(N):
    if ix % batch_N == 0:
      print('[%s]: %d/%d' % (datetime.datetime.now(), ix, N))
      batch_data = []
      for k in range(min(batch_N, N - ix)):
        idx = k + ix
        batch_data.append(((filelist[idx], im_size, method), idx))

      if not debug:
        batch_datums = p.map(combine2datum, batch_data)

    if not debug:
      cb = batch_datums[ix % batch_N]
    else:
      cb = combine2datum(batch_data[ix % batch_N])

    if len(cb) > 0:
      write_one_data(writer, cb)
      count += 1

  print("%d data samples saved" % count)
  writer.close()

def getOccRatio(filename):
	tokens = filename.split('_')
	occ_ratio = int(tokens[-2][3:]) / 100.0
	return occ_ratio

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument("--type", type=str, default='train', 
			help="training or validation data?")
	parser.add_argument("--size", type=int, default=64, 
			help="output size of images")
	parser.add_argument("--debug", action='store_true', 
			help="Debug flag")
	
	args = parser.parse_args()

	root_path = '/home-1/cli53@jhu.edu/data/chi/syn_dataset'
	input_path = os.path.join(root_path, 'crop/car_diva/car_multi', args.type)
	output_path = os.path.join(root_path, 'tfrecord/v1')

	if os.path.exists(output_path) == False:
		os.makedirs(output_path)
	
	out_filename = os.path.join(output_path, 'syn_car_multi_' + args.type + '_d' + str(args.size) + '.tfrecord')
	
	filelist = []
	for dir_ in os.listdir(input_path):
		for file_ in os.listdir(os.path.join(input_path, dir_)):
			if file_.endswith('png'):
				occ_ratio = getOccRatio(file_)
				if occ_ratio < 0.97 and occ_ratio > 0.4:
					filelist.append(os.path.join(input_path, dir_, file_[:-4]))
	
	print("Data Num: ", len(filelist))
	#0: no truncation 1: random in crop 2: random translate in crop 3: boundary cropping 4: pascal cropping
	method = 0 
	generate_tfrecord(filelist, out_filename, args.size, method, args.debug) 

if __name__ == '__main__':
	main(sys.argv)



