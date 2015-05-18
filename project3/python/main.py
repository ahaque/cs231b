

import sys
import time
import os.path

import numpy as np

from scipy.io import loadmat


################################################################
# BEGIN REQUIRED INPUT PARAMETERS

# ML_DIR contains matlab matrix files and caffe model
ML_DIR = "../ml"

# END REQUIRED INPUT PARAMETERS
################################################################

def main():

	data = {}
	data['train'] = readMatrixData('train')
	data['test'] = readMatrixData('test')

	print data['train']['gt']['2008_007640.jpg']
	print data['train']['ssearch']['2008_007640.jpg']



'''
Reads the Matlab matrix data into a nice dictionary format

Input: 'train' or 'test'
Output: A dictionary data, see examples below
	data['train']['gt']['2008_007640.jpg'] = tuple( class_labels, gt_bboxes )
	data['train']['gt']['2008_007640.jpg'] = tuple( [[2]] , [[ 90,  85, 500, 366]] )
	data['train']['ssearch']['2008_007640.jpg'] = n x 4 matrix of region proposals (bboxes)
'''
def readMatrixData(phase):
	# Read the matrix files
	raw_ims = {}
	raw_ims.update(loadmat(os.path.join(ML_DIR, phase + "_ims.mat")))

	raw_ssearch = {}
	raw_ssearch.update(loadmat(os.path.join(ML_DIR, "ssearch_" + phase + ".mat")))

	# Populate our new, cleaner dictionary
	data = {}
	data['gt'] = {}
	data['ssearch'] = {}

	for i in xrange(raw_ims['images'].shape[1]):
		filename, labels, bboxes = raw_ims['images'][0,i]
		data['gt'][filename[0]] = (labels, bboxes)
		data['ssearch'][filename[0]] = raw_ssearch['ssearch_boxes'][0,i]

	return data

if __name__ == '__main__':
	main()