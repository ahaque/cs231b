
import cv2
import sys
import time
import os.path

import numpy as np

from Util import *
from scipy.io import loadmat


################################################################
# BEGIN REQUIRED INPUT PARAMETERS

# For all DIRs, the trailing slash does not matter
# ML_DIR contains matlab matrix files and caffe model
ML_DIR = "../ml"

# IMG_DIR contains all images
IMG_DIR = "../images"

# Input size of the CNN input image (after cropping)
CNN_INPUT_SIZE = 227

# CNN Batch size. Depends on the hardware memory
CNN_BATCH_SIZE = 200

# END REQUIRED INPUT PARAMETERS
################################################################

def main():

	data = {}
	data["train"] = readMatrixData("train")
	data["test"] = readMatrixData("test")

	extractRegionFeats(data, "train")

################################################################
# extractRegionFeats(phase)
#   Extract region features from training set (extract_region_feats.m)
# 
# Input: "train" or "test"
# Output: None
#
def extractRegionFeats(data, phase):

	# Extract the image mean and compute the cropped mean
	img_mean = loadmat(os.path.join(ML_DIR, "ilsvrc_2012_mean.mat"))["image_mean"]
	offset = np.floor((img_mean.shape[0] - CNN_INPUT_SIZE)/2) + 1
	img_mean = img_mean[offset:offset+CNN_INPUT_SIZE, offset:offset+CNN_INPUT_SIZE, :]

	for i, image_name in enumerate(data[phase]["gt"].keys()):
		# Read image, compute number of batches
		img = cv2.imread(os.path.join(IMG_DIR, image_name))
		regions = data[phase]["ssearch"][image_name]

		num_regions = data[phase]["ssearch"][image_name].shape[0]
		num_batches = int(np.ceil(1.0 * num_regions / CNN_BATCH_SIZE))
		
		# Extract batches from original image
		img_batch = np.zeros((CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3, CNN_BATCH_SIZE))
		for b in xrange(num_batches):
			# Index into the regions array
			idx = b * CNN_BATCH_SIZE + i
			# TODO: Add context around the region
			print regions[idx]
			contexted = img[regions[idx][0]:regions[idx][2], regions[idx][1]:regions[idx][3], :]
			print contexted.shape
			cv2.imshow("Region", contexted)
			cv2.waitKey(0)
			#resized_region = cv2.resize(image, (100, 50)) 
		break


################################################################
# readMatrixData()
#	Reads the Matlab matrix data into a nice dictionary format
#
# Input: "train" or "test"
# Output: A dictionary data, see examples below
#	data["train"]["gt"]["2008_007640.jpg"] = tuple( class_labels, gt_bboxes )
#	data["train"]["gt"]["2008_007640.jpg"] = tuple( [[2]] , [[ 90,  85, 500, 366]] )
#	data["train"]["ssearch"]["2008_007640.jpg"] = n x 4 matrix of region proposals (bboxes)
def readMatrixData(phase):
	# Read the matrix files
	raw_ims = {}
	raw_ims.update(loadmat(os.path.join(ML_DIR, phase + "_ims.mat")))

	raw_ssearch = {}
	raw_ssearch.update(loadmat(os.path.join(ML_DIR, "ssearch_" + phase + ".mat")))

	# Populate our new, cleaner dictionary
	data = {}
	data["gt"] = {}
	data["ssearch"] = {}

	for i in xrange(raw_ims["images"].shape[1]):
		filename, labels, bboxes = raw_ims["images"][0,i]
		data["gt"][filename[0]] = (labels, bboxes)
		data["ssearch"][filename[0]] = raw_ssearch["ssearch_boxes"][0,i]

	return data


################################################################
# randomColor()
#   Generates a random color from our color list
def randomColor():
	return COLORS[np.random.randint(0, len(COLORS))]


################################################################
# displayImageWithBboxes(image_name, bboxes)
#   Displays an image with several bounding boxes
#
# Input: image_name (string)
#		 bboxes (matrix, where each row corresponds to a bbox)
# Output: None
#	
#	displayImageWithBboxes(image_name, data["train"]["gt"][image_name][1])
#
def displayImageWithBboxes(image_name, bboxes):
	img = cv2.imread(os.path.join(IMG_DIR, image_name))

	for bbox in bboxes:
		cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), randomColor(), thickness=2)

	cv2.imshow("Image", img)
	cv2.waitKey(0)


if __name__ == "__main__":
	main()