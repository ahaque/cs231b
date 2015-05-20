
import cv2
import sys
import time
import os.path
import caffe

import numpy as np

from Util import *
from scipy.io import loadmat


# BEGIN REQUIRED INPUT PARAMETERS

# For all DIRs, the trailing slash does not matter
# ML_DIR contains matlab matrix files and caffe model
ML_DIR = "../ml"
################################################################

# Trailing slash or no slash doesn't matter
CAFFE_ROOT = '/home/albert/Software/caffe' 
MODEL_DEPLOY = "../ml/cnn_deploy.prototxt"
MODEL_SNAPSHOT = "../ml/cnn512.caffemodel"

# IMG_DIR contains all images
IMG_DIR = "../images"

# Set to True if using GPU
GPU_MODE = True

# Input size of the CNN input image (after cropping)
CNN_INPUT_SIZE = 227

# CNN Batch size. Depends on the hardware memory
CNN_BATCH_SIZE = 200

# Context or 'padding' size around region proposals in pixels
CONTEXT_SIZE = 15

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
# Input: data dictionary and a string = "train" or "test"
# Output: None
#
def extractRegionFeats(data, phase):

	# Extract the image mean and compute the cropped mean
	img_mean = loadmat(os.path.join(ML_DIR, "ilsvrc_2012_mean.mat"))["image_mean"]
	offset = np.floor((img_mean.shape[0] - CNN_INPUT_SIZE)/2) + 1
	img_mean = img_mean[offset:offset+CNN_INPUT_SIZE, offset:offset+CNN_INPUT_SIZE, :]

	# Set up the Caffe network
	sys.path.insert(0, CAFFE_ROOT + 'python')

	if GPU_MODE == True:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	net = caffe.Classifier(MODEL_DEPLOY, MODEL_SNAPSHOT, mean=img_mean, channel_swap=[2,1,0], raw_scale=255)

	for i, image_name in enumerate(data[phase]["gt"].keys()):
		# Read image, compute number of batches
		img = cv2.imread(os.path.join(IMG_DIR, image_name))
		# Subtract one because bbox
		regions = data[phase]["ssearch"][image_name] - 1

		num_regions = data[phase]["ssearch"][image_name].shape[0]
		num_batches = int(np.ceil(1.0 * num_regions / CNN_BATCH_SIZE))
		
		# Extract batches from original image
		img_batch = np.zeros((CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3, CNN_BATCH_SIZE))

		for b in xrange(num_batches):
			# Create the CNN input batch
			for j in xrange(CNN_BATCH_SIZE):
				# Index into the regions array
				idx = b * CNN_BATCH_SIZE + j
				x_del = regions[idx][2] - regions[idx][0]
				y_del = regions[idx][3] - regions[idx][1]
				# If we've exhausted all examples
				if idx >= num_regions:
					break
				
				padded_region_img = getPaddedRegion(img, regions[idx])
				resized = cv2.resize(padded_region_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE)) 
				img_batch[:,:,:,j] = resized

			# Run the actual CNN to extract features
			print img_batch.shape
			scores = net.predict(img_batch)
			feat = net.blobs["fc7"].data[:,:128]
		break


################################################################
# getPaddedRegion(img, bbox)
#    Takes a bounding box and adds padding such that the coordinates
#    are in-bounds of the image
#
# Input: image (3D matrix)
#		 bbox (vector)
# Output: new_bbox (vector)
def getPaddedRegion(img, bbox):

	H, W, _ = img.shape

	x_start = max(0, bbox[0] - CONTEXT_SIZE)
	# +1 so we include the bounding box as part of the region
	x_end = min(W, bbox[2] + 1 + CONTEXT_SIZE)

	y_start = max(0, bbox[1] - CONTEXT_SIZE)
	y_end = min(H, bbox[3] + 1 + CONTEXT_SIZE)

	return img[y_start:y_end, x_start:x_end, :]


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
#	displayImageWithBboxes("img123.jpg", [[0 0 125 200]])
#
def displayImageWithBboxes(image_name, bboxes):
	img = cv2.imread(os.path.join(IMG_DIR, image_name))

	for bbox in bboxes:
		cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), randomColor(), thickness=2)

	cv2.imshow("Image", img)
	#cv2.waitKey(0)


if __name__ == "__main__":
	main()


