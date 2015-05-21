
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
# NOTE: This must match exactly value of line 3 in the deploy.prototxt file
CNN_BATCH_SIZE = 2100

# Context or 'padding' size around region proposals in pixels
CONTEXT_SIZE = 15

# The layer and number of features to use from that layer
# Check the deploy.prototxt file for a list of layers/feature outputs
FEATURE_LAYER = "fc6_ft"
NUM_CNN_FEATURES = 512

# END REQUIRED INPUT PARAMETERS
################################################################

def main():

	data = {}
	data["train"] = readMatrixData("train")
	data["test"] = readMatrixData("test")

	# Set up Caffe
	net = initCaffeNetwork()

	# Extract features from every training image
	for i, image_name in enumerate(data["train"]["gt"].keys()):
		start = time.time()
		img = caffe.io.load_image(os.path.join(IMG_DIR, image_name))
		regions = data["train"]["ssearch"][image_name]
		print "Regions:", regions.shape

		feats = extractRegionFeatsFromImage(net, img, regions)
		print "Elapsed: %f seconds" % (time.time() - start)
		break

################################################################
# initCaffeNetwork()
#   Initializes Caffe and loads the appropriate model files
# 
# Input: None
# Output: net (caffe network used to predict images)
#
def initCaffeNetwork():
	# Extract the image mean and compute the cropped mean
	img_mean = loadmat(os.path.join(ML_DIR, "ilsvrc_2012_mean.mat"))["image_mean"]
	offset = np.floor((img_mean.shape[0] - CNN_INPUT_SIZE)/2) + 1
	img_mean = img_mean[offset:offset+CNN_INPUT_SIZE, offset:offset+CNN_INPUT_SIZE, :]
	# Must be in the form (3,227,227)
	img_mean = np.swapaxes(img_mean,0,1)
	img_mean = np.swapaxes(img_mean,0,2)

	# Set up the Caffe network
	sys.path.insert(0, CAFFE_ROOT + 'python')

	if GPU_MODE == True:
		caffe.set_mode_gpu()
	else:
		caffe.set_mode_cpu()

	net = caffe.Classifier(MODEL_DEPLOY, MODEL_SNAPSHOT, mean=img_mean, channel_swap=[2,1,0], raw_scale=255)
	return net


################################################################
# extractRegionFeatsFromImage(img, regions)
#   Extract region features from an image (this runs caffe)
# 
# Input: net (the caffe network)
#		 img (as a numpy array)
#		 regions (matrix where each row is a bbox)
# Output: features (matrix of NUM_REGIONS x NUM_FEATURES)
#
def extractRegionFeatsFromImage(net, img, regions):
	# Subtract one because bboxs are indexed starting at 1
	regions -= 1

	num_regions = regions.shape[0]
	num_batches = int(np.ceil(1.0 * num_regions / CNN_BATCH_SIZE))
	features = np.zeros((num_regions, NUM_CNN_FEATURES))

	print "\tRunning %i batches" % (num_batches)
	# Extract batches from original image
	for b in xrange(num_batches):
		# Create the CNN input batch
		img_batch = []
		num_in_this_batch = 0
		start = time.time()
		print "\tStarting batch %i..." % (b)
		for j in xrange(CNN_BATCH_SIZE):
			# Index into the regions array
			idx = b * CNN_BATCH_SIZE + j
			# If we've exhausted all examples
			if idx < num_regions:
				num_in_this_batch += 1
			else:
				break
			
			padded_region_img = getPaddedRegion(img, regions[idx])
			resized = cv2.resize(padded_region_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE)) 
			img_batch.append(resized)

		print "\tBatch %i creation: %f seconds" % (b, time.time() - start)
		# Run the actual CNN to extract features
		start = time.time()
		scores = net.predict(img_batch)
		print "\tBatch %i feature extraction: %f seconds" % (b, time.time() - start)

		# The last batch will not be completely full so we don't want to save all of them
		start_idx = b*CNN_BATCH_SIZE
		features[start_idx:start_idx+num_in_this_batch,:] = net.blobs[FEATURE_LAYER].data[0:num_in_this_batch,:]
		
	print "F", features.shape

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


