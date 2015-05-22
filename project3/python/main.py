import cv2
import sys
import time
import os.path
import argparse
import caffe

import numpy as np
import matplotlib.pyplot as plt

from util import *
from scipy.io import loadmat

################################################################
# BEGIN REQUIRED INPUT PARAMETERS

# For all DIRs and paths, the trailing slash does not matter

ML_DIR = "../ml" # ML_DIR contains matlab matrix files and caffe model
IMG_DIR = "../images" # IMG_DIR contains all images
FEATURES_DIR = "../features/mean_padding" # FEATURES_DIR stores the region features for each image

CAFFE_ROOT = '/home/albert/Software/caffe' # Caffe installation directory
MODEL_DEPLOY = "../ml/cnn_deploy.prototxt" # CNN architecture file
MODEL_SNAPSHOT = "../ml/cnn512.caffemodel" # CNN weights

GPU_MODE = True # Set to True if using GPU

# CNN Batch size. Depends on the hardware memory
# NOTE: This must match exactly value of line 3 in the deploy.prototxt file
CNN_BATCH_SIZE = 1000 # CNN batch size
CNN_INPUT_SIZE = 227 # Input size of the CNN input image (after cropping)
CONTEXT_SIZE = 16 # Context or 'padding' size around region proposals in pixels

# The layer and number of features to use from that layer
# Check the deploy.prototxt file for a list of layers/feature outputs
FEATURE_LAYER = "fc6_ft"
NUM_CNN_FEATURES = 512

NUM_CLASSES = 3 # Number of object classes

# END REQUIRED INPUT PARAMETERS
################################################################

original_img_mean = None
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", help="extract, train, or test")
	args = parser.parse_args()

	if args.mode not in ["extract", "train", "test"]:
		print "Error: MODE must be one of: 'extract' 'train' 'test'"
		print "Usage: python main.py --mode MODE"
		sys.exit(-1)

	# Read the Matlab data files
	# data["train"]["gt"]["2008_007640.jpg"] = tuple( class_labels, gt_bboxes )
	# data["train"]["gt"]["2008_007640.jpg"] = tuple( [[2]] , [[ 90,  85, 500, 366]] )
	# data["train"]["ssearch"]["2008_007640.jpg"] = n x 4 matrix of region proposals (bboxes)
	data = {}
	data["train"] = readMatrixData("train")
	data["test"] = readMatrixData("test")

	# Equivalent to the starter code: train_rcnn.m
	if args.mode == "train":
		# For each object class
		for c in xrange(1, NUM_CLASSES+1):
			# Train a SVM for this class
			model = trainClassifierForClass(data, c)
			break
		pass

	# Equivalent to the starter code: test_rcnn.m
	if args.mode == "test":
		pass

	# Equivalent to the starter code: extract_region_feats.m
	if args.mode == "extract":
		# Set up Caffe
		net = initCaffeNetwork()

		# Extract features from every training image
		for i, image_name in enumerate(data["train"]["gt"].keys()):
			start = time.time()
			img = caffe.io.load_image(os.path.join(IMG_DIR, image_name))
			img = cv2.imread(os.path.join(IMG_DIR, image_name))
			regions = data["train"]["ssearch"][image_name]
			
			print "Processing Image %i: %s\tRegions: %i" % (i, image_name, regions.shape[0])

			features = extractRegionFeatsFromImage(net, img, regions)
			print "\tTotal Time: %f seconds" % (time.time() - start)
			np.save(os.path.join(FEATURES_DIR, image_name), features)


def trainClassifierForClass(data, class_id):
	# Go through each image and build the training set with pos/neg labels
	X_train = []
	y_train = []
	for i, image_name in enumerate(data["train"]["gt"].keys()):
		# If this image has no detections?
		if data["train"]["gt"][image_name][0].shape[0] == 0:
			continue

		labels = np.array(data["train"]["gt"][image_name][0][0])
		gt_bboxes = np.array(data["train"]["gt"][image_name][1])
		regions = data["train"]["ssearch"]["2008_007640.jpg"]
		print labels, bboxes

		IDX = np.where(labels == class_id)
		# For each region, see if it overlaps > 0.5 with one of the IDX bboxes
		# If yes, extract features from this region (make sure to add padding)
		# Add the feature vectors to X_train as a positive example

		# If overlap is low < 0.3, then add to X_train as a negative example

		print "-------------------------------------"

################################################################
# initCaffeNetwork()
#   Initializes Caffe and loads the appropriate model files
# 
# Input: None
# Output: net (caffe network used to predict images)
#
def initCaffeNetwork():
	# Extract the image mean and compute the cropped mean
	global original_img_mean
	original_img_mean = loadmat(os.path.join(ML_DIR, "ilsvrc_2012_mean.mat"))["image_mean"]
	offset = np.floor((original_img_mean.shape[0] - CNN_INPUT_SIZE)/2) + 1
	original_img_mean = original_img_mean[offset:offset+CNN_INPUT_SIZE, offset:offset+CNN_INPUT_SIZE, :]
	# Must be in the form (3,227,227)
	img_mean = np.swapaxes(original_img_mean,0,1)
	img_mean = np.swapaxes(img_mean,0,2)	
	# Used for warping
	original_img_mean = original_img_mean.astype(np.uint8)

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
	# Subtract one because bboxs are indexed starting at 1 but numpy is at 0
	regions -= 1

	num_regions = regions.shape[0]
	num_batches = int(np.ceil(1.0 * num_regions / CNN_BATCH_SIZE))
	features = np.zeros((num_regions, NUM_CNN_FEATURES))

	# Extract batches from original image
	for b in xrange(num_batches):
		# Create the CNN input batch
		img_batch = []
		num_in_this_batch = 0
		start = time.time()
		for j in xrange(CNN_BATCH_SIZE):
			# Index into the regions array
			idx = b * CNN_BATCH_SIZE + j

			# If we've exhausted all examples
			if idx < num_regions:
				num_in_this_batch += 1
			else:
				break
			
			warped = warpRegion(img, regions[idx])
			#padded_region_img = getPaddedRegion(img, regions[idx])
			#resized = cv2.resize(padded_region_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE)) 
			img_batch.append(warped)

		#print "\tBatch %i creation: %f seconds" % (b, time.time() - start)
		# Run the actual CNN to extract features
		start = time.time()
		scores = net.predict(img_batch)
		print "\tBatch %i / %i: %f seconds" % (b+1, num_batches, time.time() - start)

		# The last batch will not be completely full so we don't want to save all of them
		start_idx = b*CNN_BATCH_SIZE
		features[start_idx:start_idx+num_in_this_batch,:] = net.blobs[FEATURE_LAYER].data[0:num_in_this_batch,:]
		
	return features


def warpRegion(img, bbox):
	global original_img_mean
	H, W, _ = img.shape
	bbH = bbox[3] - bbox[1] + 1 # Plus one to include the box as part of the region
	bbW = bbox[2] - bbox[0] + 1

	original_region = img[bbox[1]:bbox[3], bbox[0]:bbox[2],:]
	cv2.imshow("Original", original_region)

	subimg_size = float(CNN_INPUT_SIZE - CONTEXT_SIZE) # Usually 227-16 = 211

	# Pad the image with -1's
	# -1's indicate that this pixel will be replaced with the image mean
	indicator_pad_size = 100
	padded_img = -1 * np.ones((H + 2*indicator_pad_size, W + 2*indicator_pad_size, 3))

	# Add the region to the center of this new "padded" image
	start = indicator_pad_size
	padded_img[start:start+H, start:start+W, :] = img

	# Compute the scaling factor. Original region box must be sized to subimg_size
	scaleH = subimg_size / bbH
	scaleW = subimg_size / bbW

	# Compute how many context pixels we need from the original image
	contextW = int(np.ceil(CONTEXT_SIZE / scaleW))
	contextH = int(np.ceil(CONTEXT_SIZE / scaleH))

	# Get the new region which includes context from the padded image
	translated_bbox = bbox + indicator_pad_size
	startY = translated_bbox[1] - contextH
	startX = translated_bbox[0] - contextW
	endY = translated_bbox[3] + contextH + 1
	endX = translated_bbox[2] + contextW + 1
	cropped_region = padded_img[startY:endY, startX:endX, :]

	# Resize the image and replace -1 with the image mean
	resized_img = cv2.resize(cropped_region, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), interpolation=cv2.INTER_LINEAR) 

	# Replace any -1 with the mean image
	resized_img[resized_img < 0] = original_img_mean[resized_img < 0]

	#cv2.imshow("Mean-Padded", resized_img.astype(np.uint8))
	#cv2.imshow("Mean", original_img_mean)
	#cv2.waitKey(0)

	return resized_img

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


