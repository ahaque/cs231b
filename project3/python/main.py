import cv2
import sys
import time
import os.path
import argparse
# import caffe
from settings import *

import cPickle as cp
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from train_rcnn import *

original_img_mean = None
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

def main():
	# init_globals()
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", help="extract, train, or test", required=True)
	parser.add_argument("--num_gpus", help="For feature extraction, total number of GPUs you will use")
	parser.add_argument("--gpu_id", help="For feature extraction, GPU ID [0,num_gpus) for which part to run")
	args = parser.parse_args()

	if args.mode not in ["extract", "train", "test"]:
		print "\tError: MODE must be one of: 'extract' 'train' 'test'"
		sys.exit(-1)

	if args.mode == "extract":
		if args.num_gpus is None or args.gpu_id is None:
			print "\tFor extraction mode, must specify number of GPUs and this GPU id"
			print "\tpython main.py --mode extract --num_gpus NUM_GPUS --gpu_id GPU_ID"
			sys.exit(-1)

		num_gpus = int(args.num_gpus)
		gpu_id = int(args.gpu_id)

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
		models = []
		threads = []

		if not os.path.isdir(MODELS_DIR):
			os.makedirs(MODELS_DIR)

		for c in xrange(1, NUM_CLASSES+1):
			# Train a SVM for this class
			model = trainClassifierForClass(data, c)
			model_file_name = os.path.join(MODELS_DIR, 'svm_%d_%s.mdl'%(c, FEATURE_LAYER))
			with open(model_file_name, 'w') as fp:
				cp.dump(model, fp)
			#thread = threading.Thread(target=trainClassifierForClass, args=[data, c])
			#hread.start()
			#threads.append(thread)

		# # Train the background model
		# negative_model = trainBackgroundClassifier(data)
		# model_file_name = os.path.join(MODELS_DIR, 'svm_n_%s.mdl'%(FEATURE_LAYER))
		# with open(model_file_name, 'w') as fp:
		# 	cp.dump(model, fp)

		#print "Waiting for threads to finish..."
		#for thread in threads:
		#    thread.join()

	# Equivalent to the starter code: test_rcnn.m
	if args.mode == "test":
		pass

	# Equivalent to the starter code: extract_region_feats.m
	if args.mode == "extract":

		if not os.path.isdir(FEATURES_DIR):
			os.makedirs(FEATURES_DIR)
		
		# Set up Caffe
		net = initCaffeNetwork(gpu_id)

		for EXTRACT_MODE in ["train", "test"]:
			# Create the workload for each GPU
			ls = data[EXTRACT_MODE]["gt"].keys()
			assignments = list(chunks(ls, num_gpus))
			payload = assignments[gpu_id]

			print "Processing %i images on GPU ID %i. Total GPUs: %i" % (len(payload), gpu_id, num_gpus)
			for i, image_name in enumerate(payload):
				start = time.time()
				img = cv2.imread(os.path.join(IMG_DIR, image_name))
				# Also need to extract features from GT bboxes
				# Sometimes an image has zero GT bboxes
				if data[EXTRACT_MODE]["gt"][image_name][1].shape[0] > 0:
					regions = np.vstack((data[EXTRACT_MODE]["gt"][image_name][1], data[EXTRACT_MODE]["ssearch"][image_name]))
				else:
					regions = data[EXTRACT_MODE]["ssearch"][image_name]

				print "Processing Image %i: %s\tRegions: %i" % (i, image_name, regions.shape[0])

				features = extractRegionFeatsFromImage(net, img, regions)
				print "\tTotal Time: %f seconds" % (time.time() - start)

				np.save(os.path.join(FEATURES_DIR, image_name + '.npy'), features)

# Takes a list and splits it into roughly equal parts
def chunks(items, num_gpus):
	n = int(np.ceil(1.0*len(items)/num_gpus))
	for i in xrange(0, len(items), n):
		yield items[i:i+n]

################################################################
# initCaffeNetwork()
#   Initializes Caffe and loads the appropriate model files
# 
# Input: None
# Output: net (caffe network used to predict images)
#
def initCaffeNetwork(gpu_id):
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
	# sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))

	if GPU_MODE == True:
		caffe.set_mode_gpu()
		local_id = gpu_id % 4
		caffe.set_device(local_id)
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

	H, W, _ = img.shape

	# Pad the image with -1's
	# -1's indicate that this pixel will be replaced with the image mean
	padded_img = -1 * np.ones((H + 2*INDICATOR_PAD_SIZE, W + 2*INDICATOR_PAD_SIZE, 3))

	# Add the region to the center of this new "padded" image
	start = INDICATOR_PAD_SIZE
	padded_img[start:start+H, start:start+W, :] = img

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
			
			warped = warpRegion(padded_img, regions[idx])
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

################################################################
# warpRegion(img, bbox)
#	Takes an input image and bbox and outputs the warped version
#	The warped version includes a guaranteed padding size around
#   the warped version and will use the imagenet mean if the padding
#	exceeds the original image dimensions
#
# Input: img (H x W x 3 matrix)
# 		 bbox (vector of length 4)
# Output: resized_img (227x227 warped image)
#
def warpRegion(padded_img, bbox, debug=False):
	global original_img_mean
	
	bbH = bbox[3] - bbox[1] + 1 # Plus one to include the box as part of the region
	bbW = bbox[2] - bbox[0] + 1

	translated_bbox = bbox + INDICATOR_PAD_SIZE

	original_region = padded_img[translated_bbox[1]:translated_bbox[3], translated_bbox[0]:translated_bbox[2],:]
	if debug:
		temp_region = np.copy(original_region).astype(np.uint8)
		temp_region[temp_region < 0 ] = 0
		cv2.imshow("Original", temp_region)

	subimg_size = float(CNN_INPUT_SIZE - CONTEXT_SIZE) # Usually 227-16 = 211

	# Compute the scaling factor. Original region box must be sized to subimg_size
	scaleH = subimg_size / bbH
	scaleW = subimg_size / bbW

	# Compute how many context pixels we need from the original image
	contextW = int(np.ceil(CONTEXT_SIZE / scaleW))
	contextH = int(np.ceil(CONTEXT_SIZE / scaleH))

	# Get the new region which includes context from the padded image
	startY = translated_bbox[1] - contextH
	startX = translated_bbox[0] - contextW
	endY = translated_bbox[3] + contextH + 1
	endX = translated_bbox[2] + contextW + 1
	cropped_region = padded_img[startY:endY, startX:endX, :]

	# Resize the image and replace -1 with the image mean
	resized_img = cv2.resize(cropped_region, (CNN_INPUT_SIZE, CNN_INPUT_SIZE), interpolation=cv2.INTER_LINEAR) 

	# Replace any -1 with the mean image
	resized_img[resized_img < 0] = original_img_mean[resized_img < 0]

	if debug:
		cv2.imshow("Mean-Padded", resized_img.astype(np.uint8))
		cv2.imshow("Mean", original_img_mean)
		cv2.waitKey(0)

	return resized_img

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


## TODO
# Randomly select validation set
# Argparse all the gazillion options
# Try L2 later, try strength
# More positive examples from overlapping boxes
# Weigh the positive examples more