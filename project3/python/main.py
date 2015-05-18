
import cv2
import sys
import time
import os.path

import numpy as np

from scipy.io import loadmat


################################################################
# BEGIN REQUIRED INPUT PARAMETERS

# For all DIRs, the trailing slash does not matter
# ML_DIR contains matlab matrix files and caffe model
ML_DIR = "../ml"

# IMG_DIR contains all images
IMG_DIR = "../images"

# END REQUIRED INPUT PARAMETERS
################################################################

COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

def main():

	data = {}
	data["train"] = readMatrixData("train")
	data["test"] = readMatrixData("test")

	#print data["train"]["gt"].keys()
	for image_name in data["train"]["gt"].keys():
		#displayImageWithBboxes(image_name, data["train"]["gt"][image_name][1])
		overlap = computeOverlap(data["train"]["gt"][image_name][1], \
								 data["train"]["ssearch"][image_name])

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
# Generates a random color from our color list
def randomColor():
	return COLORS[np.random.randint(0, len(COLORS))]

################################################################
# Displays an image with several bounding boxes
#
# Input: image_name (string)
#		 bboxes (matrix, where each row corresponds to a bbox)
# Output: None
def displayImageWithBboxes(image_name, bboxes):
	img = cv2.imread(os.path.join(IMG_DIR, image_name))

	for bbox in bboxes:
		cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), randomColor(), thickness=2)

	cv2.imshow("Image", img)
	cv2.waitKey(0)

################################################################
# Computes the overlap between bbox and all bboxes
#
# Input: bbox (1x4 vector)
#		 bboxes (nx4 matrix)
# Output: vector of size n of overlap values
def computeOverlap(bbox, bboxes):
	bbox = bbox[0]
	x1 = np.maximum(bboxes[:,0], bbox[0])
	y1 = np.maximum(bboxes[:,1], bbox[1])
	x2 = np.maximum(bboxes[:,2], bbox[2])
	y2 = np.maximum(bboxes[:,3], bbox[3])

	w = x2 - x1 + 1
	h = y2 - y1 + 1

	inter = np.multiply(w, h)
	bboxes_area = np.multiply((bboxes[:,2]-bboxes[:,0]+1), (bboxes[:,3]-bboxes[:,1]+1))
	bbox_area = np.multiply((bboxes[:,2]-bboxes[:,0]+1),(bboxes[:,3]-bboxes[:,1]+1))

	overlap = np.divide(inter, bboxes_area + bbox_area - inter)
	overlap[w <= 0] = 0
	overlap[h <= 0] = 0
	return overlap

if __name__ == "__main__":
	main()