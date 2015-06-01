import cv2
import os.path
import sys

import numpy as np

from settings import *

COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]

################################################################
# computeOverlap(bbox, bboxes)
#   Computes the overlap between bbox and all bboxes
#
# Input: bbox (1x4 vector)
#		 bboxes (nx4 matrix)
# Output: vector of size n of overlap values
#
# 	overlap = computeOverlap(data["train"]["gt"][image_name][1], \
#							     data["train"]["ssearch"][image_name])
def computeOverlap(bbox, bboxes):
	# bbox can be a singular array, or a matrix.
	if len(bbox.shape) > 1:
		bbox = bbox[0]
	x1 = np.maximum(bboxes[:,0], bbox[0])
	y1 = np.maximum(bboxes[:,1], bbox[1])
	x2 = np.minimum(bboxes[:,2], bbox[2])
	y2 = np.minimum(bboxes[:,3], bbox[3])

	w = x2 - x1 + 1
	h = y2 - y1 + 1

	inter = np.multiply(w, h)
	inter[w <= 0] = 0
	inter[h <= 0] = 0
	
	bboxes_area = np.multiply((bboxes[:,2]-bboxes[:,0]+1), (bboxes[:,3]-bboxes[:,1]+1))
	bbox_area = np.multiply((bbox[2]-bbox[0]+1),(bbox[3]-bbox[1]+1))
	union = bboxes_area + (bbox_area * np.ones(bboxes_area.shape)) - inter
	union[w <= 0] = 1
	union[h <= 0] = 1
	
	overlap = np.divide(1.0*inter, union)

	# overlap[w <= 0] = 0
	# overlap[h <= 0] = 0

	return overlap

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
def displayImageWithBboxes(image_name, bboxes, gt_bboxes=None, color=None): 
	print bboxes
	bboxes = bboxes.astype(np.int32)
	img = cv2.imread(os.path.join(IMG_DIR, image_name))

	cv2.imshow("Original", img)
	for bbox in bboxes:
		if color is None:
			box_color = randomColor()
		else:
			box_color = color
		cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, thickness=1)

	title = "Image (GT bboxes : %d)"%(0)
	if gt_bboxes is not None:
		title = "Image (GT bboxes : %d)"%(len(gt_bboxes))
		for gt_bbox in gt_bboxes:
			if color is None:
				box_color = randomColor()
			else:
				box_color = color
			cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0,0,0), thickness=4)

	cv2.imshow(title, img)
	cv2.moveWindow(title, 600, 0)
	if cv2.waitKey(0) == ord('q'):
		sys.exit(0)
	cv2.destroyWindow(title)

def normalizeFeaturesIndependent(features):
    # If no features, return
    if features.shape[0] == 0:
        return features

    mu = np.mean(features, axis=1)
    std = np.std(features, axis=1)

    result = features - np.tile(mu, (features.shape[1], 1)).T
    result = np.divide(result, np.tile(std, (features.shape[1], 1)).T)

    return result

################################################################
# normalizeFeatures(features)
#   Takes a matrix of features (each row is a feature) and
#   normalizes each row to mean=0, variance=1
#
# Input: features (n x NUM_CNN_FEATURES matrix)
# Output: result (n x NUM_CNN_FEATURES matrix)
#
def normalizeFeatures(features):
    # If no features, return
    if features.shape[0] == 0:
        return features

    mu = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    result = features - np.tile(mu, (features.shape[0], 1))
    result = np.divide(result, np.tile(std, (features.shape[0], 1)))

    return result

def stack(data):
	result = None
	if len(data) == 0:
		result = None
	elif len(data) == 1:
		result = data[0]
	else:
		result = np.vstack(tuple(data))

	return result