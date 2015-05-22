
# 

import cv2
import os.path

import numpy as np

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
	bbox = bbox[0]
	x1 = np.maximum(bboxes[:,0], bbox[0])
	y1 = np.maximum(bboxes[:,1], bbox[1])
	x2 = np.maximum(bboxes[:,2], bbox[2])
	y2 = np.maximum(bboxes[:,3], bbox[3])

	w = x2 - x1 + 1
	h = y2 - y1 + 1

	inter = np.multiply(w, h)
	bboxes_area = np.multiply((bboxes[:,2]-bboxes[:,0]+1), (bboxes[:,3]-bboxes[:,1]+1))
	bbox_area = np.multiply((bbox[2]-bbox[0]+1),(bbox[3]-bbox[1]+1))

	overlap = np.divide(1.0*inter, bboxes_area + bbox_area - inter)
	overlap[w <= 0] = 0
	overlap[h <= 0] = 0

	return overlap

