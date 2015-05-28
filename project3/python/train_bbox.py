
import cv2
import sys
import time
import os.path
import argparse
import caffe
import util

from settings import *
from sklearn import linear_model

def trainBboxRegressionForClass(data, classid):

	models = []

	# Loop through all images and get features and bbox
	for i, image_name in enumerate(data["train"]["gt"].keys()):
        if not os.path.isfile(os.path.join(FEATURES_DIR, image_name + '.npy')):
            continue

        # Find positive examples by computing overlap of all regions with GT bbox
		regions = data["train"]["ssearch"][image_name].astype(np.int32) 
		overlaps = np.zeros((len(IDX), regions.shape[0]))

		for j, gt_bbox in enumerate(gt_bboxes[IDX]):
			overlaps[j,:] = util.computeOverlap(gt_bbox, regions)
		highest_overlaps = overlaps.max(0)

		# Select Positive/Negatives Regions
		# Store the features of these positive examples and bbox coordinates
		positive_idx = np.where(highest_overlaps > POSITIVE_THRESHOLD)[0]
		X_train.append(features[IDX, :]) # GT boxes
		X_train.append(features[positive_idx, :]) # High overlapping regions

		# Compute the bbox parameters: (center_x, center_y, log(width), log(height))

	    # Now we should formulate X_train which contains N bbox features ( N x 512 )
	    # and y_train which is N x 4 (center_x, center_y, log(width), log(height))

    for i in xrange(y_train.shape[1]):
    	model = linear_model.Ridge(alpha = .5)
    	model.fit(X_train, y_train[:,i])
    	models.append(model)

    # Now train 4 different regressions, one for each bbox parameter (y_train)
    # So in total, for the three classes, we will have 12 regressions