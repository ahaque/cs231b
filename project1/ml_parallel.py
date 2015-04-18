import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import grabcut
import Queue
import time
import threading

#################################################################
# BEGIN REQUIRED INPUT PARAMETERS

# Bounding box and file extension
BBOX_DIR = "bboxes/"
BBOX_EXT = ".txt"

# Input images and file extension
DATA_DIR = "data_GT/"
DATA_EXT = ".jpg"

# Ground truth segmentation result and file extension
SEG_DIR = "seg_GT/"
SEG_EXT = ".bmp"

# Output log containing accuracy, jaccard, num iterations, runtime for each image
LOG_FILENAME = "ml.log"

NUM_THREADS = 4

# END REQUIRED INPUT PARAMETERS
#################################################################

def computeJaccard(segmentation, ground_truth):
	intersection = np.sum(np.logical_and(ground_truth > 0, segmentation > 0))
	union = np.sum((segmentation + ground_truth) > 0)
	return 100 * intersection / union

def computeAccuracy(segmentation, ground_truth):
	num_pixels = segmentation.shape[0]*segmentation.shape[1]
	num_correct_foreground = np.sum(np.logical_and(ground_truth > 0, segmentation > 0))
	num_correct_background = np.sum(np.logical_and(ground_truth == 0, segmentation == 0))
	return 100 * (num_correct_background + num_correct_foreground) / num_pixels

def processImage(q, tid, image_name):
	start_time = time.time()
	print "[Thread " + str(tid) + "] Starting:" + image_name
	bbox_file = open(BBOX_DIR + image_name + BBOX_EXT, "r")
	bbox = map(int, bbox_file.readlines()[0].strip().split(" "))

	image = plt.imread(DATA_DIR + image_name + DATA_EXT)

	# Call GrabCut. Pass the image and bounding box.
	segmentation = grabcut.grabcut(image, bbox)

	# Compare the resulting segmentation to the GT segmentation
	# ground_truth is a grayscale image (2D matrix)
	ground_truth = plt.imread(SEG_DIR + image_name + SEG_EXT)

	# Compute accuracy
	accuracy = computeAccuracy(segmentation, ground_truth)
	# Compute Jaccard similarity
	jaccard = computeJaccard(segmentation, ground_truth)

	# Write to log file
	print image_name, accuracy, jaccard
	end_time = time.time()
	print "[Thread " + str(tid) + "] Finished in:" + str(end_time - start_time) + " seconds"
	q.put((image_name, accuracy, jaccard))

def main():
	# Get image names
	filenames = os.listdir(DATA_DIR)
	image_names = [f.replace(DATA_EXT, '') for f in filenames]

	all_accuracies = []
	all_jaccards = []

	if len(sys.argv) == 2:
		image_names = [sys.argv[1]]

	q = Queue.Queue()
	thread_list = []
	# Loop through all images
	for tid, image_name in enumerate(image_names):
		thread_list.append(threading.Thread(target=processImage, args=(q, tid, image_name)))
		if tid == 3:
			break
	
	for t in thread_list:
		t.start()

	for t in thread_list:
		t.join()

	s = q.get()
	print s

	print "------------------------------------------------------"
	print "Number of Images:", len(filenames)
	print "Average Accuracy:", np.mean(np.array(all_accuracies))
	print "Average Jaccard:", np.mean(np.array(all_jaccards))
	print "------------------------------------------------------"

if __name__ == '__main__':
    main()