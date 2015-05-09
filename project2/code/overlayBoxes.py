
import cv2
import argparse
import os
import numpy as np

##################################################################
# BEGIN REQUIRED INPUTS
# Which video sequence to run
DATASET = "Car4"

# Full path location to the original dataset directory
ORIGINAL_DATASET_DIR = "/Users/albert/Github/cs231b/project2/data"

# File extension of input images
ORIGINAL_DATASET_EXT = ".jpg"

# Feature being used
FEATURE = "Raw Pixels"

# Learning method
LEARNING_METHOD = "SVM"

# Full path location to the results folder (contains a folder for each squence)
RESULTS_FOLDER = "/Users/albert/Github/cs231b/project2/code/_output"


# END REQUIRED INPUTS
##################################################################

def main():
	args = get_args()

	# Get GT detections
	gt_bboxes = getGTDetections()

	# Get predicted detections
	pred_bboxes = getPredDetections()
	print pred_bboxes

	# Get list of images for this dataset
	num_frames = getNumFrames()

	# For each frame, overlay the bbox
	for i in xrange(1, num_frames + 1):
		# Read the image file
		temp = "%04d" % (i,)
		filename = temp + ORIGINAL_DATASET_EXT
		filename = os.path.join(ORIGINAL_DATASET_DIR, DATASET, "img", filename)

		# Get the points for GT and predicted
		gt1 = (gt_bboxes[i][0:2])
		gt2 = (gt_bboxes[i][2:4])
		print gt1, gt2

		img = cv2.cvtColor(cv2.imread(filename), cv2.CV_GRAY2RGB)
		cv2.Rectangle(img, gt1, gt2, (255,255,0), thickness=2, lineType=8, shift=0)

		cv2.imshow("Hello", img)
		cv2.waitKey(10)
		break

# Gets the predicted detection information
def getPredDetections():
	input_file = open(os.path.join(RESULTS_FOLDER, DATASET, "tld.txt"), "r")
	lines = input_file.readlines()
	result = []
	for line in lines:
		tokens = map(float, line.strip().split(","))
		tokens[0:4] = map(int, tokens[0:4])
		result.append(tokens)

	return result

# Gets the number of input frames for the given dataset
def getNumFrames():
	# Get directory listing
	ls = os.listdir(os.path.join(ORIGINAL_DATASET_DIR, DATASET, "img"))

	# Find number of images
	count = 0
	for filename in ls:
		if ORIGINAL_DATASET_EXT in filename:
			count += 1

	return count

# Reads the ground truth file and returns a matrix where row i corresponds to the
# ground truth bounding box for frame i
def getGTDetections():
	input_file = open(os.path.join(ORIGINAL_DATASET_DIR, DATASET, "groundtruth.txt"), "r")
	lines = input_file.readlines()
	result = []
	for line in lines:
		tokens = map(int, line.strip().split(","))
		result.append(tokens)
	return result

def get_args():
    global DATASET
    parser = argparse.ArgumentParser(
        description='Evaluation code for the Grabcut algorithm. \
                    \nUses a single core and processes images \
                    sequentially')

    parser.add_argument('-d', '--dataset', required=True, help='Video sequence to execute on')

    args = parser.parse_args()
    DATASET = args.dataset

if __name__ == '__main__':
	main()