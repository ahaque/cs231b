
import os

import matplotlib.pyplot as plt
import numpy as np

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

# END REQUIRED INPUT PARAMETERS
#################################################################

def main():
	# Get image names
	filenames = os.listdir(DATA_DIR)
	image_names = [f.replace(DATA_EXT, '') for f in filenames]

	# Loop through all images
	for image_name in image_names:
		bbox_file = open(BBOX_DIR + image_name + BBOX_EXT, "r")
		bbox = map(int, bbox_file.readlines()[0].strip().split(" "))

		image = plt.imread(DATA_DIR + image_name + DATA_EXT)

		# Call GrabCut. Pass the image and bounding box.

		# Convert the resulting segmentation to grayscale

		# Compare the resulting segmentation to the GT segmentation
		# ground_truth is a grayscale image (2D matrix)
		ground_truth = plt.imread(SEG_DIR + image_name + SEG_EXT)

		# Perform a logical AND on the ground_truth and resulting segmentation

		# Compute accuracy

		# Compute Jaccard similarity

		# Write to log file

if __name__ == '__main__':
    main()