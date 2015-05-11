
import cv2
import argparse
import os
import numpy as np

##################################################################
# BEGIN REQUIRED INPUTS
# Which video sequence to run
DATASET = "Lab"

# Full path location to the original dataset directory
DATASET_DIR = "/home/albert/Github/cs231b/project2/data"

# File extension of input images
DATASET_EXT = ".jpg"

#
# END REQUIRED INPUTS
##################################################################

def main():
	
	ls = os.listdir(os.path.join(DATASET_DIR, DATASET))
	ls = sorted(ls)

	for i, filename in enumerate(ls):
		new_filename = str(i+1).zfill(4) + ".jpg"
		source = os.path.join(DATASET_DIR, DATASET, filename)
		dest = os.path.join(DATASET_DIR, DATASET, new_filename)
		
		os.rename(source, dest)

if __name__ == '__main__':
	main()