import sys
import os

import matplotlib.pyplot as plt
import numpy as np

#################################################################
# BEGIN REQUIRED INPUT PARAMETERS

# Bounding box and file extension
BBOX_DIR = "bboxes/"
BBOX_EXT = ".txt"
BBOX_TIGHT_DIR = "bboxes-tight/"
BBOX_TIGHT_EXT = ".txt"

# Ground truth segmentation result and file extension
SEG_DIR = "seg_GT/"
SEG_EXT = ".bmp"

# Output log containing accuracy, jaccard, num iterations, runtime for each image
LOG_FILENAME = "ml.log"

# END REQUIRED INPUT PARAMETERS
#################################################################

def main():
    # Get image names
    filenames = os.listdir(SEG_DIR)
    image_names = [f.replace(SEG_EXT, '') for f in filenames]

    # Create output folder
    if not os.path.exists(BBOX_TIGHT_DIR):
        os.makedirs(BBOX_TIGHT_DIR)

    # Loop through all images
    for i,image_name in enumerate(image_names):
        bbox_file = open(BBOX_DIR + image_name + BBOX_EXT, "r")
        bbox = map(int, bbox_file.readlines()[0].strip().split(" "))
        print image_name
        print bbox

        segmentation = plt.imread(SEG_DIR + image_name + SEG_EXT)

        segmentation = segmentation/255

        min_x = -1
        max_x = -1
        for i in xrange(segmentation.shape[1]):
            if min_x == -1:
                if np.sum(segmentation[:,i]) != 0:
                    min_x = i-1
            else:
                if np.sum(segmentation[:,i]) == 0:
                    max_x = i
                    break
        if max_x == -1:
            max_x = segmentation.shape[1]-1
        min_y = -1
        max_y = -1
        for i in xrange(segmentation.shape[0]):
            if min_y == -1:
                if np.sum(segmentation[i,:]) != 0:
                    min_y = i-1
            else:
                if np.sum(segmentation[i,:]) == 0:
                    max_y = i
                    break
        if max_y == -1:
            max_y = segmentation.shape[0]-1
       


        print [min_x, min_y, max_x, max_y]
        with open(BBOX_TIGHT_DIR + image_name + BBOX_TIGHT_EXT, 'w') as fp:
            print >>fp, "%d %d %d %d"%(min_x, min_y, max_x, max_y)
        print ""

        # plt.subplot(1, 2, 1)
        # plt.imshow(segmentation)
        # plt.subplot(1, 2, 2)
        # plt.imshow(segmentation[min_y:max_y, min_x:max_x])
        # plt.show()

if __name__ == '__main__':
    main()