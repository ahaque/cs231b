import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import grabcut
import time
import threading
from multiprocessing import Process, Queue
import multiprocessing
import argparse

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

NUM_PROCS = multiprocessing.cpu_count()

# END REQUIRED INPUT PARAMETERS
#################################################################
def get_args():
    global BBOX_DIR
    global DATA_DIR
    global SEG_DIR
    parser = argparse.ArgumentParser(
        description='Evaluation code for the Grabcut algorithm. \
                    \nUses all the cores on the machine and processes images \
                    in parallel')
    parser.add_argument('image_file', default = None, nargs='?',
        help='Input image name (without extension or path) if you want to process a single image only')

    parser.add_argument('-b', '--bboxes', default = BBOX_DIR,
        help='Directory containing bounding boxes')
    parser.add_argument('-d', '--data', default = DATA_DIR,
        help='Directory containing images')
    parser.add_argument('-s', '--segmentations', default = SEG_DIR,
        help='Directory containing segmentations')

    args = parser.parse_args()
    BBOX_DIR = args.bboxes + '/'
    DATA_DIR = args.data + '/'
    SEG_DIR = args.segmentations + '/'

    return args

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
    segmentation = grabcut.grabcut(image, bbox, image_name, num_iterations=8, num_components=2)

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

    args = get_args()
    if args.image_file != None:
        image_names = [args.image_file]

    q = Queue()
    thread_list = []
    # Loop through all images
    for tid, image_name in enumerate(image_names):
        thread = Process(target=processImage, args=(q, tid, image_name))
        thread_list.append(thread)
        thread.start()

        while len(thread_list) >= NUM_PROCS:
            threads_to_remove = set()
            for i,t in enumerate(thread_list):
                t.join(1.0)
                if not t.is_alive():
                    threads_to_remove.add(i)
            if len(threads_to_remove) == 0:
                continue
            else:
                thread_list = [thread for i,thread in enumerate(thread_list) if i not in threads_to_remove]

    while len(thread_list) > 0:
            threads_to_remove = set()
            for i,t in enumerate(thread_list):
                t.join(1.0)
                if not t.is_alive():
                    threads_to_remove.add(i)
            if len(threads_to_remove) == 0:
                continue
            else:
                thread_list = [thread for i,thread in enumerate(thread_list) if i not in threads_to_remove]

    while not q.empty():
        img_name, acc, jaccard = q.get()
        all_accuracies.append(acc)
        all_jaccards.append(jaccard)

    print "------------------------------------------------------"
    print "Number of Images:", len(filenames)
    print "Average Accuracy:", np.mean(np.array(all_accuracies))
    print "Average Jaccard:", np.mean(np.array(all_jaccards))
    print "------------------------------------------------------"

if __name__ == '__main__':
    main()