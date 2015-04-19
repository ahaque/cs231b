import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import grabcut
import time
import threading
import Queue
import subprocess
import atexit

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
procs_list = dict()

@atexit.register
def kill_subprocesses():
    for img_name in procs_list:
        procs_list[img_name].kill()

def computeJaccard(segmentation, ground_truth):
    intersection = np.sum(np.logical_and(ground_truth > 0, segmentation > 0))
    union = np.sum((segmentation + ground_truth) > 0)
    return 100 * intersection / union

def computeAccuracy(segmentation, ground_truth):
    num_pixels = segmentation.shape[0]*segmentation.shape[1]
    num_correct_foreground = np.sum(np.logical_and(ground_truth > 0, segmentation > 0))
    num_correct_background = np.sum(np.logical_and(ground_truth == 0, segmentation == 0))
    return 100 * (num_correct_background + num_correct_foreground) / num_pixels

def processImage(q, machine, image_name):
    start_time = time.time()
    server = "fdalvi@%s.stanford.edu"%machine
    command = "cd private/cs231b/project1/;python ml.py " + image_name
    process = subprocess.Popen(["ssh","-t", server,command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    procs_list[image_name] = process
    output = process.communicate()[0]
    index = output.find(image_name)
    while index == -1:
        # sys.stdout.write('Server %s is down\n'%server)
        # sys.stdout.flush()
        machine = "corn%02d"%(int(machine[4:])+1)
        
        server = "fdalvi@%s.stanford.edu"%machine
        # sys.stdout.write('Trying %s\n'%server)
        # sys.stdout.flush()
        process = subprocess.Popen(["ssh","-t", server,command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.communicate()[0]
        index = output.find(image_name)
    
    # sys.stdout.write(output[index: output.find("\n",index)] + "\n")
    # sys.stdout.flush()
    scores = output[index: output.find("\n",index)].split()
    q.put((image_name, float(scores[1]), float(scores[2])))
    del procs_list[image_name]

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
        server = "corn%02d"%(tid+5) 
        thread = threading.Thread(name=image_name,target=processImage, args=(q, server, image_name))
        thread_list.append(thread)
        thread.start()

    while len(thread_list) > 0:
        threads_to_remove = set()
        for i,t in enumerate(thread_list):
            t.join(1.0)
            if not t.is_alive():
                print 'Thread for %s done.'%t.getName()
                threads_to_remove.add(i)
        if len(threads_to_remove) == 0:
            continue
        else:
            thread_list = [thread for i,thread in enumerate(thread_list) if i not in threads_to_remove]

    results = []
    while not q.empty():
        img_name, acc, jaccard = q.get()
        results.append((img_name, acc, jaccard))
        all_accuracies.append(acc)
        all_jaccards.append(jaccard)

    results.sort(key=lambda tup: tup[0]) 
    for img_name, acc, jaccard in results:
        print img_name, acc, jaccard

    print "------------------------------------------------------"
    print "Number of Images:", len(filenames)
    print "Average Accuracy:", np.mean(np.array(all_accuracies))
    print "Average Jaccard:", np.mean(np.array(all_jaccards))
    print "------------------------------------------------------"

if __name__ == '__main__':
    main()