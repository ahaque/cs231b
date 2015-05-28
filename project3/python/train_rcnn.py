import os
import sys
import time
import util
import numpy as np
import random

from sklearn import svm
from settings import *

def trainClassifierForClass(data, class_id, epochs=1, memory_size=1000, debug=False):
    X_pos = []
    X_neg = []

    num_images = len(data["train"]["gt"].keys())
    start_time = time.time()

    for i, image_name in enumerate(data["train"]["gt"].keys()):
        # Load features from file for current image
        features_file_name = os.path.join(FEATURES_DIR, image_name + '.npy')
        if not os.path.isfile(features_file_name):
            print 'ERROR: Missing features file \'%s\''%(features_file_name) 
            sys.exit(1)
        features = np.load(features_file_name)

        num_gt_bboxes = data["train"]["gt"][image_name][0].shape[1]

        # Case 1: No GT boxes in image. Cannot compute overlap with regions.
        # Case 2: No GT boxes in image for current class
        # Case 3: GT boxes in image for current class
        if num_gt_bboxes == 0: # Case 1
            X_neg.append(features)
        else:
            labels = np.array(data["train"]["gt"][image_name][0][0])
            gt_bboxes = np.array(data["train"]["gt"][image_name][1])
            IDX = np.where(labels == class_id)[0]

            if len(IDX) == 0: # Case 2
                X_neg.append(features)
            else:
                regions = data["train"]["ssearch"][image_name]

                overlaps = np.zeros((regions.shape[0], len(IDX)))
                for j, gt_bbox in enumerate(gt_bboxes[IDX]):
                    overlaps[:,j] = util.computeOverlap(gt_bbox, regions)
                highest_overlaps = overlaps.max(axis=1)

                # TODO: PLOTTT THIISSS
                # import matplotlib.pyplot as plt
                # plt.hist(highest_overlaps[highest_overlaps>0.001], bins=200)
                # plt.show()
                 
                assert(max(IDX) < num_gt_bboxes)

                # Select Positive/Negatives Regions
                positive_idx = np.where(highest_overlaps > POSITIVE_THRESHOLD)[0]
                positive_idx += num_gt_bboxes
                X_pos.append(features[IDX, :]) # GT box
                X_pos.append(features[positive_idx, :]) # GT box overlapping regions

                # Only add negative examples where bbox is far from all GT boxes
                negative_idx = np.where(highest_overlaps < NEGATIVE_THRESHOLD)[0]
                negative_idx += num_gt_bboxes
                X_neg.append(features[negative_idx, :])

        if (i+1) % 50 == 0:
            print "Finished %i / %i.\tElapsed: %f" % (i+1, num_images, time.time()-start_time)


    model = None
    print 'Stacking...'
    start_time = time.time()
    X_pos = util.stack(X_pos)
    X_neg = util.stack(X_neg)
    end_time = time.time()
    if debug: print 'Stacking took: %f'%(end_time - start_time)

    if debug: print 'Normalizing and adding bias to positive...'
    start_time = time.time()
    X_pos = util.normalizeFeatures(X_pos)
    X_pos = np.concatenate((np.ones((X_pos.shape[0], 1)), X_pos), axis=1)    
    end_time = time.time()
    if debug: print 'Normalizing and adding bias to positive took: %f'%(end_time - start_time)

    num_positives = X_pos.shape[0]
    num_negatives = X_neg.shape[0]
    hard_negs = []
    curr_num_hard_negs = 0
    for epoch in xrange(epochs):
        X_neg_idx = np.random.permutation(num_negatives)

        start_idx = 0
        increment = num_positives*3
        while start_idx < num_negatives:
            end_idx = min(start_idx + increment, num_negatives)
            if model is None:
                if debug: print 'Picking Features'
                start_time = time.time()
                X_neg_subset = X_neg[X_neg_idx[start_idx:end_idx], :]
                end_time = time.time()
                if debug: print 'Picking took: %f'%(end_time - start_time)

                if debug: print 'Normalizing and adding bias to negative subset...'
                start_time = time.time()
                X_neg_subset = util.normalizeFeatures(X_neg_subset)
                X_neg_subset = np.concatenate((np.ones((X_neg_subset.shape[0], 1)), X_neg_subset), axis=1)    
                end_time = time.time()
                if debug: print 'Normalizing and adding bias to negative subset took: %f'%(end_time - start_time)

                print 'Training SVM...'
                start_time = time.time()
                model = trainSVM(X_pos, X_neg_subset)
                end_time = time.time()
                print 'Training took: %f'%(end_time - start_time)

                hard_negs.append(X_neg_subset)
            else:
                if debug: print 'Picking Features'
                start_time = time.time()
                X_neg_subset = X_neg[X_neg_idx[start_idx:end_idx], :]
                end_time = time.time()
                if debug: print 'Picking took: %f'%(end_time - start_time)

                if debug: print 'Normalizing and adding bias to negative subset...'
                start_time = time.time()
                X_neg_subset = util.normalizeFeatures(X_neg_subset)
                X_neg_subset = np.concatenate((np.ones((X_neg_subset.shape[0], 1)), X_neg_subset), axis=1)    
                end_time = time.time()
                if debug: print 'Normalizing and adding bias to negative subset took: %f'%(end_time - start_time)

                # Classify negative features using small_svm
                y_hat = model.predict(X_neg_subset)
                hard_idx = np.where(y_hat == 1)[0]
                hard_negs_subset = X_neg_subset[hard_idx, :]
                hard_negs.append(hard_negs_subset)
                
                curr_num_hard_negs += hard_negs_subset.shape[0]

                # Check if we need to retrain SVM
                if curr_num_hard_negs > 1000:
                    hard_negs = util.stack(hard_negs)

                    print 'Retraining SVM (Num hard: %d)...'%(curr_num_hard_negs)
                    start_time = time.time()
                    model = trainSVM(X_pos, hard_negs)
                    end_time = time.time()
                    print 'Retraining took: %f'%(end_time - start_time)

                    hard_negs = [hard_negs]
                    curr_num_hard_negs = 0
                else:
                    print 'No retrain required (Num hard: %d)'%(curr_num_hard_negs)

            start_idx += increment

    print 'Retraining SVM...'
    hard_negs = util.stack(hard_negs)
    model = trainSVM(X_pos, hard_negs, debug=True)
    sys.exit(0)

    return model


def trainBackgroundClassifier(data, debug=False):
    # TODO: FIX THIS FUNCTION
    # Go through each image and build the training set with pos/neg labels
    X_train = []
    y_train = []
    start_time = time.time()
    num_images = len(data["train"]["gt"].keys())

    for i, image_name in enumerate(data["train"]["gt"].keys()):     
        # Load features from file for current image
        features = np.load(os.path.join(FEATURES_DIR, image_name + '.npy'))

        num_gt_bboxes = data["train"]["gt"][image_name][0].shape[1]

        # If no GT boxes in image, add all regions as positive
        if num_gt_bboxes == 0:
            pos_features = features
            X_train.append(util.normalizeFeatures(pos_features))
            y_train.append(np.ones((pos_features.shape[0], 1)))
        else:
            gt_bboxes = np.array(data["train"]["gt"][image_name][1]).astype(np.int32) # Otherwise uint8 by default

            # ADD NEGATIVE EXAMPLES
            neg_features = features[0:num_gt_bboxes, :]
            X_train.append(util.normalizeFeatures(neg_features))
            y_train.append(np.zeros((neg_features.shape[0], 1)))

            
            regions = data["train"]["ssearch"][image_name].astype(np.int32) 
            overlaps = np.zeros((num_gt_bboxes, regions.shape[0]))
            
            for j, gt_bbox in enumerate(gt_bboxes):
                overlaps[j,:] = util.computeOverlap(gt_bbox, regions)

            # ADD POSITIVE EXAMPLES
            # If no GT bboxes for this class, highest_overlaps would be all
            # zeros, and all regions would be negative features
            highest_overlaps = overlaps.max(0)

            # Only add negative examples where bbox is far from all GT boxes
            negative_idx = np.where(highest_overlaps < NEGATIVE_THRESHOLD)[0]
            neg_features = features[negative_idx, :]

            X_train.append(util.normalizeFeatures(neg_features))
            y_train.append(np.zeros((neg_features.shape[0], 1)))

        if i % 50 == 0 and i > 0:
            print "Finished %i / %i.\tElapsed: %f" % (i, num_images, time.time()- start_time)

    X_train = np.vstack(tuple(X_train))
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1) # Add the bias term
    y_train = np.squeeze(np.vstack(tuple(y_train))) # Makes it a 1D array, required by SVM
    print 'classifier num total', X_train.shape, y_train.shape

    return trainSVM(X_train, y_train)
    
def trainSVM(pos_features, neg_features, debug=False):
    start_time = time.time()

    if debug: 
        print "Num Positive:", pos_features.shape
        print "Num Negatives:", neg_features.shape
        print "Num Total:", pos_features.shape[0] + neg_features.shape[0]

    IDX = np.random.permutation(neg_features.shape[0])
    print IDX.shape
    IDX = IDX[0:min(pos_features.shape[0], neg_features.shape[0])]
    print IDX.shape
    neg_features = neg_features[IDX,:]
    print neg_features.shape
    
    # Build inputs
    X = np.vstack((pos_features,neg_features))
    # X = util.normalizeFeatures(X) # Normalize
    # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # Add the bias term

    y = [np.ones((pos_features.shape[0], 1)), np.zeros((neg_features.shape[0], 1))]
    y = np.squeeze(np.vstack(tuple(y)))

    # Train the SVM
    model = svm.LinearSVC(penalty="l1", dual=False, class_weight={0:1, 1:5})
    if debug: print "Training SVM..."
    print X.shape, y.shape
    model.fit(X, y)

    # Compute training accuracy
    if debug: print "Testing SVM..."
    y_hat = model.predict(X)
    num_correct = np.sum(y == y_hat)

    if debug:
        print "Positive Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==1)) / np.sum(y==1)
        print "Negative Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==0)) / np.sum(y==0)
        print "Training Accuracy:", 1.0 * num_correct / y.shape[0]
        print 'Total Time: %d seconds'%(time.time() - start_time)
        print "-------------------------------------"

    return model

def main():
    print "Error: Do not run train_rcnn.py directly. You should use main.py."

if __name__ == '__main__':
    main()
