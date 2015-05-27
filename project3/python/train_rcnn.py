import os
import time
import util
import numpy as np

from sklearn import svm
from settings import *

def stack(data):
	result = None
	if len(data) == 0:
		result = None
	elif len(data) == 1:
		result = data[0]
	else:
		result = np.vstack(tuple(data))

	return result

def trainClassifierForClass(data, class_id, epochs=1, memory_size=1000, debug=False):
    X_pos = []
    X_neg = []
    X_neg_small_trained = [] # WHAT THE SMALL SVM IS TRAINED ON
    X_neg_curr_image = [] # NEGS FOR CURR IMAGE ONLY
    X_neg_hard_since_train = [] # ALL HARD NEGS SINCE LAST TRAIN
    X_neg_initial = [] # ALL NEGS FOR SVM TRAIN
    curr_num_hard_negs = 0
    
    num_images = len(data["train"]["gt"].keys())
    small_svm = None
    start_time = time.time()

    for epoch in xrange(epochs):
        for i, image_name in enumerate(data["train"]["gt"].keys()):
            if not os.path.isfile(os.path.join(FEATURES_DIR, image_name + '.npy')):
                continue

            X_neg_curr_image = []
            # Load features from file for current image
            features = np.load(os.path.join(FEATURES_DIR, image_name + '.npy'))

            num_gt_bboxes = data["train"]["gt"][image_name][0].shape[1]

            # Case 1: No GT boxes in image. Cannot compute overlap with regions.
            # Case 2: No GT boxes in image for current class
            # Case 3: GT boxes in image for current class

            if num_gt_bboxes == 0: # Case 1
                # All regions are negative examples
                X_neg_curr_image.append(features)
            else:
                labels = np.array(data["train"]["gt"][image_name][0][0])
                gt_bboxes = np.array(data["train"]["gt"][image_name][1]).astype(np.int32) # Otherwise uint8 by default
                IDX = np.where(labels == class_id)[0]

                if len(IDX) == 0: # Case 2
                    X_neg_curr_image.append(features)
                else: # Case 3
                    # Compute Overlaps
                    regions = data["train"]["ssearch"][image_name].astype(np.int32) 
                    overlaps = np.zeros((len(IDX), regions.shape[0]))

                    for j, gt_bbox in enumerate(gt_bboxes[IDX]):
                        overlaps[j,:] = util.computeOverlap(gt_bbox, regions)
                    highest_overlaps = overlaps.max(0)

                    # TODO: PLOTTT THIISSS
                    # import matplotlib.pyplot as plt
                    # plt.hist(highest_overlaps[highest_overlaps>0.001], bins=200)
                    # plt.show()

                    # Select Positive/Negatives Regions
                    positive_idx = np.where(highest_overlaps > POSITIVE_THRESHOLD)[0]
                    X_pos.append(features[IDX, :]) # GT box
                    X_pos.append(features[positive_idx, :]) # GT box overlapping regions

                    # Only add negative examples where bbox is far from all GT boxes
                    negative_idx = np.where(highest_overlaps < NEGATIVE_THRESHOLD)[0]
                    X_neg_curr_image.append(features[negative_idx, :])

                    # if len(X_neg_small) != 0:
                    #   print 'Num Regions:%d, Negatives:%d'%(regions.shape[0], stack(X_neg_small).shape[0])
                    # else:
                    #   print 'Num Regions:%d, Negatives:%d'%(regions.shape[0], 0)

            if small_svm is None:
                X_neg_initial += X_neg_curr_image
            
            hard_negs = np.zeros((0,1))
            if small_svm is not None:
                pos_features = stack(X_pos)
                neg_features = stack(X_neg_curr_image)
                num_neg_features = neg_features.shape[0]

                # Classify negative features using small_svm
                # Find features classified as positive
                X = normalizeFeatures(neg_features) # Normalize
                X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # Add the bias term

                # X = curr image negatives
                y_hat = small_svm.predict(X)

                hard_idx = np.where(y_hat == 1)[0]
                hard_negs = neg_features[hard_idx, :]
                
                curr_num_hard_negs += hard_negs.shape[0]

                if hard_negs.shape[0] == 0:
                    if i % 50 == 0 and i > 0:
                        print "Finished %i / %i.\tElapsed: %f (Hard: %d)" % (i, num_images, time.time()-start_time, curr_num_hard_negs)
                    # X_neg_small = X_neg_small[0:-len(X_neg_curr)]
                    # print "Finished %i / %i.\tElapsed: %f (Hard: %d)" % (i, num_images, time.time()-start_time, curr_num_hard_negs)
                    continue

                X_neg_hard_since_train.append(hard_negs)

                # easy_negs = stack(X_neg_small_trained)
                # X_neg_small = [hard_negs, easy_negs]

                # easy_idx = np.where(y_hat == 0)[0]
                # easy_negs = neg_features[easy_idx, :]
                # # X_neg_small = ALL previous negative examples 
                # # hard_negs = current image hard negative examples
                # X = normalizeFeatures(easy_negs) # Normalize
                # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # Add the bias term
                # dists = small_svm.decision_function(X)
                # sorted_idx = np.argsort(dists)

                # num_easy_negs = max(0, memory_size - hard_negs.shape[0])
                # if num_easy_negs > 0:
                #   easy_negs = easy_negs[sorted_idx[0:num_easy_negs], :]
                #   X_neg_small = [hard_negs, easy_negs]
                # else:
                #   X_neg_small = [hard_negs]

                # Check if we need to retrain SVM
                if curr_num_hard_negs > 0.1*memory_size:
                    neg_features = stack(X_neg_small_trained + X_neg_hard_since_train)
                    print 'Retraining small SVM (Pos: %d, Neg: %d)...'%(pos_features.shape[0], neg_features.shape[0])

                    small_svm = trainSVM(pos_features[0:4,:], neg_features, debug=False)

                    # y_hat = small_svm.predict(X)
                    # hard_idx = np.where(y_hat == 1)[0]
                    # print 'Num hard negs after retraining: %d'%(len(hard_idx))

                    X_neg.append(neg_features[0:curr_num_hard_negs, :])

                    X_neg_small_trained = X_neg_small_trained + X_neg_hard_since_train
                    curr_num_hard_negs = 0
                    X_neg_hard_since_train = []

            elif len(X_pos) > 0:
                # NO SVM HAS BEEN TRAINED YET
                # WE HAVE INITIAL NEGS in X_neg_initial
                pos_features = stack(X_pos)
                neg_features = stack(X_neg_initial)
                num_neg_features = neg_features.shape[0]
                # Train the SVM
                print 'Training small SVM (Pos: %d, Neg: %d)...'%(pos_features.shape[0], neg_features.shape[0])
                small_svm = trainSVM(pos_features, neg_features)

                X_pos = [pos_features]
                X_neg = [neg_features]
                X_neg_small_trained = [neg_features]
            
            if i % 50 == 0 and i > 0:
                print "Finished %i / %i.\tElapsed: %f (Hard: %d)" % (i, num_images, time.time()-start_time, curr_num_hard_negs)
                # print "Finished %i / %i.\tElapsed: %f" % (i, num_images, time.time()-start_time)

    if curr_num_hard_negs > 0:
        neg_features = stack(X_neg_small_trained + X_neg_hard_since_train)
        X_neg.append(neg_features[0:curr_num_hard_negs, :])

    pos_features = stack(X_pos)
    neg_features = stack(X_neg)

    return trainSVM(pos_features, neg_features, debug=True)


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
            X_train.append(normalizeFeatures(pos_features))
            y_train.append(np.ones((pos_features.shape[0], 1)))
        else:
            gt_bboxes = np.array(data["train"]["gt"][image_name][1]).astype(np.int32) # Otherwise uint8 by default

            # ADD NEGATIVE EXAMPLES
            neg_features = features[0:num_gt_bboxes, :]
            X_train.append(normalizeFeatures(neg_features))
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

            X_train.append(normalizeFeatures(neg_features))
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
    
    # Build inputs
    X = np.vstack((pos_features,neg_features))
    X = normalizeFeatures(X) # Normalize
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # Add the bias term

    y = [np.ones((pos_features.shape[0], 1)), np.zeros((neg_features.shape[0], 1))]
    y = np.squeeze(np.vstack(tuple(y)))

    # Train the SVM
    model = svm.LinearSVC(penalty="l1", dual=False)
    if debug: print "Training SVM..."
    model.fit(X, y)

    # Compute training accuracy
    if debug: print "Testing SVM..."
    y_hat = model.predict(X)
    num_correct = np.sum(y == y_hat)

    if debug:
        print "Training Accuracy:", 1.0 * num_correct / y.shape[0]
        print 'Total Time: %d seconds'%(time.time() - start_time)
        print "-------------------------------------"

    return model


################################################################
# normalizeFeatures(features)
#   Takes a matrix of features (each row is a feature) and
#   normalizes each row to mean=0, variance=1
#
# Input: features (n x NUM_CNN_FEATURES matrix)
# Output: result (n x NUM_CNN_FEATURES matrix)
#
def normalizeFeatures(features):
    # If no features, return
    if features.shape[0] == 0:
        return features

    mu = np.mean(features, axis=1)
    std = np.std(features, axis=1)

    result = features - np.tile(mu, (features.shape[1], 1)).T
    result = np.divide(result, np.tile(std, (features.shape[1], 1)).T)

    return result