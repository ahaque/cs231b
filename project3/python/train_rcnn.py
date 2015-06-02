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
    X_pos_val = []
    X_neg_val = []

    # Split train into train and val set
    val_set_images = set(random.sample(data["train"]["gt"].keys(), 100))

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
            if image_name in val_set_images:
                X_neg_val.append(features)
            else:
                X_neg.append(features)
        else:
            labels = np.array(data["train"]["gt"][image_name][0][0])
            gt_bboxes = np.array(data["train"]["gt"][image_name][1])
            IDX = np.where(labels == class_id)[0]

            if len(IDX) == 0: # Case 2
                if image_name in val_set_images:
                    X_neg_val.append(features)
                else:
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
                # X_pos.append(features[IDX, :]) # GT box
                # X_pos.append(features[positive_idx, :]) # GT box overlapping regions
                if image_name in val_set_images:
                    X_pos_val.append(features[IDX, :])
                    X_pos_val.append(features[positive_idx, :])
                else:
                    X_pos.append(features[IDX, :])
                    X_pos.append(features[positive_idx, :])
                

                # Only add negative examples where bbox is far from all GT boxes
                negative_idx = np.where(highest_overlaps < NEGATIVE_THRESHOLD)[0]
                negative_idx += num_gt_bboxes
                if image_name in val_set_images:
                    X_neg_val.append(features[negative_idx, :])
                else:
                    X_neg.append(features[negative_idx, :])

        if (i+1) % 50 == 0:
            print "Finished %i / %i.\tElapsed: %f" % (i+1, num_images, time.time()-start_time)

    model = None
    scaler = None
    print 'Stacking...'
    start_time = time.time()
    X_pos = util.stack(X_pos)
    X_neg = util.stack(X_neg)
    X_pos_val = util.stack(X_pos_val)
    X_neg_val = util.stack(X_neg_val)
    end_time = time.time()
    if debug: print 'Stacking took: %f'%(end_time - start_time)

    print X_pos.shape, X_neg.shape, X_pos_val.shape, X_neg_val.shape

    if debug: print 'Normalizing and adding bias to positive...'
    start_time = time.time()
    # X_pos = util.normalizeAndAddBias(X_pos)
    end_time = time.time()
    if debug: print 'Normalizing and adding bias to positive took: %f'%(end_time - start_time)

    num_positives = X_pos.shape[0]
    num_negatives = X_neg.shape[0]
    hard_negs = []
    curr_num_hard_negs = 0
    for epoch in xrange(epochs):
        X_neg_idx = np.random.permutation(num_negatives)

        start_idx = 0
        increment = 2*num_positives
        while start_idx < num_negatives:
            print "[INFO] Negative traversal process: %d / %d"%(start_idx, num_negatives)
            end_idx = min(start_idx + increment, num_negatives)
            if model is None:
                if debug: print 'Picking Features'
                start_time = time.time()
                X_neg_subset = X_neg[X_neg_idx[start_idx:end_idx], :]
                end_time = time.time()
                if debug: print 'Picking took: %f'%(end_time - start_time)

                if debug: print 'Normalizing and adding bias to negative subset...'
                start_time = time.time()
                # X_neg_subset = util.normalizeAndAddBias(X_neg_subset)   
                end_time = time.time()
                if debug: print 'Normalizing and adding bias to negative subset took: %f'%(end_time - start_time)

                print 'Training SVM...'
                start_time = time.time()
                model, scaler = trainSVM(X_pos, X_neg_subset)
                end_time = time.time()
                print 'Training took: %f'%(end_time - start_time)

                # getValidationAccuracy(model, X_pos_val, X_neg_val)

                hard_negs.append(X_neg_subset)
                num_iter_since_retrain = 0
            else:
                if debug: print 'Picking Features'
                start_time = time.time()
                X_neg_subset = X_neg[X_neg_idx[start_idx:end_idx], :]
                end_time = time.time()
                if debug: print 'Picking took: %f'%(end_time - start_time)

                if debug: print 'Normalizing and adding bias to negative subset...'
                start_time = time.time()
                X_neg_subset_1, _ = util.normalizeAndAddBias(X_neg_subset, scaler)
                end_time = time.time()
                if debug: print 'Normalizing and adding bias to negative subset took: %f'%(end_time - start_time)

                # Classify negative features using small_svm
                y_hat = model.predict(X_neg_subset_1)
                hard_idx = np.where(y_hat == 1)[0]
                hard_negs_subset = X_neg_subset[hard_idx, :]
                hard_negs.append(hard_negs_subset)
                
                curr_num_hard_negs += hard_negs_subset.shape[0]
                print 'Num Hard: %d / %d'%(hard_negs_subset.shape[0], X_neg_subset.shape[0])

                # Check if we need to retrain SVM
                if curr_num_hard_negs > 10000:
                    hard_negs = util.stack(hard_negs)

                    # Filter negs
                    X, _ = util.normalizeAndAddBias(hard_negs, scaler)
                    conf = model.decision_function(X)
                    # y = model.predict(X)
                    # print conf[IDX[10000:10100]], y[IDX[10000:10100]]

                    # Discard 1/4th easy examples
                    # IDX = np.argsort(conf)
                    # IDX = IDX[IDX.shape[0]/4:]
                    
                    # Keep only max_num_neg negatives
                    max_num_neg = X_pos.shape[0] * 3
                    IDX = np.argsort(-1*conf)
                    IDX = IDX[0:min(max_num_neg, IDX.shape[0])]
                    
                    # Keep only negatives with decision score > -1.0
                    # IDX = np.where(conf > -1.0)[0]
                    print 'Old num hard negs: %d'%(hard_negs.shape[0])
                    hard_negs = hard_negs[IDX,:]
                    print 'New num hard negs: %d'%(hard_negs.shape[0])

                    print 'Retraining SVM (Skipped %d retrains) (Num hard: %d)...'%(num_iter_since_retrain, curr_num_hard_negs)
                    start_time = time.time()
                    model, scaler = trainSVM(X_pos, hard_negs)
                    end_time = time.time()
                    print 'Retraining took: %f'%(end_time - start_time)
                    # getValidationAccuracy(model, X_pos_val, X_neg_val)

                    hard_negs = [hard_negs]
                    curr_num_hard_negs = 0
                    num_iter_since_retrain = 0
                else:
                    num_iter_since_retrain += 1
                    # print 'No retrain required (Num hard: %d)'%(curr_num_hard_negs)

            start_idx += increment

    print 'Retraining SVM (Final)...'
    hard_negs = util.stack(hard_negs)
    X, _ = util.normalizeAndAddBias(hard_negs, scaler)
    conf = model.decision_function(X)
    
    # Keep only max_num_neg negatives
    max_num_neg = X_pos.shape[0] * 3
    IDX = np.argsort(-1*conf)
    IDX = IDX[0:min(max_num_neg, IDX.shape[0])]
    hard_negs = hard_negs[IDX,:]
    model, scaler = trainSVM(X_pos, hard_negs, model=model, debug=True)

    getValidationAccuracy(model, scaler, X_pos_val, X_neg_val)
    
    return (model, scaler)

def getValidationAccuracy(model, scaler, X_pos_val, X_neg_val):
    # Check accuracy on validation set
    print '\n\n'
    print 'Validation set accuracy'
    print '-----------------------'
    with open(os.path.join(MODELS_DIR, 'results.txt'), 'a') as fp:
        print >> fp, '\n'
        print >> fp, 'Validation set accuracy'
        print >> fp, '-----------------------'
    # X_pos_val = util.normalizeAndAddBias(X_pos_val)
    # X_neg_val = util.normalizeAndAddBias(X_neg_val)
    X = util.stack([X_pos_val, X_neg_val])

    X = scaler.transform(X)

    y = [np.ones((X_pos_val.shape[0], 1)), np.zeros((X_neg_val.shape[0], 1))]
    y = np.squeeze(np.vstack(tuple(y)))

    print X.shape, y.shape
    
    # Classify negative features using small_svm
    y_hat = model.predict(X)
    print "Positive Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==1)) / np.sum(y==1)
    print "Negative Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==0)) / np.sum(y==0)
    print "Training Accuracy:", 1.0 * np.sum(y == y_hat) / y.shape[0]
    print "-----------------------"
    with open(os.path.join(MODELS_DIR, 'results.txt'), 'a') as fp:
        print >> fp, "Positive Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==1)) / np.sum(y==1)
        print >> fp, "Negative Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==0)) / np.sum(y==0)
        print >> fp, "Training Accuracy:", 1.0 * np.sum(y == y_hat) / y.shape[0]
        print >> fp, "-------------------------------------"
    
def trainSVM(pos_features, neg_features, model=None, debug=False):
    start_time = time.time()

    if debug: 
        print "Num Positive:", pos_features.shape
        print "Num Negatives:", neg_features.shape
        print "Num Total:", pos_features.shape[0] + neg_features.shape[0]
        with open(os.path.join(MODELS_DIR, 'results.txt'), 'a') as fp:
            print >> fp, "Num Positive:", pos_features.shape
            print >> fp, "Num Negatives:", neg_features.shape
            print >> fp, "Num Total:", pos_features.shape[0] + neg_features.shape[0]

    print 'Training on %d pos and %d neg'%(pos_features.shape[0], neg_features.shape[0])

    # IDX = np.random.permutation(neg_features.shape[0])
    # IDX = IDX[0:min(pos_features.shape[0], neg_features.shape[0])]
    # IDX = IDX[0:]
    # neg_features = neg_features[IDX,:]
    # print neg_features.shape
    
    # Build inputs
    X = np.vstack((pos_features,neg_features))
    X, scaler = util.normalizeAndAddBias(X)

    y = [np.ones((pos_features.shape[0], 1)), np.zeros((neg_features.shape[0], 1))]
    y = np.squeeze(np.vstack(tuple(y)))

    # Train the SVM
    model = svm.LinearSVC(penalty="l2", dual=False, fit_intercept=True, intercept_scaling=1, C=0.0001)
    if debug: print "Training SVM..."
    # print X.shape, y.shape
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
        with open(os.path.join(MODELS_DIR, 'results.txt'), 'a') as fp:
            print >> fp, "Positive Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==1)) / np.sum(y==1)
            print >> fp, "Negative Accuracy:", 1.0 * np.sum(np.logical_and(y == y_hat, y==0)) / np.sum(y==0)
            print >> fp, "Training Accuracy:", 1.0 * num_correct / y.shape[0]
            print >> fp, 'Total Time: %d seconds'%(time.time() - start_time)

    return model, scaler

def main():
    print "Error: Do not run train_rcnn.py directly. You should use main.py."

if __name__ == '__main__':
    main()
