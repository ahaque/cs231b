import os
import time
import util
import cv

import numpy as np
import cPickle as cp

from sklearn import svm
from settings import *
from train_bbox import *

classes = ['CAR', 'CAT', 'PERSON']

def nms(data):
    candidates = np.copy(data)
    results = []

    while True:
        # print candidates.shape
        curr_candidate = candidates[0, :]
        rest_candidates = candidates[1:, :]
        overlaps = util.computeOverlap(curr_candidate, rest_candidates)
        IDX = np.where(overlaps > 0.0001)[0]

        # mean_bbox = np.vstack((curr_candidate,rest_candidates[IDX]))
        # mean_bbox = np.mean(mean_bbox, axis=0).astype(np.uint32)
        # print mean_bbox
        results.append(curr_candidate)

        if len(np.where(overlaps < 0.0001)[0]) == 0:
            break
        candidates = rest_candidates[np.where(overlaps < 0.5)[0]]

    return np.array(results)

def detect(image_name, model, data, debug=False):
    # Load features from file for current image
    if not os.path.isfile(os.path.join(FEATURES_DIR, image_name + '.npy')):
        print 'ERROR: detect(): Features not found for ', image_name
        return
    
    features = np.load(os.path.join(FEATURES_DIR, image_name + '.npy'))
    gt_bboxes = data["test"]["gt"][image_name]
    features = features[len(gt_bboxes[0][0]):, :]

    result_bboxes = []
    result_idx = [] # Length of 3, each element contains a list of indices where each index corresponds to a detected bbox
    result_conf = []

    print 'Image:',image_name
    features = util.normalizeFeatures(features)
    X = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1) # Add the bias term
    y_hat = model.predict(X)
    confidence_scores = model.decision_function(X)

    all_regions = data["test"]["ssearch"][image_name]

    IDX = np.where(y_hat == 1)[0]

    # If our model detects no bboxes
    if len(IDX) == 0:
        return None, None

    print "IDX", IDX.shape
    candidates = all_regions[IDX, :]
    candidate_conf = confidence_scores[IDX]

    print "Candidate conf", candidate_conf.shape
    sorted_IDX = np.argsort(-1*candidate_conf)
    print "sorted idx", sorted_IDX
    candidates = candidates[sorted_IDX, :]
    result_bboxes.append(candidates)
    result_conf.append(candidate_conf[sorted_IDX][0])
    # Result idx is the idx of detected bboxes in the original 2000 proposals
    result_idx.append(IDX[sorted_IDX])

    #print candidates.shape[0]
    #candidates = nms(candidates)
    #print candidates.shape[0]

    #print '\tClass: %s --> %d / %d'%(classes[i], len(np.where(y_hat == 1)[0]), y_hat.shape[0])
    #util.displayImageWithBboxes(image_name, candidates[0:20,:])

    #print result_bboxes[np.argmax(result_conf)]
    #util.displayImageWithBboxes(image_name, np.array([result_bboxes[np.argmax(result_conf)]]))
    
    result_bboxes = np.array([result_bboxes])
    result_idx = np.array([result_idx])
    return result_idx, result_bboxes

def test(data, debug=False):
    # Load the models
    regression_models = None
    model_file_name = os.path.join(MODELS_DIR, 'bbox_ridge_reg.mdl')
    with open(model_file_name) as fp:
        regression_models = cp.load(fp)

    svm_models = dict()
    for c in [1,2,3]:
        model_file_name = os.path.join(MODELS_DIR, 'svm_%d_%s.mdl'%(c, FEATURE_LAYER))
        with open(model_file_name) as fp:
            svm_models[c] = cp.load(fp)

    # Test on the test set (or validation set)
    rcnn_result = dict()
    for i, image_name in enumerate(data["test"]["gt"].keys()):
        result = []
        features_file_name = os.path.join(FEATURES_DIR, image_name + '.npy')
        if not os.path.isfile(features_file_name):
            print 'ERROR: Missing features file \'%s\''%(features_file_name) 
        
        features = np.load(features_file_name)

        for c in [1,2,3]:
            # Run the detector
            proposal_ids, proposal_bboxes = detect(image_name, svm_models[c], data)
            # If no boxes were detected
            if proposal_ids is None:
                result.append([])
                continue

            # Run the regressor
            proposal_bboxes = np.squeeze(proposal_bboxes)
            proposal_features = np.squeeze(features[proposal_ids,:])
            print "Proposal features", proposal_features.shape
            print "Proposal boxes", proposal_bboxes.shape
            proposals = predictBoundingBox(regression_models[c], proposal_features, proposal_bboxes)
            # Run NMS
            proposals = nms(proposals)
            result.append(proposals)
        
        # Store the result
        rcnn_result[image_name] = result

        # Visualize images
        labels = np.array(data["test"]["gt"][image_name][0][0])
        all_gt_bboxes = np.array(data["test"]["gt"][image_name][1])
            
        for c in [1,2,3]:
            if len(result[0]) > 0:
                IDX = np.where(labels == c)[0]
                gt_bboxes = None
                if len(IDX) > 0:
                    gt_bboxes = all_gt_bboxes[IDX,:]
                util.displayImageWithBboxes(image_name, result[0], gt_bboxes, color=util.COLORS[c])
        

def main():
    print "Error: Do not run test_rcnn.py directly. You should use main.py."

if __name__ == '__main__':
    main()


