import os
import time
import util
import cv

import numpy as np
import cPickle as cp

from sklearn import svm
from settings import *

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

        mean_bbox = np.vstack((curr_candidate,rest_candidates[IDX]))
        mean_bbox = np.mean(mean_bbox, axis=0).astype(np.uint32)
        # print mean_bbox
        results.append(mean_bbox)

        if len(np.where(overlaps < 0.0001)[0]) == 0:
            break
        candidates = rest_candidates[np.where(overlaps < 0.5)[0]]

    return np.array(results)

def detect(image_name, models, data, debug=False):
    # Load features from file for current image
    if not os.path.isfile(os.path.join(FEATURES_DIR, image_name + '.npy')):
        print 'Features not found for',image_name
        return
    
    features = np.load(os.path.join(FEATURES_DIR, image_name + '.npy'))
    gt_bboxes = data["test"]["gt"][image_name]
    features = features[len(gt_bboxes[0][0]):, :]

    result_bboxes = []
    result_conf = []

    print 'Image:',image_name
    for i, model in enumerate(models):
        X = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1) # Add the bias term
        y_hat = model.predict(X)
        confidence_scores = model.decision_function(X)

        all_regions = data["test"]["ssearch"][image_name]

        # print 'all_regions',all_regions.shape[0]
        # print 'features',features.shape[0]
        # print 'y_hat',y_hat.shape[0]
        # print 'confidence_scores',confidence_scores.shape[0]
        # assert(all_regions.shape[0] == features.shape[0])

        IDX = np.where(y_hat == 1)[0]
        candidates = all_regions[IDX, :]
        candidate_conf = confidence_scores[IDX]

        sorted_IDX = np.argsort(-1*candidate_conf)
        candidates = candidates[sorted_IDX, :]
        result_bboxes.append(candidates[0, :])
        result_conf.append(candidate_conf[sorted_IDX][0])

        #print candidates.shape[0]
        candidates = nms(candidates)
        #print candidates.shape[0]

        #print '\tClass: %s --> %d / %d'%(classes[i], len(np.where(y_hat == 1)[0]), y_hat.shape[0])
        #util.displayImageWithBboxes(image_name, candidates[0:20,:])
    
    print result_bboxes[np.argmax(result_conf)]
    util.displayImageWithBboxes(image_name, np.array([result_bboxes[np.argmax(result_conf)]]))

def test(data, debug=False):
    # Load the classifiers
    models = []
    for c in [1,2,3]:
        model_file_name = os.path.join(MODELS_DIR, 'svm_%d_%s.mdl'%(c, FEATURE_LAYER))
        with open(model_file_name) as fp:
            models.append(cp.load(fp))

    for i, image_name in enumerate(data["test"]["gt"].keys()):
        detect(image_name, models, data)