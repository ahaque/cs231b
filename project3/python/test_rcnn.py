import os
import time
import util
import cv
import det_eval

import numpy as np
import cPickle as cp

from sklearn import svm
from settings import *
from train_bbox import *

classes = ['CAR', 'CAT', 'PERSON']

def nms(data, debug=False):
    if debug: print data.shape[0],'->',
    candidates = np.copy(data)
    results = []

    while candidates.size != 0:
        # print candidates.shape
        curr_candidate = candidates[0, :]
        rest_candidates = candidates[1:, :]
        overlaps = util.computeOverlap(curr_candidate, rest_candidates)
        
        # IDX = np.where(overlaps > 0.3)[0]
        # mean_bbox = np.vstack((curr_candidate,rest_candidates[IDX]))
        # mean_bbox = np.mean(mean_bbox, axis=0).astype(np.uint32)
        # print mean_bbox
        results.append(curr_candidate)

        # if len(np.where(overlaps < 0.3)[0]) == 0:
        #     break
        candidates = rest_candidates[np.where(overlaps < 0.5)[0], :]
    if debug: print np.array(results).shape[0]
    return np.array(results)

def detect(image_name, model, data, debug=False):
    model, scaler = model
    # Load features from file for current image
    if not os.path.isfile(os.path.join(FEATURES_DIR, image_name + '.npy')):
        print 'ERROR: detect(): Features not found for ', image_name
        return
    
    features = np.load(os.path.join(FEATURES_DIR, image_name + '.npy'))

    # Get rid of GT_Bbox features from feature matrix
    gt_bboxes = data["gt"][image_name]
    num_gt_bboxes = 0
    if len(gt_bboxes[0]) != 0:
        num_gt_bboxes = len(gt_bboxes[0][0])
    # print '%s: Removing %d gt bboxes'%(image_name,num_gt_bboxes), data["gt"][image_name][0]
    features = features[num_gt_bboxes:, :]

    result_bboxes = []
    result_idx = [] # Length of 3, each element contains a list of indices where each index corresponds to a detected bbox
    result_features = []
    result_conf = []

    X, _ = util.normalizeAndAddBias(features, scaler)
    if debug:
        print 'X', X.shape
    # X = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1) # Add the bias term
    y_hat = model.predict(X)
    confidence_scores = model.decision_function(X)

    all_regions = data["ssearch"][image_name]

    IDX = np.where(y_hat == 1)[0]

    # If our model detects no bboxes
    if len(IDX) == 0:
        return None, None, None

    # print "IDX", IDX.shape
    candidates = all_regions[IDX, :]
    candidates_features = features[IDX, :]
    candidate_conf = confidence_scores[IDX]

    assert(np.min(candidate_conf) >= 0)

    # print "Candidate conf", candidate_conf.shape
    sorted_IDX = np.argsort(-1*candidate_conf)
    # print "sorted idx", sorted_IDX
    candidates = candidates[sorted_IDX, :]
    candidate_conf = candidate_conf[sorted_IDX]
    candidate_conf = np.reshape(candidate_conf, (candidate_conf.shape[0], 1))

    if debug:
        print 'Candidates:',candidates.shape
        print 'Confidences:',candidate_conf.shape

    candidates = np.hstack((candidates, candidate_conf))

    result_bboxes.append(candidates)
    result_features.append(candidates_features)
    # result_conf.append(candidate_conf)
    # Result idx is the idx of detected bboxes in the original 2000 proposals
    result_idx.append(IDX[sorted_IDX])
    

    #print candidates.shape[0]
    #candidates = nms(candidates)
    #print candidates.shape[0]

    #print '\tClass: %s --> %d / %d'%(classes[i], len(np.where(y_hat == 1)[0]), y_hat.shape[0])
    #util.displayImageWithBboxes(image_name, candidates[0:20,:])

    #print result_bboxes[np.argmax(result_conf)]
    #util.displayImageWithBboxes(image_name, np.array([result_bboxes[np.argmax(result_conf)]]))
    
    # result_bboxes = np.array([result_bboxes])
    # result_bboxes = np.array([result_bboxes])
    # result_idx = np.array([result_idx])

    # If we have a single candidate, the features will be of size (512) but we need it to be (1,512)    
    return IDX[sorted_IDX], candidates, candidates_features

def test(data, debug=False):
    # classes = ['CAR']
    class_ids = [1,2,3]
    # Load the models
    regression_models = None
    model_file_name = os.path.join(MODELS_DIR, 'bbox_ridge_reg.mdl')
    with open(model_file_name) as fp:
         regression_models = cp.load(fp)

    svm_models = dict()
    for c in class_ids:
        model_file_name = os.path.join(MODELS_DIR, 'svm_%d_%s.mdl'%(c, FEATURE_LAYER))
        with open(model_file_name) as fp:
            svm_models[c] = cp.load(fp)

    local_data = data['test']

    # Test on the test set (or validation set)
    num_images = len(local_data["gt"].keys())
    rcnn_result = dict()
    all_gt_bboxes = {c:[] for c in classes}
    all_pred_bboxes = {c:[] for c in classes}

    for i, image_name in enumerate(local_data["gt"].keys()):
        if i%25 == 0:
            print 'Processing Image #%d/%d'%(i+1, num_images)
        result = []
        # features_file_name = os.path.join(FEATURES_DIR, image_name + '.npy')
        # if not os.path.isfile(features_file_name):
        #     print 'ERROR: Missing features file \'%s\''%(features_file_name) 
        
        # features = np.load(features_file_name)

        for c in class_ids:
            # Run the detector
            proposal_ids, proposal_bboxes, proposal_features = detect(image_name, svm_models[c], local_data)
            
            # If no boxes were detected
            if proposal_ids is None:
                result.append(np.zeros((0,5)))
                continue

            # Run the regressor
            #proposal_bboxes = np.squeeze(proposal_bboxes)
            #proposal_features = np.squeeze(proposal_features)
            if len(proposal_bboxes.shape) == 1:
                proposal_bboxes = np.reshape(proposal_bboxes, (1,5))
            # print "Proposal boxes", proposal_bboxes.shape
            proposal_bboxes = predictBoundingBox(regression_models[c], proposal_features, proposal_bboxes)
            # print "Proposals after regression", proposal_bboxes.shape
            # Run NMS
            # print 'B:',np.max([proposal_bboxes[:,4]]),proposal_bboxes[0,4]
            # print proposal_bboxes
            proposals = nms(proposal_bboxes)
            # for b in proposal_bboxes.tolist():
            #     print "[%0.2f %0.2f %0.2f %0.2f] %0.4f"%tuple(b)
            # print 'A:',np.max([proposals[:,4]]),proposals[0,4]
            # print "Proposals after nms", proposals.shape
            # result.append(np.hstack((proposals, np.ones((proposals.shape[0], 1)))))
            result.append(proposals)
        
        # Store the result
        rcnn_result[image_name] = result

        # Visualize images
        num_gt_bboxes = 0
        if len(local_data["gt"][image_name][0]) != 0:
            num_gt_bboxes = len(local_data["gt"][image_name][0][0])
        if num_gt_bboxes > 0:
            labels = np.array(local_data["gt"][image_name][0][0])
            gt_bboxes = np.array(local_data["gt"][image_name][1])
        else:
            labels = np.array([])
            gt_bboxes = np.array([])

        for c in class_ids:
            top_bbox = np.array([])
            if result[c-1].shape[0] > 0:
                all_pred_bboxes[classes[c-1]].append(result[c-1])
                top_bbox = np.array(result[c-1][0, 0:4])
            else:
                all_pred_bboxes[classes[c-1]].append(np.zeros((0,5)))

            IDX = np.where(labels == c)[0]
            if len(IDX) > 0:
                gt_bboxes_curr_class = gt_bboxes[IDX,:]
                all_gt_bboxes[classes[c-1]].append(gt_bboxes_curr_class)
            else:
                gt_bboxes_curr_class = None
                all_gt_bboxes[classes[c-1]].append(np.zeros((0,4)))
            # util.displayImageWithBboxes(image_name, all_pred_bboxes[classes[c-1]][-1][0:2,0:4], gt_bboxes_curr_class, color=util.COLORS[c])
    
    evaluation = [(c,det_eval.det_eval(all_gt_bboxes[c], all_pred_bboxes[c])) for c in classes]
    total = 0.0
    print 'Average Precision'
    print '-----------------'
    for c,e in evaluation:
        ap, _, _ = e
        print c,ap
        print '%s: %0.4f'%(c, ap)
        total += ap
    print '%s: %0.4f'%('mAP', total/3)
        

def main():
    print "Error: Do not run test_rcnn.py directly. You should use main.py."

if __name__ == '__main__':
    main()


