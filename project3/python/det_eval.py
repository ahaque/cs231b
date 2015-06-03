import numpy as np
import util
try:
    import matlab.engine
    matlab_ready = True
except ImportError:
    print 'Warning: Matlab python engine not found. Will evaluate using python implementation.'
    matlab_ready = False

def det_eval_matlab(gt_bboxes, pred_bboxes):
    if not matlab_ready:
        print 'Matlab not found. Falling back to python implementation'
        return det_eval(gt_bboxes, pred_bboxes)

    eng = matlab.engine.start_matlab()
    eng.cd(r'../starter_code/', nargout=0)
    mat_gt_bboxes = []
    for x in gt_bboxes:
        # print x.shape
        mat_gt_bboxes.append(matlab.double(x.tolist()))

    mat_pred_bboxes = []
    for x in pred_bboxes:
        # print x.shape
        mat_pred_bboxes.append(matlab.double(x.tolist()))

    [ap, prec, rec] = eng.det_eval(mat_gt_bboxes, mat_pred_bboxes, nargout=3)
    prec = np.squeeze(np.array(prec))
    rec = np.squeeze(np.array(rec))

    return ap, prec, rec

def det_eval(gt_bboxes, pred_bboxes):
    # det_eval
    # Arguments:
    # pred_bboxes: Cell array of predicted bounding boxes for a single class.
    #   Each element corresponds to a single image and is a matrix of dimensions
    #   n x 5, where n is the number of bounding boxes. Each bounding box is
    #   represented as [x1 y1 x2 y2 score], where 'score' is the detection score.
    # gt_bboxes: Cell array of ground truth bounding boxes in the same format as
    # pred_bboxes, without the score component.
    #
    # Returns:
    #   ap: average precision
    #   prec: The precision at each point on the PR curve
    #   rec: The recall at each point on the PR curve.
    minoverlap = 0.5
    assert(len(gt_bboxes) == len(pred_bboxes))
    num_images = len(gt_bboxes)

    num_gt_boxes = np.sum([x.shape[0] for x in gt_bboxes])
    num_pred_boxes = np.sum([x.shape[0] for x in pred_bboxes])
    [all_pred_bboxes, image_ids] = sort_bboxes(pred_bboxes)

    tp = np.zeros((num_pred_boxes, 1))
    fp = np.zeros((num_pred_boxes, 1))

    detected_pred_bboxes = [None]*num_images
    detected = [None]*num_images

    for pred_ind in xrange(num_pred_boxes):
        # if mod(pred_ind, 4096) == 0:
        #     fprintf(mstring('calculating detection box %d of %d\\n'), pred_ind, num_pred_boxes)
        # end
        num_gt_im_boxes = len(gt_bboxes[image_ids[pred_ind]])
        if detected[image_ids[pred_ind][0]] is None:
            detected[image_ids[pred_ind][0]] = np.zeros((num_gt_im_boxes, 1))
        
        bb = all_pred_bboxes[pred_ind, :]

        overlaps = np.zeros((num_gt_im_boxes))
        
        for g in xrange(num_gt_im_boxes):
            gt_bbox = gt_bboxes[image_ids[pred_ind]][g,:]
            overlaps[g] = util.computeOverlap(gt_bbox, np.array([bb]))

        if overlaps.size != 0:
            jmax = np.argmax(overlaps, axis=0)
            ovmax = overlaps[jmax]
        else:
            ovmax = float('-inf')

        # assign detection as true positive/don't care/false positive
        im_detected = detected[image_ids[pred_ind]]
        if ovmax >= minoverlap:
            if not im_detected[jmax]:
                tp[pred_ind] = 1            #true positive
                detected[image_ids[pred_ind]][jmax] = 1
                if detected_pred_bboxes[image_ids[pred_ind]] is None:
                    detected_pred_bboxes[image_ids[pred_ind]]=np.array([bb])
                else:
                    detected_pred_bboxes[image_ids[pred_ind]]=util.stack([detected_pred_bboxes[image_ids[pred_ind]], bb]); 
            else:
                fp[pred_ind] = 1            #false positive (multiple detections)
        else:
            fp[pred_ind] = 1        #false positive


    # compute precision/recall
    npos = num_gt_boxes
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = 1.0*tp / npos
    prec = np.divide(tp, (fp + tp))
    ap = VOCap(rec, prec)

    return ap, prec, rec

def sort_bboxes(pred_bboxes):
    num_pred_boxes = np.sum([x.shape[0] for x in pred_bboxes])

    bboxes = util.stack([b for b in pred_bboxes if b.size > 0])
    if bboxes is None:
        return np.zeros((0, 5)), np.zeros((0, 1), dtype=int)
    image_ids = np.zeros((bboxes.shape[0], 1), dtype=int)

    # Concatenate them
    bbox_ind = 0
    for im_ind in xrange(len(pred_bboxes)):
        num_bbox_curr_image = pred_bboxes[im_ind].shape[0]
        image_ids[bbox_ind:bbox_ind + num_bbox_curr_image, 0] = im_ind
        bbox_ind = bbox_ind + num_bbox_curr_image

    # Sort
    IDX = np.argsort(-1*bboxes[:, -1])
    bboxes = bboxes[IDX, :]
    image_ids = image_ids[IDX]

    return bboxes, image_ids


def VOCap(rec, prec):
    mrec = np.concatenate((np.array([0]), rec, np.array([1])))
    mpre = np.concatenate((np.array([0]), prec, np.array([0])))

    for i in xrange(len(mpre)-2,-1,-1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1;
    ap = np.sum(np.multiply((mrec[i]-mrec[i-1]), mpre[i]))

    return ap
