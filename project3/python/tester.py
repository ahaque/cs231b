import util
import numpy as np
import det_eval
import random
try:
    import matlab.engine
    run_eval_test = True
except ImportError:
    print 'Warning: Matlab python engine not found. Some tests will not be run.'
    run_eval_test = False

TOLERANCE = 1e-7

def main():
    bbox_overlap_test()
    if run_eval_test:
        eval_test()

def bbox_overlap_test():
    print 'Testing BBOX overlap...'
    ########## BBOX overlap test ##########
    region_bbox = np.array([50, 50, 100, 100])
    gt_bboxes = np.array([[75, 25, 125, 75], # 0.14
                        [50, 200, 100, 250], # 0 
                        [50, 50, 100, 100] # 1
                        ])
    out = util.computeOverlap(region_bbox, gt_bboxes)
    assert(np.abs(out[0] - 0.14935926) <= TOLERANCE)
    assert(np.abs(out[1] - 0) <= TOLERANCE)
    assert(np.abs(out[2] - 1) <= TOLERANCE)
    print "Passed all tests."

def eval_test(show_output=False):
    print 'Testing evalutation code...'
    num_images = random.randint(4,10)
    gt_bboxes = []
    pred_bboxes = []

    for i in xrange(num_images):
        if show_output: print 'Image %d'%(i+1)
        num_gt_bboxes = random.randint(0,5)
        num_pred_bboxes = random.randint(0,5)

        gt_bboxes.append([])
        for g in xrange(num_gt_bboxes):
            start_x = random.randint(0,300)
            start_y = random.randint(0,300)
            end_x = random.randint(start_x + 10,500)
            end_y = random.randint(start_y + 10,500)
            gt_bboxes[-1].append([start_x, start_y, end_x, end_y])
            if show_output: print 'GT: [%d,%d,%d,%d]'%(start_x, start_y, end_x, end_y)

        pred_bboxes.append([])
        for g in xrange(num_gt_bboxes):
            start_x = random.randint(0,300)
            start_y = random.randint(0,300)
            end_x = random.randint(start_x + 10,500)
            end_y = random.randint(start_y + 10,500)
            score = random.random()
            pred_bboxes[-1].append([start_x, start_y, end_x, end_y, score])
            if show_output: print 'PRED: [%d,%d,%d,%d]'%(start_x, start_y, end_x, end_y)
    
    # gt_bboxes = [[[100,91,160,500],
    #                     [268,176,307,467],
    #                     [62,112,491,219],
    #                     [122,230,394,332],
    #                     [50,84,319,109]],
    #             [[253,225,308,262],
    #                     [145,182,425,329],
    #                     [220,96,424,407],
    #                     [12,194,128,365]]]
    # pred_bboxes = [[[63,2,84,494, 1.0],
    #                     [80,73,403,180, 1.0],
    #                     [34,282,445,487, 1.0],
    #                     [92,129,218,311, 1.0],
    #                     [299,19,468,94, 1.0],
    #                     [264,56,461,371, 1.0]],
    #             [[204,47,247,175, 1.0],
    #                     [227,180,374,373, 1.0],
    #                     [224,153,425,493, 1.0],
    #                     [192,159,467,218, 1.0],
    #                     [262,176,473,343, 1.0]]]

    ap, prec, rec = det_eval.det_eval([np.array(x) for x in gt_bboxes], [np.array(x) for x in pred_bboxes])
    if show_output:
        print 'Python Average precision:', ap
        print 'Python Precision:',prec
        print 'Python Recall:',rec

    ### Run Matlab code ###
    # MATLAB_PATH = '/Applications/MATLAB_R2015a.app/bin/matlab'
    # OPTIONS = '-nodisplay -nosplash -nodesktop -r "det_eval({[100,91,160,500;268,176,307,467;62,112,491,219;122,230,394,332;50,84,319,109],[253,225,308,262;145,182,425,329;220,96,424,407;12,194,128,365]}, {[63,2,84,494, 1.0; 80,73,403,180, 1.0; 34,282,445,487, 1.0; 92,129,218,311, 1.0; 299,19,468,94, 1.0; 264,56,461,371, 1.0],[204,47,247,175, 1.0;227,180,374,373, 1.0;224,153,425,493, 1.0;192,159,467,218, 1.0;262,176,473,343, 1.0]}); exit"'
    # command = [MATLAB_PATH, '-nodisplay', '-nosplash', '-nodesktop', '-r','det_eval({[100,91,160,500;268,176,307,467;62,112,491,219;122,230,394,332;50,84,319,109],[253,225,308,262;145,182,425,329;220,96,424,407;12,194,128,365]}, {[63,2,84,494, 1.0; 80,73,403,180, 1.0; 34,282,445,487, 1.0; 92,129,218,311, 1.0; 299,19,468,94, 1.0; 264,56,461,371, 1.0],[204,47,247,175, 1.0;227,180,374,373, 1.0;224,153,425,493, 1.0;192,159,467,218, 1.0;262,176,473,343, 1.0]}); exit']
    # process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # output = process.communicate()[0]
    # print output
    eng = matlab.engine.start_matlab()
    eng.cd(r'../starter_code/', nargout=0)
    [m_ap, m_prec, m_rec] = eng.det_eval([matlab.double(x) for x in gt_bboxes], [matlab.double(x) for x in pred_bboxes], nargout=3)
    m_prec = np.squeeze(np.array(m_prec))
    m_rec = np.squeeze(np.array(m_rec))

    if show_output:
        print 'Matlab Average precision:', m_ap
        print 'Matlab Precision:', m_prec
        print 'Matlab Recall:', m_rec

    assert(np.abs(ap - m_ap) <= TOLERANCE)
    assert(prec.size == m_prec.size)
    for i in xrange(prec.size):
        assert(np.abs(prec[i] - m_prec[i]) <= TOLERANCE)
    assert(rec.shape[0] == m_rec.shape[0])
    for i in xrange(rec.size):
        assert(np.abs(rec[i] - m_rec[i]) <= TOLERANCE)
    
    print "Passed all tests."
     

if __name__ == '__main__':
    main()