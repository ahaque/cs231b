import os
import util
import sys

from train_rcnn import *
from test_rcnn import *

def process_results(hyperparams, parent_dir):
    print 'Processing results...'

    with open(os.path.join(parent_dir,'tuning_results.csv'), 'w') as wp:
        print >> wp, 'Cost, Bias, Positive Weight, Negative memory, \
        CAR Train Pos accuracy, CAR Train Neg accuracy, CAR Train total accuracy, \
        CAR Validation Pos accuracy, CAR Validation Neg accuracy, CAR Validation total accuracy, \
        CAT Train Pos accuracy, CAT Train Neg accuracy, CAT Train total accuracy, \
        CAT Validation Pos accuracy, CAT Validation Neg accuracy, CAT Validation total accuracy, \
        PERSON Train Pos accuracy, PERSON Train Neg accuracy, PERSON Train total accuracy, \
        PERSON Validation Pos accuracy, PERSON Validation Neg accuracy, PERSON Validation total accuracy, \
        CAR AP, CAT_AP, PERSON_AP, mAP'

    for i,hyperparam in enumerate(hyperparams):
        c, b, w, n = hyperparam
        if w != 'auto':
            w_str = '%04d'%w[1]
        else:
            w_str = w
        c_str = '%0.5f'%c
        tmp = c_str.split('.')
        c_str = ('000' + tmp[0])[-3:] + '-' + tmp[1]
        folder_name = 'svm_%s_%02d_%s_%05d'%(c_str,b,w_str,n)
        output_path = os.path.join(parent_dir, folder_name)

        with open(os.path.join(output_path, 'hyperparams_result.txt'), 'r') as fp:
            c_f = float(fp.next().strip().split(':')[-1])
            b_f = int(fp.next().strip().split(':')[-1])

            t_w = fp.next().strip().split(':')
            if len(t_w) == 2:
                w_f = 'auto'
            else:
                w_f = str(int(t_w[-1][:-1]))
            n_f = int(fp.next().strip().split(':')[-1])

            # print c,c_f,' ',b,b_f,' ',w,w_f,' ',n,n_f,' '
            fp.next() # Read ---------------
            fp.next() # Read Class: CAR
            car_train_pos, car_train_neg, car_train_tot = tuple([float(x) for x in fp.next().strip().split(':')[1].split(',')])
            car_val_pos, car_val_neg, car_val_tot = tuple([float(x) for x in fp.next().strip().split(':')[1].split(',')])

            fp.next() # Read \n
            fp.next() # Read \n
            fp.next() # Read Class: CAT
            cat_train_pos, cat_train_neg, cat_train_tot = tuple([float(x) for x in fp.next().strip().split(':')[1].split(',')])
            cat_val_pos, cat_val_neg, cat_val_tot = tuple([float(x) for x in fp.next().strip().split(':')[1].split(',')])

            fp.next() # Read \n
            fp.next() # Read \n
            fp.next() # Read Class: PERSON
            person_train_pos, person_train_neg, person_train_tot = tuple([float(x) for x in fp.next().strip().split(':')[1].split(',')])
            person_val_pos, person_val_neg, person_val_tot = tuple([float(x) for x in fp.next().strip().split(':')[1].split(',')])


            fp.next() # Read \n
            fp.next() # Read \n
            fp.next() # Read Class: Average Precision:
            car_ap = float(fp.next().strip().split(':')[-1])
            cat_ap = float(fp.next().strip().split(':')[-1])
            person_ap = float(fp.next().strip().split(':')[-1])
            mAP = float(fp.next().strip().split(':')[-1])

            with open(os.path.join(parent_dir,'tuning_results.csv'), 'a') as wp:
                print >> wp, '%f,%d,%s,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f'%(
                    c_f, b_f, w_f, n_f,
                    car_train_pos, car_train_neg, car_train_tot, car_val_pos, car_val_neg, car_val_tot,
                    cat_train_pos, cat_train_neg, cat_train_tot, cat_val_pos, cat_val_neg, cat_val_tot,
                    person_train_pos, person_train_neg, person_train_tot, person_val_pos, person_val_neg, person_val_tot,
                    car_ap, cat_ap, person_ap, mAP)

 
def tune_params(hyperparams, num_procs, process_id, parent_dir):
    data = {}
    data["train"] = util.readMatrixData("train")
    data["test"] = util.readMatrixData("test")

    for i,hyperparam in enumerate(hyperparams):
        if (i%num_procs) != process_id:
            continue
        c, b, w, n = hyperparam
        if w != 'auto':
            w_str = '%04d'%w[1]
        else:
            w_str = w
        c_str = '%0.5f'%c
        tmp = c_str.split('.')
        c_str = ('000' + tmp[0])[-3:] + '-' + tmp[1]

        folder_name = 'svm_%s_%02d_%s_%05d'%(c_str,b,w_str,n)
        output_path = os.path.join(parent_dir, folder_name)

        # For each object class
        models = []
        threads = []

        # Make directory creating thread safe
        # Sometimes many threads don't fine MODELS_DIR
        # but only one can create it
        try:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        except:
            pass

        with open(os.path.join(output_path, 'hyperparams_result.txt'), 'w') as fp:
            print >> fp, "C:", c
            print >> fp, "Bias:", b
            print >> fp, "Positive Class weight:", w
            print >> fp, "Negative examples memory:", n
            print >> fp, "-------------------------------------"

        classes = ['CAR', 'CAT', 'PERSON']
        for class_id in [1,2,3]:
            # Train a SVM for this class
            model, scaler, train_acc, val_acc = trainClassifierForClass(data, class_id, memory_size=n, C=c, B=b, W=w, evaluate=True, output_dir=output_path)
            model = (model, scaler)
            model_file_name = os.path.join(output_path, 'svm_%d_%s.mdl'%(class_id, FEATURE_LAYER))
            with open(model_file_name, 'w') as fp:
                cp.dump(model, fp)

            with open(os.path.join(output_path, 'hyperparams_result.txt'), 'a') as fp:
                print >> fp, "Class:", classes[class_id-1]
                print >> fp, "Training Accuracy (Pos, Neg, Total): %0.4f, %0.4f, %0.4f"%train_acc 
                print >> fp, "Validation Accuracy (Pos, Neg, Total): %0.4f, %0.4f, %0.4f"%val_acc
                print >> fp, "\n"

        evaluation, mAP = test(data, models_dir=output_path)

        with open(os.path.join(output_path, 'hyperparams_result.txt'), 'a') as fp:
            print >> fp, "Average Precision:"
            for c,e in evaluation:
                ap, _, _ = e
                print >> fp, '%s: %0.4f'%(c, ap)
            print >> fp, '%s: %0.4f'%('mAP', mAP)       

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print 'Usage: python hyperparam_tuning.py --process'
        print '\t\t\t or'
        print 'Usage: python hyperparam_tuning.py <num-procs> <process-id>'
        sys.exit(1)

    parent_dir = '../models/hyperparams/'
    C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    B = [1, 5, 10, 50]
    W = [{1:1}, {1:2}, {1:5}, {1:6}, {1:10}, {1:50}, 'auto']
    N = [5000, 10000, 20000, 30000, 50000]

    hyperparams = []
    for c in C:
        for b in B:
            for w in W:
                for n in N:
                    hyperparams.append((c,b,w,n))

    if len(sys.argv) == 3:
        num_procs = int(sys.argv[1])
        process_id = int(sys.argv[2])
        tune_params(hyperparams, num_procs, process_id, parent_dir)
    else:
        process_results(hyperparams, parent_dir)

if __name__ == '__main__':
    main()
