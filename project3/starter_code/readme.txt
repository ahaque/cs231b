describe what we give them
describe what they need to implement

===================================
Overview
===================================
In this project you'll be implementing the feature extraction, SVM training, and bounding box regression parts of an R-CNN. As input, you are provided images (750 training and 250 testing), ground truth bounding boxes to train and evaluate with, region proposals from selective search, and some sample code for how to extract features from regions using the caffe framework.


===================================
What we give you
===================================
readme.txt: This file!

detection_images.zip: All 1,000 images. Download these from the course project page.

train_ims.mat: Contains bounding box and class annotations for each of the 750 training image. It contains a struct array 'images'. Each element of 'images' corresponds to a single image and has the format:
  -fname: The filename of the image (within the folder detection_images.zip is extracted to)
  -bboxes: An n x 4 matrix of bounding boxes, where each row corresponds to a unique object instance. Bounding boxes are represented as [x1 y2 x2 y2].
  -classes: A vector of class labels in the same order as the bounding boxes. Class 1 is 'car', class 2 is 'cat', and class 3 is 'person'.

test_ims.mat: Bounding box and class annotations for each of the 250 testing images, in the same format as train_ims.mat. Ordinarily, we would also have a validation set for parameter tuning, but for the purposes of this class we're skipping that.

ssearch_train.mat: Selective search regions for each training image. It contains a cell array 'ssearch_boxes', where each element contains an n x 4 matrix of region proposal bounding boxes returned by selective search.

ssearch_test.mat: Selective search regions for each testing image in the same format as for training.

extract_cnn_feat_demo.m: An example script that extracts CNN features from a few regions in an image, using the caffe framework.

Makefile.config.rye: A makefile for use on the rye farmshare machines (see section: Caffe).

ilsvrc_2012_mean.mat: A file containing the mean image in the ILSVRC 2012 dataset, used for extracting CNN features.

cnn_deploy.prototxt: A file describing the CNN architecture used to extract features. It will extract 512-dimensional features.

cnn512.caffemodel: The trained CNN used for extracting features. 

display_bbox.m: Displays a bounding box on an image.

det_eval.m: A function you can use to calculate, precision, recall, and average precision for one class.

boxoverlap.m: A function that calculates the IOU of a set of bounding boxes with another bounding box.


===================================
What you need to implement
===================================
(Note: All of these files contain comments including more details and things to watch out for


extract_region_feats.m: Extract features around each region proposal

train_rcnn.m: Trains an R-CNN based on the extracted features

train_bbox_reg.m: Trains a bounding box regressor

test_rcnn.m: Run the R-CNN on test images and evaluate them.



===================================
Caffe
===================================
Caffe is a framework for convolutional neural networks. The version of caffe we'll be using for this project is rc2, available at https://github.com/BVLC/caffe/archive/rc2.zip

Installation instructions for caffe are available at http://caffe.berkeleyvision.org/installation.html.  There are guides for Ubuntu, OS X, and Red Hat/CentOS/Fedora. The installation instructions they provide are reasonably good, and we recommend following them closely if you're installing on your own machine.

If you don't have a machine capable of running caffe, or just don't want do deal with the hassle, use the rye01 or rye02 machines on farmshare. To do this, follow the installation instructions as normal, with the following two exceptions:
-Use the provided Makefile.config.rye as your Makefile.config
-Before running caffe, run 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-6.0/lib64/:/afs/ir/users/j/k/jkrause/cs231b/caffe/include/lmdb/mdb/libraries/liblmdb:/afs/ir/users/j/k/jkrause/cs231b/caffe/include/hdf5-1.8.14/out/lib'.  This will include a few libraries that aren't on rye by default. If you're using bash, it might be useful to put this in your .bashrc file in your home directory. If you have any difficulty installing on rye (we've tested on rye01), save yourself a lot of time and ask on Piazza :)

When compiling, you'll need to run both 'make all' to make the basic caffe files, and 'make matcaffe' to make the matlab interface.
