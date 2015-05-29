################################################################
# BEGIN REQUIRED INPUT PARAMETERS

# For all DIRs and paths, the trailing slash does not matter
global ML_DIR
ML_DIR = "../ml" # ML_DIR contains matlab matrix files and caffe model
global IMG_DIR
IMG_DIR = "../images" # IMG_DIR contains all images
global FEATURES_DIR
FEATURES_DIR = "../features/cnn512_fc6" # FEATURES_DIR stores the region features for each image
global MODELS_DIR
MODELS_DIR = '../models/iterative/'

global MODEL_DEPLOY
MODEL_DEPLOY = "../ml/cnn_deploy.prototxt" # CNN architecture file
global MODEL_SNAPSHOT
MODEL_SNAPSHOT = "../ml/cnn512.caffemodel" # CNN weights

# global MODEL_SNAPSHOT
# MODEL_SNAPSHOT = "../ml/VGG_ILSVRC_16_layers.caffemodel"
# global MODEL_DEPLOY
# MODEL_DEPLOY = "../ml/VGG_ILSVRC_16_layers_deploy.prototxt"

global GPU_MODE
GPU_MODE = True # Set to True if using GPU

# CNN Batch size. Depends on the hardware memory
# NOTE: This must match exactly value of line 3 in the deploy.prototxt file
global CNN_BATCH_SIZE
CNN_BATCH_SIZE = 250 # CNN batch size
global CNN_INPUT_SIZE
CNN_INPUT_SIZE = 227 # Input size of the CNN input image (after cropping)

global CONTEXT_SIZE
CONTEXT_SIZE = 16 # Context or 'padding' size around region proposals in pixels

# The layer and number of features to use from that layer
# Check the deploy.prototxt file for a list of layers/feature outputs
global FEATURE_LAYER
FEATURE_LAYER = "fc6_ft"
global NUM_CNN_FEATURES
NUM_CNN_FEATURES = 512

global NUM_CLASSES
NUM_CLASSES = 3 # Number of object classes

global INDICATOR_PAD_SIZE
INDICATOR_PAD_SIZE = 100
global POSITIVE_THRESHOLD
POSITIVE_THRESHOLD = 0.7
global NEGATIVE_THRESHOLD
NEGATIVE_THRESHOLD = 0.3
global BBOX_POSITIVE_THRESHOLD
BBOX_POSITIVE_THRESHOLD = 0.6

# END REQUIRED INPUT PARAMETERS
################################################################