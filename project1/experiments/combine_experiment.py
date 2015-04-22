import numpy as np
import os
import sys

DIR = sys.argv[1] + '/'
filenames = os.listdir(DIR)
accuracy = None
jaccard = None
for image_file in filenames:
    data = np.genfromtxt(os.path.join(DIR, image_file), delimiter=',')
    if accuracy == None:
        accuracy = np.zeros(data.shape, dtype=float)
        jaccard = np.zeros(data.shape, dtype=float)
    if 'accuracy' in image_file:
        accuracy += data
    if 'jaccard' in image_file:
        jaccard += data

np.savetxt(DIR + 'combined_accuracy.csv', accuracy, delimiter=",")
np.savetxt(DIR + 'combined_jaccard.csv', jaccard, delimiter=",")