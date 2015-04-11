import sys
import pymaxflow
import pylab
import numpy as np

eps = 0.01

im = pylab.imread(sys.argv[1]).astype(np.float32)

indices = np.arange(im.size).reshape(im.shape).astype(np.int32)
g = pymaxflow.PyGraph(im.size, im.size * 3)

g.add_node(im.size)

# adjacent
diffs = np.abs(im[:, 1:] - im[:, :-1]).ravel() + eps
e1 = indices[:, :-1].ravel()
e2 = indices[:, 1:].ravel()
g.add_edge_vectorized(e1, e2, diffs, 0 * diffs)

# adjacent up
diffs = np.abs(im[1:, 1:] - im[:-1, :-1]).ravel() + eps
e1 = indices[1:, :-1].ravel()
e2 = indices[:-1, 1:].ravel()
g.add_edge_vectorized(e1, e2, diffs, 0 * diffs)

# adjacent down
diffs = np.abs(im[:-1, 1:] - im[1:, :-1]).ravel() + eps
e1 = indices[:-1, :-1].flatten()
e2 = indices[1:, 1:].ravel()
g.add_edge_vectorized(e1, e2, diffs, 0 * diffs)

# link to source/sink
g.add_tweights_vectorized(indices[:, 0], (np.ones(indices.shape[0]) * 1.e9).astype(np.float32), np.zeros(indices.shape[0], np.float32))
g.add_tweights_vectorized(indices[:, -1], np.zeros(indices.shape[0], np.float32), (np.ones(indices.shape[0]) * 1.e9).astype(np.float32))

print "calling maxflow"
g.maxflow()

out = g.what_segment_vectorized()
pylab.imshow(out.reshape(im.shape))
pylab.show()
