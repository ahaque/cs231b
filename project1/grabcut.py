from matplotlib.patches import Rectangle
from graph_tool.all import *
from gmm import GMM
import matplotlib.pyplot as plt
import numpy as np
import argparse

try:
    import third_party.pymaxflow.pymaxflow as pymaxflow
except ImportError:
    import pymaxflow

import time
import sys

# Global constants
gamma = 50
beta = 1e-5 # TODO: optimize beta Boykov and Jolly 2001

SOURCE = -1
SINK = -2

# tic-toc
start_time = 0

# If energy changes less than CONVERGENCE_CRITERIA percent from the last iteration
# we will terminate
CONVERGENCE_CRITERON = 0.02

def tic():
    global start_time
    start_time = time.time()
def toc(task_label):
    global start_time
    print "%s took %0.4f s"%(task_label, time.time() - start_time)


# get_args function
# Intializes the arguments parser and reads in the arguments from the command
# line.
# 
# Returns: args dict with all the arguments
def get_args():
    parser = argparse.ArgumentParser(
        description='Implementation of the GrabCut algorithm.')
    parser.add_argument('-i','--image_file', 
        nargs=1, help='Input image name along with its relative path')
    parser.add_argument('-b','--bbox', nargs=4, default=None,
        help='Bounding box of the foreground object')

    return parser.parse_args()

# load_image function
# Loads an image using matplotlib's built in image reader
# Note: Requires PIL (python imaging library) to be installed if the image is
# not a png
# 
# Returns: img matrix with the contents of the image
def load_image(img_name):
    print 'Reading %s...' % img_name
    return plt.imread(img_name)

# RectSelector class
# Enables prompting user to select a rectangular area on a given image
class RectSelector:
    def __init__(self, ax):
        self.button_pressed = False
        self.start_x = 0
        self.start_y = 0
        self.canvas = ax.figure.canvas
        self.ax = ax
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.rectangle = []
    
    # Handles the case when the mouse button is initially pressed
    def on_press(self,event):
        self.button_pressed = True

        # Save the initial coordinates
        self.start_x = event.xdata
        self.start_y = event.ydata
        selected_rectangle = Rectangle((self.start_x,self.start_y),
                        width=0,height=0, fill=False, linestyle='dashed')

        # Add new rectangle onto the canvas
        self.ax.add_patch(selected_rectangle)
        self.canvas.draw()

    # Handles the case when the mouse button is released
    def on_release(self,event):
        self.button_pressed = False

        # Check if release happened because of mouse moving out of bounds,
        # in which case we consider it to be an invalid selection
        if event.xdata == None or event.ydata == None:
            return
        x = event.xdata
        y = event.ydata

        width = x - self.start_x
        height = y - self.start_y
        selected_rectangle = Rectangle((self.start_x,self.start_y),
                    width,height, fill=False, linestyle='solid')

        # Remove old rectangle and add new one
        self.ax.patches = []
        self.ax.add_patch(selected_rectangle)
        self.canvas.draw()
        xs = sorted([self.start_x, x])
        ys = sorted([self.start_y, y])
        self.rectangle = [xs[0], ys[0], xs[1], ys[1]]

        # Unblock plt
        plt.close()

    def on_move(self,event):
        # Check if the mouse moved out of bounds,
        # in which case we do not care about its position
        if event.xdata == None or event.ydata == None:
            return

        # If the mouse button is pressed, we need to update current rectangular
        # selection
        if self.button_pressed:
            x = event.xdata
            y = event.ydata

            width = x - self.start_x
            height = y - self.start_y
            
            selected_rectangle = Rectangle((self.start_x,self.start_y),
                            width,height, fill=False, linestyle='dashed')

            # Remove old rectangle and add new one
            self.ax.patches = []
            self.ax.add_patch(selected_rectangle)
            self.canvas.draw()

def get_user_selection(img):
    if img.shape[2] != 3:
        print 'This image does not have all the RGB channels, you do not need to work on it.'
        return
    
    # Initialize rectangular selector
    fig, ax = plt.subplots()
    selector = RectSelector(ax)
    
    # Show the image on the screen
    ax.imshow(img)
    plt.show()

    # Control reaches here once the user has selected a rectangle, 
    # since plt.show() blocks.
    # Return the selected rectangle
    return selector.rectangle

def initialization(img, bbox, debug=False):
    xmin, ymin, xmax, ymax = bbox
    height, width, _ = img.shape
    alpha = np.zeros((height, width), dtype=np.int8)

    for h in xrange(height): # Rows
        for w in xrange(width): # Columns
            if (w > xmin) and (w < xmax) and (h > ymin) and (h < ymax):
                # Foreground
                alpha[h,w] = 1
            else:
                # Background
                alpha[h,w] = 0

    foreground_gmm = GMM(5)
    background_gmm = GMM(5)

    foreground_gmm.initialize_gmm(img[alpha==1,:])
    background_gmm.initialize_gmm(img[alpha==0,:])

    if debug:
        plt.imshow(alpha*265)
        plt.show()
        for i in xrange(alpha.shape[0]):
            for j in xrange(alpha.shape[1]):
                print alpha[i,j],
            print ''

    return alpha, foreground_gmm, background_gmm

# Currently creates a meaningless graph
def create_graph(img, neighbor_list):
    num_neighbors = 8

    num_nodes = img.shape[0]*img.shape[1] + 2
    num_edges = img.shape[0]*img.shape[1]*num_neighbors

    g = pymaxflow.PyGraph(num_nodes, num_edges)

    # Creating nodes
    g.add_node(num_nodes-2)

    return g

# alpha,k - specific values
def get_pi(alpha, k, gmms):
    return gmms[alpha].weights[k]

def get_cov_det(alpha, k, gmms):
    return gmms[alpha].gaussians[k].sigma_det

def get_mean(alpha, k, gmms):
    return gmms[alpha].gaussians[k].mean

def get_cov_inv(alpha, k, gmms):
    return gmms[alpha].gaussians[k].sigma_inv

# Its not log_prob but we are calling it that for convinience
def get_log_prob(alpha, k, gmms, z_pixel):
    term = (z_pixel - get_mean(alpha, k, gmms))
    return 0.5 * np.dot(np.dot(term.T, get_cov_inv(alpha, k, gmms)), term)

def get_energy(alpha, k, gmms, z, smoothness_matrix):
    # Compute U
    U = 0
    for h in xrange(z.shape[0]):
        for w in xrange(z.shape[1]):
            U += -np.log(get_pi(alpha[h,w], k[h,w], gmms)) \
                + 0.5 * np.log(get_cov_det(alpha[h,w], k[h,w], gmms)) \
                + get_log_prob(alpha[h,w], k[h,w], gmms, z[h,w,:])

    # Compute V
    V = 0
    for h in xrange(z.shape[0]):
        for w in xrange(z.shape[1]):
            # Loop through neighbors
            for (nh, nw) in smoothness_matrix[(h,w)].keys():
                if alpha[h,w] != alpha[nh,nw]:
                    V += smoothness_matrix[(h,w)][(nh, nw)]
    V = gamma * V

    return U + V

def get_unary_energy_vectorized(alpha, k, gmms, pixels, debug=False):
    pi_base = gmms[alpha].weights
    pi = pi_base[k].reshape(pixels.shape[0])

    dets_base = np.array([gmms[alpha].gaussians[i].sigma_det for i in xrange(len(gmms[alpha].gaussians))])
    dets = dets_base[k].reshape(pixels.shape[0])

    means_base = np.array([gmms[alpha].gaussians[i].mean for i in xrange(len(gmms[alpha].gaussians))])
    means = np.swapaxes(means_base[k], 1, 2)
    means = means.reshape((means.shape[0:2]))

    cov_base = np.array([gmms[alpha].gaussians[i].sigma_inv for i in xrange(len(gmms[alpha].gaussians))])
    cov = np.swapaxes(cov_base[k], 1, 3)
    cov = cov.reshape((cov.shape[0:3]))

    term = pixels - means
    middle_matrix = (np.multiply(term, cov[:, :, 0]) + np.multiply(term, cov[:, :, 1]) +np.multiply(term, cov[:, :, 2]))
    log_prob = np.sum(np.multiply(middle_matrix, term), axis=1)

    if debug:
        print pi.shape
        print dets.shape
        print log_prob.shape

    return -np.log(pi) \
        + 0.5 * np.log(dets) \
        + 0.5 * log_prob

def get_unary_energy(alpha, k, gmms, z, pixel):
    h,w = pixel
    return -np.log(get_pi(alpha, k[h,w], gmms)) \
            + 0.5 * np.log(get_cov_det(alpha, k[h,w], gmms)) \
            + get_log_prob(alpha, k[h,w], gmms, z[h,w,:])

def get_pairwise_energy(alpha, pixel_1, pixel_2, smoothness_matrix):
    (h,w) = pixel_1
    (nh,nw) = pixel_2
    V = 0
    if alpha[h,w] != alpha[nh,nw]:
        # print 'Pairwise',(h,w), (nh,nw)
        V = smoothness_matrix[(h,w)][(nh, nw)]

    return gamma *V

def compute_beta(z, debug=False):
    accumulator = 0
    m = z.shape[0]
    n = z.shape[1]

    for h in xrange(m-1):
        if debug: print 'Computing row', h
        for w in xrange(n):
            accumulator += np.linalg.norm(z[h,w,:] - z[h+1,w,:])**2

    for h in xrange(m):
        if debug: print 'Computing row', h
        for w in xrange(n-1):
            accumulator += np.linalg.norm(z[h,w,:] - z[h,w+1,:])**2

    num_comparisons = float(2*(m*n) - m - n)

    beta = (2*(accumulator/num_comparisons))**-1

    return beta
            

def compute_smoothness(z, debug=False):
    EIGHT_NEIGHBORHOOD = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    FOUR_NEIGHBORHOOD = [(-1,0), (0,-1), (0,1), (1,0)]

    height, width, _ = z.shape
    global beta
    smoothness_matrix = dict()

    beta = compute_beta(z)
    print 'beta',beta

    for h in xrange(height):
        if debug:
            print 'Computing row',h
        for w in xrange(width):
            if (h,w) not in smoothness_matrix:
                smoothness_matrix[(h,w)] = dict()
            for hh,ww in FOUR_NEIGHBORHOOD:
                nh, nw = h + hh, w + ww
                if nw < 0 or nw >= width:
                    continue
                if nh < 0 or nh >= height:
                    continue

                if (nh,nw) not in smoothness_matrix:
                    smoothness_matrix[(nh,nw)] = dict()

                if (h,w) in smoothness_matrix[(nh,nw)]:
                    continue

                smoothness_matrix[(h,w)][(nh, nw)] = \
                    np.exp(-1 * beta * np.linalg.norm(z[h,w,:] - z[nh,nw,:]))
                smoothness_matrix[(nh,nw)][(h,w)] = smoothness_matrix[(h,w)][(nh, nw)]

                if debug:
                    print (h,w),'->',(nh,nw),":",z[h,w,:], z[nh,nw,:], smoothness_matrix[(h,w)][(nh, nw)]

    return smoothness_matrix

def grabcut(img, bbox, debug=False, drawImage=False):
    print 'Initializing gmms'
    tic()
    alpha, foreground_gmm, background_gmm = initialization(img, bbox)
    k = np.zeros((img.shape[0],img.shape[1]), dtype=int)
    toc('Initializing gmms')

    print 'Computing smoothness matrix...'
    tic()
    smoothness_matrix = compute_smoothness(img, debug=False)
    toc('Computing smoothness matrix')

    global SOURCE
    global SINK
    
    FOREGROUND = 1
    BACKGROUND = 0
    previous_energy = sys.float_info.max
    print 'Starting EM'
    for iteration in xrange(1,11):
        print 'Iteration %d'%iteration
        # 1. Assigning GMM components to pixels
        tic()
        pixels = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
        foreground_components = foreground_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))
        background_components = background_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))

        k[alpha==1] = foreground_components[alpha==1]
        k[alpha==0] = background_components[alpha==0]

        # for h in xrange(img.shape[0]):
        #     for w in xrange(img.shape[1]):
        #         if alpha[h,w] == 1:
        #             k[h,w] = foreground_gmm.get_component(img[h,w,:])
        #         else:
                    # k[h,w] = background_gmm.get_component(img[h,w,:])

        toc('Assigning GMM components')

        # COLORS = [[255,0,0],[0,255,0], [0,0,255], [255,255,0], [255,0,255]]
        # res = np.zeros(img.shape, dtype=np.uint8)
        # for h in xrange(img.shape[0]):
        #     for w in xrange(img.shape[1]):
        #         res[h,w,:] = COLORS[k[h,w]] 
        # plt.imshow(res)
        # plt.show()

        # 2. Learn GMM parameters
        tic()
        foreground_assignments = -1*np.ones(k.shape)
        foreground_assignments[alpha==1] = k[alpha==1]

        background_assignments = -1*np.zeros(k.shape)
        background_assignments[alpha==0] = k[alpha==0]

        foreground_gmm.update_components(img, foreground_assignments)
        background_gmm.update_components(img, background_assignments)

        toc('Updating GMM parameters')

        # 3. Estimate segmentation using min cut
        # Update weights
        # Compute Unary weights

        tic()
        graph = create_graph(img, smoothness_matrix)
        theta = (background_gmm, foreground_gmm)
        total_energy = 0

        k_flattened =  k.reshape((img.shape[0]*img.shape[1], 1))
        foreground_energies = get_unary_energy_vectorized(1, k_flattened, theta, pixels)
        background_energies = get_unary_energy_vectorized(0, k_flattened, theta, pixels)

        unary_total_time = 0.0
        pairwise_time = 0.0
        for h in xrange(img.shape[0]):
            for w in xrange(img.shape[1]):
                index = h*img.shape[1] + w
                # If pixel is outside of bounding box, assign large unary energy
                # See Jon's lecture notes on GrabCut, slide 11
                if w < bbox[0] or w > bbox[2] or h < bbox[1] or h > bbox[3]:
                    w1 = 1e9
                    w2 = 1e9
                else:
                    # Source: Compute U for curr node
                    start_time = time.time()
                    w1 = foreground_energies[index] # Foregound
                    w2 = background_energies[index] # Background
                    
                    unary_total_time += time.time() - start_time

                # Sink: Compute U for curr node
                graph.add_tweights(index, w1, w2)

                # Compute pairwise edge weights
                start_time = time.time()
                for (nh, nw) in smoothness_matrix[(h,w)].keys():
                    neighbor_index = nh * img.shape[1] + nw
                    edge_weight = get_pairwise_energy(alpha, (h,w), (nh,nw), smoothness_matrix)
                    graph.add_edge(index, neighbor_index, edge_weight, edge_weight)
                pairwise_time = time.time() - start_time
                    # print (h,w),'->',(nh,nw),':',edge_weights[edge_map[(h,w,nh,nw)]]
        toc("Creating graph")
        print "unary_total_time:", unary_total_time
        print "pairwise_time:", pairwise_time

        # Graph has been created, run minCut

        tic()
        graph.maxflow()
        partition = graph.what_segment_vectorized()
        toc("Min cut")

        # Update alpha
        tic()
        num_changed_pixels = 0
        for index in xrange(len(partition)):
            h = index // img.shape[1]
            w = index %  img.shape[1]
            if partition[index] != alpha[h,w]:
                alpha[h,w] = partition[index]
                num_changed_pixels += 1
        toc("Updating alphas")

        # Terminate once the energy has converged
        # total_energy = get_energy(alpha, k, (background_gmm, foreground_gmm), img, smoothness_matrix)
        # relative_change = abs(previous_energy - total_energy)/previous_energy
        # previous_energy = total_energy
        # 
        relative_change = num_changed_pixels/float(img.shape[0]*img.shape[1])

        if iteration % 10 == 0 or relative_change < CONVERGENCE_CRITERON:
            result = np.reshape(partition, (img.shape[0], img.shape[1]))*255
            result = result.astype(dtype=np.uint8)
            result = np.dstack((result, result, result))
            plt.imshow(result)
            plt.show()

        print "-------------------------------------------------"
        print 'Relative change was %f'%relative_change

        if relative_change < CONVERGENCE_CRITERON:
            "EM has converged. Terminating."
            break

        # print "Relative Energy Change:", relative_change
        print "-------------------------------------------------"

    return alpha


def main():
    args = get_args()
    img = load_image(*args.image_file)

    if args.bbox:
        bbox = [int(p) for p in args.bbox]
    else:
        bbox = get_user_selection(img)
    print 'Bounding box:',bbox
    # small_banana_bbox = [3.3306451612903203, 3.7338709677419359, 94.661290322580641, 68.25]
    # big_banana_bbox = [25.306451612903231, 26.596774193548299, 605.95161290322574, 439.49999999999994]
    # bbox = small_banana_bbox
     
    grabcut(img, bbox)


# TODO:
# gt : clear namespace
# 4 neighbors
# Optimize node matrix creation with index computation while creating graph
# Optimize pairwise edge weight computation

# Exact bounding box from segmentation
# Singular covariance matrix - Add 1e^-8*identity
# Manage Empty clusters`
# 
if __name__ == '__main__':
    main()