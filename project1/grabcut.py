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
def create_graph(img):
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
    # print k
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
    # middle_matrix = (np.multiply(term, cov[:, :, 0]) + np.multiply(term, cov[:, :, 1]) +np.multiply(term, cov[:, :, 2]))
    middle_matrix = np.array([np.sum(np.multiply(term, cov[:, :, 0]),axis=1),
                              np.sum(np.multiply(term, cov[:, :, 1]),axis=1),
                              np.sum(np.multiply(term, cov[:, :, 2]),axis=1)]).T
    # print middle_matrix.shape

    log_prob = np.sum(np.multiply(middle_matrix, term), axis=1)

    # print 0.5*log_prob[5], get_log_prob(alpha, k[5], gmms, pixels[5])
    # print 0.5*log_prob[1], get_log_prob(alpha, k[1], gmms, pixels[1])

    # print 0.5*log_prob[500], get_log_prob(alpha, k[500], gmms, pixels[500])

    # print 0.5*log_prob[20], get_log_prob(alpha, k[20], gmms, pixels[20])



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
        # print 'Pairwise',(h,w), (nh,nw), V
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

def compute_beta_vectorized(z, debug=False):
    accumulator = 0
    m = z.shape[0]
    n = z.shape[1]

    vert_shifted = z - np.roll(z, 1, axis=0)
    temp = np.sum(np.multiply(vert_shifted, vert_shifted), axis=2)
    accumulator = np.sum(temp[1:,:])

    horiz_shifted = z - np.roll(z, 1, axis=1)
    temp = np.sum(np.multiply(horiz_shifted, horiz_shifted), axis=2)
    accumulator += np.sum(temp[:,1:])

    num_comparisons = float(2*(m*n) - m - n)

    beta = (2*(accumulator/num_comparisons))**-1

    return beta      

def compute_smoothness(z, debug=False):
    EIGHT_NEIGHBORHOOD = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    FOUR_NEIGHBORHOOD = [(-1,0), (0,-1), (0,1), (1,0)]

    height, width, _ = z.shape
    global beta
    smoothness_matrix = dict()

    beta = compute_beta_vectorized(z)
    if debug:
        print 'beta',beta

    for h in xrange(height):
        if debug:
            print 'Computing row',h
        for w in xrange(width):
            if (h,w) not in smoothness_matrix:
                smoothness_matrix[(h,w)] = dict()
            for hh,ww in EIGHT_NEIGHBORHOOD:
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
                    np.exp(-1 * beta * (np.linalg.norm(z[h,w,:] - z[nh,nw,:])**2))
                smoothness_matrix[(nh,nw)][(h,w)] = smoothness_matrix[(h,w)][(nh, nw)]

                if debug:
                    print (h,w),'->',(nh,nw),":",z[h,w,:], z[nh,nw,:], smoothness_matrix[(h,w)][(nh, nw)]

    return smoothness_matrix

def compute_smoothness_vectorized(z, debug=False):
    EIGHT_NEIGHBORHOOD = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    FOUR_NEIGHBORHOOD = [(0,-1,0), (1,1,0), (2,0,-1), (3,0,1)]

    height, width, _ = z.shape
    global beta
    smoothness_matrix = dict()

    beta = compute_beta_vectorized(z)
    if debug:
        print 'beta',beta

    vert_shifted_down = z - np.roll(z, 1, axis=0) # prev guy - start access at 1
    vert_shifted_up = z - np.roll(z, -1, axis=0) # next guy - end access before 1

    horiz_shifted_left = z - np.roll(z, 1, axis=1)
    horiz_shifted_right = z - np.roll(z, -1, axis=1)

    energies = []
    # (i,j) gives norm(z[i,j] - z[i-1,j])
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(vert_shifted_down, vert_shifted_down), axis=2)))
    # (i,j) gives norm(z[i,j] - z[i+1,j])
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(vert_shifted_up, vert_shifted_up), axis=2)))
    # (i,j) gives norm(z[i,j] - z[i,j-1])
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(horiz_shifted_left, horiz_shifted_left), axis=2)))
    # (i,j) gives norm(z[i,j] - z[i,j+1])
    energies.append(np.exp(-1 * beta * np.sum(np.multiply(horiz_shifted_right, horiz_shifted_right), axis=2)))
    #print temp[18, 3], (np.linalg.norm(z[18,3,:] - z[18,4,:])**2)

    for h in xrange(height):
        for w in xrange(width):
            if (h,w) not in smoothness_matrix:
                smoothness_matrix[(h,w)] = dict()
            for i,hh,ww in FOUR_NEIGHBORHOOD:
                nh, nw = h + hh, w + ww
                if nw < 0 or nw >= width:
                    continue
                if nh < 0 or nh >= height:
                    continue

                if (nh,nw) not in smoothness_matrix:
                    smoothness_matrix[(nh,nw)] = dict()

                if (h,w) in smoothness_matrix[(nh,nw)]:
                    continue

                smoothness_matrix[(h,w)][(nh, nw)] = energies[i][h,w]
                smoothness_matrix[(nh,nw)][(h,w)] = smoothness_matrix[(h,w)][(nh, nw)]

                if debug:
                    print (h,w),'->',(nh,nw),":",z[h,w,:], z[nh,nw,:], smoothness_matrix[(h,w)][(nh, nw)]

    return smoothness_matrix


def grabcut(img, bbox, debug=False, drawImage=False):
    if debug: 
        print 'Initializing gmms'
        tic()
    alpha, foreground_gmm, background_gmm = initialization(img, bbox)
    k = np.zeros((img.shape[0],img.shape[1]), dtype=int)
    if debug:
        toc('Initializing gmms')

    if debug:
        print 'Computing smoothness matrix...'
        tic()
    # tic()
    # s1 = compute_smoothness(img, debug=False)
    # toc('Computing smoothness matrix normally')
    # tic()
    smoothness_matrix = compute_smoothness_vectorized(img, debug=False)
    # toc('Computing smoothness matrix vectorally')

    # for (h,w) in s1:
    #     if len(s1[(h,w)]) != len(smoothness_matrix[(h,w)]):
    #         print 'PROBLEM lens not equal at',(h,w)
    #     for (nh,nw) in s1[(h,w)]:
    #         if abs(s1[(h,w)][(nh,nw)] - smoothness_matrix[(h,w)][(nh,nw)]) > 1e-12:
    #             print 'PROBLEM at',(h,w),(nh,nw),'->',s1[(h,w)][(nh,nw)],'!=',smoothness_matrix[(h,w)][(nh,nw)]
    # print 'Finished check'
    # print smoothness_matrix
    if debug:
        toc('Computing smoothness matrix')

    global SOURCE
    global SINK
    
    FOREGROUND = 1
    BACKGROUND = 0
    previous_energy = sys.float_info.max
    if debug:
        print 'Starting EM'

    for iteration in xrange(1,11):
        if debug:
            print "-------------------------------------------------"
            print 'Iteration %d'%iteration
            print np.sum(alpha)/float(img.shape[0]*img.shape[1])
        # 1. Assigning GMM components to pixels
        if debug:
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
        if debug:
            toc('Assigning GMM components')

        # # K-means visualization
        # COLORS = [[255,0,0],[0,255,0], [0,0,255], [255,255,0], [255,0,255]]
        # res = np.zeros(img.shape, dtype=np.uint8)
        # for h in xrange(img.shape[0]):
        #     for w in xrange(img.shape[1]):
        #         res[h,w,:] = COLORS[k[h,w]] 
        # plt.imshow(res)
        # plt.show()

        # 2. Learn GMM parameters
        if debug:
            tic()
        foreground_assignments = -1*np.ones(k.shape)
        foreground_assignments[alpha==1] = k[alpha==1]

        background_assignments = -1*np.ones(k.shape)
        background_assignments[alpha==0] = k[alpha==0]

        # print k.shape, foreground_assignments.shape
        # print np.sum(foreground_assignments != -1), np.sum(background_assignments != -1)
        # print img.shape[0]*img.shape[1]

        # print k[alpha==0]
        # return
        # 
        # print 'beforeeeee',len(foreground_gmm.gaussians)
        # print 'beforeeeee',len(background_gmm.gaussians)

        foreground_gmm.update_components(img, foreground_assignments)
        background_gmm.update_components(img, background_assignments)
        if debug:
            toc('Updating GMM parameters')

        if debug:
            tic()
        foreground_components = foreground_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))
        background_components = background_gmm.get_component(pixels).reshape((img.shape[0], img.shape[1]))

        k[alpha==1] = foreground_components[alpha==1]
        k[alpha==0] = background_components[alpha==0]
        if debug:
            toc('Assigning GMM components again')

        # print 'Foreground',[np.sum(foreground_components==i) for i in xrange(5)]
        # print 'Background',[np.sum(background_components==i) for i in xrange(5)]

        # print 'aftaaaaaaaa',len(foreground_gmm.gaussians)
        # print 'aftaaaaaaaa',len(background_gmm.gaussians)

        # 3. Estimate segmentation using min cut
        # Update weights
        # Compute Unary weights
        if debug:
            tic()
        graph = create_graph(img)
        theta = (background_gmm, foreground_gmm)
        total_energy = 0

        k_flattened =  k.reshape((img.shape[0]*img.shape[1], 1))
        foreground_energies = get_unary_energy_vectorized(1, k_flattened, theta, pixels)
        background_energies = get_unary_energy_vectorized(0, k_flattened, theta, pixels)

        pairwise_energies = np.zeros(img.shape[0:2], dtype=float)
        unary_total_time = 0.0
        pairwise_time = 0.0
        # done_with = set()
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
                    # print w1,get_unary_energy(1, k, theta, img, (h,w))
                    # print w2,get_unary_energy(0, k, theta, img, (h,w))
                    # print ''

                    
                    unary_total_time += time.time() - start_time

                # Sink: Compute U for curr node
                graph.add_tweights(index, w1, w2)

                # Compute pairwise edge weights
                start_time = time.time()
                pairwise_energy = 0.0
                for (nh, nw) in smoothness_matrix[(h,w)].keys():
                    neighbor_index = nh * img.shape[1] + nw
                    # if (index, neighbor_index) in done_with:
                    #     continue
                    edge_weight = get_pairwise_energy(alpha, (h,w), (nh,nw), smoothness_matrix)
                    graph.add_edge(index, neighbor_index, edge_weight,edge_weight)
                    # done_with.add((index, neighbor_index))
                    # done_with.add((neighbor_index, index))
# 
                    pairwise_energy += edge_weight
                #if debug:
                # if w1 < 1e8: 
                #     print (h,w), w1, w2, pairwise_energy
                pairwise_energies[h,w] = pairwise_energy
                pairwise_time = time.time() - start_time
                    # print (h,w),'->',(nh,nw),':',edge_weights[edge_map[(h,w,nh,nw)]]
                    # 
        # import matplotlib.cm as cm
        # plt.subplot(2, 2, 1)
        # plt.imshow(np.floor(foreground_energies/np.max(foreground_energies)*255).reshape(img.shape[0:2]), cmap = cm.Greys_r)
        # plt.subplot(2, 2, 2)
        # plt.imshow(np.floor(background_energies/np.max(background_energies)*255).reshape(img.shape[0:2]), cmap = cm.Greys_r)
        # plt.subplot(2, 2, 3)
        # plt.imshow(np.floor(pairwise_energies/np.max(pairwise_energies)*255).reshape(img.shape[0:2]), cmap = cm.Greys_r)
        # plt.subplot(2, 2, 4)
        # plt.imshow(np.floor(pairwise_energies/np.max(pairwise_energies)*255).reshape(img.shape[0:2]), cmap = cm.Greys_r)
        # plt.show()


        if debug:
            toc("Creating graph")
            print "unary_total_time:", unary_total_time
            print "pairwise_time:", pairwise_time

        # Graph has been created, run minCut
        if debug:
            tic()
        graph.maxflow()
        partition = graph.what_segment_vectorized()
        if debug:
            toc("Min cut")

        # Update alpha
        if debug:
            tic()
        # num_changed_pixels = 0
        # for index in xrange(len(partition)):
        #     h = index // img.shape[1]
        #     w = index %  img.shape[1]
        #     if partition[index] != alpha[h,w]:
        #         alpha[h,w] = partition[index]
        #         num_changed_pixels += 1
        partition = partition.reshape(alpha.shape)
        # n = num_changed_pixels
        num_changed_pixels = np.sum(np.abs(partition-alpha))
        alpha = partition

        if debug:
            toc("Updating alphas")

        # Terminate once the energy has converged
        # total_energy = get_energy(alpha, k, (background_gmm, foreground_gmm), img, smoothness_matrix)
        # relative_change = abs(previous_energy - total_energy)/previous_energy
        # previous_energy = total_energy
        # 
        relative_change = num_changed_pixels/float(img.shape[0]*img.shape[1])

        if drawImage:
            if iteration % 10 == 0:# or relative_change < CONVERGENCE_CRITERON:
                result = np.reshape(partition, (img.shape[0], img.shape[1]))*255
                result = result.astype(dtype=np.uint8)
                result = np.dstack((result, result, result))
                plt.imshow(result)
                plt.show()
        if debug:
            print 'Relative change was %f'%relative_change

        if relative_change < CONVERGENCE_CRITERON:
            if debug:
                print "EM has converged. Terminating."
            # break

        # print "Relative Energy Change:", relative_change

    return alpha


def main():
    args = get_args()
    img = load_image(*args.image_file)

    if args.bbox:
        bbox = [int(p) for p in args.bbox]
    else:
        bbox = get_user_selection(img)
    print 'Bounding box:',bbox

    # plt.imshow(img)
    # currentAxis = plt.gca()
    # currentAxis.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], facecolor="grey"))
    # plt.show()

    # small_banana_bbox = [3.3306451612903203, 3.7338709677419359, 94.661290322580641, 68.25]
    # big_banana_bbox = [27.887096774193537, 33.048387096774036, 604.66129032258061, 451.11290322580641]
    # bbox = small_banana_bbox
     
    grabcut(img, bbox, debug=True, drawImage=True)

# TODO:
# gt : clear namespace
# [DONE] 4 neighbors
# Optimize node matrix creation with index computation while creating graph
# [DONE] Optimize pairwise edge weight computation
# Exact bounding box from segmentation
# Singular covariance matrix - Add 1e^-8*identity
# Manage Empty clusters`
# 
if __name__ == '__main__':
    main()