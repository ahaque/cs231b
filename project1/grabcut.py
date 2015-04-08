from matplotlib.patches import Rectangle
from graph_tool.all import *
from gmm import GMM
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Global constants
gamma = 50
beta = 0 # TODO: optimize beta Boykov and Jolly 2001

SOURCE = -1
SINK = -2


# get_args function
# Intializes the arguments parser and reads in the arguments from the command
# line.
# 
# Returns: args dict with all the arguments
def get_args():
    parser = argparse.ArgumentParser(
        description='Implementation of the GrabCut algorithm.')
    parser.add_argument('image_file', 
        nargs=1, help='Input image name along with its relative path')

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
    global SOURCE
    global SINK

    g = Graph(directed=False)

    source = g.add_vertex()
    sink = g.add_vertex()
    edge_weights = g.new_edge_property("float")

    # Creating nodes
    edge_map = dict()
    node_matrix = []
    for h in xrange(img.shape[0]):
        current_row = []
        node_matrix.append(current_row)
        for w in xrange(img.shape[1]):
            current_row.append(g.add_vertex())

    # Create edges
    for h in xrange(img.shape[0]):
        for w in xrange(img.shape[1]):
            # Create tweights
            curr_node = node_matrix[h][w]
            edge_map[(SOURCE,SOURCE,h,w)] = g.add_edge(source, curr_node)
            edge_map[(h,w,SINK,SINK)] = g.add_edge(curr_node, sink)
            
            for nh, nw in neighbor_list[(h,w)].keys():
                if (h,w,nh,nw) not in edge_map:
                    edge_map[(h,w,nh,nw)] = g.add_edge(source, node_matrix[nh][nw])
                    edge_map[(nh,nw,h,w)] = edge_map[(h,w,nh,nw)]

    return g, edge_weights, edge_map

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
        V = smoothness_matrix[(h,w)][(nh, nw)]

    return gamma *V

def compute_smoothness(z, debug=False):
    height, width, _ = z.shape
    global beta
    smoothness_matrix = dict()

    for h in xrange(z.shape[0]):
        if debug:
            print 'Computing row',h
        for w in xrange(z.shape[1]):
            smoothness_matrix[(h,w)] = dict()
            for hh in [-1,0,1]:
                for ww in [-1,0,1]:
                    if hh == 0 and ww == 0:
                        continue
                    if w + ww < 0 or w + ww >= width:
                        continue
                    if h + hh < 0 or h + hh >= height:
                        continue

                    smoothness_matrix[(h,w)][(h+hh, w+ww)] = \
                        np.exp(-1 * beta * np.linalg.norm(z[h,w,:] - z[h+hh,w+ww,:]))

    return smoothness_matrix

def main():
    args = get_args()
    img = load_image(*args.image_file)
    
    # bbox = get_user_selection(img)
    bbox = [10, 10, img.shape[0]-10, img.shape[1]-10]

    print 'Initializing gmms'
    alpha, foreground_gmm, background_gmm = initialization(img, bbox)
    k = np.zeros((img.shape[0],img.shape[1]), dtype=int)

    print 'Computing smoothness matrix...'
    smoothness_matrix = compute_smoothness(img)

    print 'Creating image graph'
    graph, edge_weights, edge_map = create_graph(img, smoothness_matrix)

    global SOURCE
    global SINK
    
    FOREGROUND = 1
    BACKGROUND = 0
    print 'Starting EM'
    while True:
        # 1. Assigning GMM components to pixels
        for h in xrange(img.shape[0]):
            for w in xrange(img.shape[1]):
                if alpha[h,w] == 1:
                    k[h,w] = foreground_gmm.get_component(img[h,w,:])
                else:
                    k[h,w] = background_gmm.get_component(img[h,w,:])

        # 2. Learn GMM parameters
        foreground_assignments = -1*np.ones(k.shape)
        foreground_assignments[alpha==1] = k[alpha==1]

        background_assignments = -1*np.zeros(k.shape)
        background_assignments[alpha==0] = k[alpha==0]

        foreground_gmm.update_components(img, foreground_assignments)
        background_gmm.update_components(img, background_assignments)

        # 3. Estimate segmentation using min cut
        # print get_energy(alpha, k, (background_gmm, foreground_gmm), img, smoothness_matrix)
        
        # Update weights
        # # TODO: move energy computation here and update edge weights
        theta = (background_gmm, foreground_gmm)
        for h in xrange(img.shape[0]):
            for w in xrange(img.shape[1]):
                # Source: Compute U for curr node
                w1 = get_unary_energy(1, k, theta, img, (h, w)) # Foregound
                w2 = get_unary_energy(0, k, theta, img, (h, w)) # Background

                # Sink: Compute U for curr node
                edge_weights[edge_map[(SOURCE,SOURCE,h,w)]] = w1 # Source
                edge_weights[edge_map[(h,w,SINK,SINK)]] = w2 # Sink

                # Compute pairwise edge weights
                for (nh, nw) in smoothness_matrix[(h,w)].keys():
                    edge_weights[edge_map[(h,w,nh,nw)]] = get_pairwise_energy(alpha, (h,w), (nh,nw), smoothness_matrix)

        # Graph has been created, run minCut
        print 'Performing minCut'
        mc, partition = min_cut(graph, edge_weights)

        print 'Drawing graph'
        pos = g.vertex_properties["pos"]
        graph_draw(g, pos=pos, edge_pen_width=weight, vertex_fill_color=part,
                output="example-min-cut.pdf")
        

        # for edge in edge_map:
        #     (sh,sw,dh,dw) = edge
        #     edge_weights[edge_map[edge]] = 
        

        break

if __name__ == '__main__':
    main()