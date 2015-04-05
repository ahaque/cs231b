from matplotlib.patches import Rectangle
from graph_tool.all import *
from gmm import GMM
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Global constants
gamma = 50
beta = 0 # TODO: optimize beta Boykov and Jolly 2001


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

    for h in range(height): # Rows
        for w in range(width): # Columns
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
def create_graph():
    g = Graph(directed=False)
    v1 = g.add_vertex()
    v2 = g.add_vertex()

    e = g.add_edge(v1, v2)

    vprop_vint = g.new_vertex_property("vector<float>")
    vprop_vint[g.vertex(1)] = [1.2, 3.5]

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
            for (nh, nw) in smoothness_matrix[(h,w)].keys():
                print (h,w), (nh, nw)

def compute_smoothness(z):
    height, width, _ = z.shape
    global beta
    smoothness_matrix = dict()

    for h in xrange(z.shape[0]):
        print 'In row', h
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
        get_energy(alpha, k, (background_gmm, foreground_gmm), img, smoothness_matrix)

        # 3. Estimate segmentation using min cut

        break

if __name__ == '__main__':
    main()