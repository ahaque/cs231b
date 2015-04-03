import numpy as np

class Gaussian:
    # mean: mean for the gaussian
    # sigma: Covariance matrix 
    def __init__(self, mean=np.zeros((3,1)), sigma=np.eye(3)):
        self.mean = np.array(mean)
        self.sigma = np.array(sigma)
        
        self.sigma_det = np.linalg.det(sigma)
        self.sigma_inv = np.linalg.inv(sigma)

        self.k = 3
        self.TWO_PI_3 = (2*np.pi)**self.k
        self.term1 = 1/np.sqrt(self.TWO_PI_3 * self.sigma_det)

    def compute_probability(self, x, debug=False):
        x = np.array(x)
        
        if debug:
            print 'term1',self.term1
            print 'diff',(x - self.mean)
            print 'diff',(x - self.mean).T
            print 'inv',self.sigma_inv
            print 'mult',np.dot(np.dot((x - self.mean).T, self.sigma_inv), (x - self.mean))
            print ''
        
        return self.term1 * np.exp(-0.5 * np.dot(np.dot((x - self.mean).T, self.sigma_inv), (x - self.mean).T))

    def update_parameters(self, data):
        self.mean = np.mean(data, axis=0)
        self.sigma = np.cov(data, rowvar=0)

        self.sigma_det = np.linalg.det(self.sigma)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.term1 = 1/np.sqrt(self.TWO_PI_3 * self.sigma_det)

def gaussian_test():
    mean = [128, 100, 52]
    cov = [[ 1.23730032,  0.4888689,  -0.28632465],
           [ 0.4888689,   1.09926646, -0.15930243],
           [-0.28632465, -0.15930243,  0.94947671]]

    g = Gaussian()
    g.update_parameters(np.random.multivariate_normal(mean=mean, cov=cov, size=100000))

    if not np.mean(abs(mean - g.mean)) < 1e-2 and np.mean(abs(sigma - g.sigma)):
        print 'Error'

def main():
    gaussian_test()

if __name__ == '__main__':
    main()