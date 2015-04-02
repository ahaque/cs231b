import numpy as np

class Gaussian:
    # mean: mean for the gaussian
    # sigma: Covariance matrix 
    def __init__(self, mean=np.zeros((3,1)), sigma=np.ones((3,3))):
        self.mean = mean
        self.sigma = sigma

        # self.sigma_det = np.linalg.det(sigma)
        # self.sigma_inv = np.linalg.inv(sigma)

        self.k = 3
        self.TWO_PI_3 = (2*np.pi)**self.k
        # self.term1 = 1/np.sqrt(self.TWO_PI_3 * self.sigma_det)

    def compute_probability(x):
        return self.term1 * np.exp(-0.5 * (x - mean).T * self.sigma_inv * (x - mean))

    def update_parameters(self, data):
        self.mean = np.mean(data, axis=0)
        self.sigma = np.cov(data, rowvar=0)

        self.sigma_det = np.linalg.det(self.sigma)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.term1 = 1/np.sqrt(self.TWO_PI_3 * self.sigma_det)

def main():
    cov = [[1, 0.5, 0.2], 
            [0.3, 1, -0.2], 
            [-0.9, -0.4, 1]]
    g = Gaussian()

    g.update_parameters(np.random.multivariate_normal(mean=[128,128,128], cov=cov, size=10000000))

    print g.mean, g.sigma

if __name__ == '__main__':
    main()