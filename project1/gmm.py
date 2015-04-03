from gaussian import Gaussian 
from sklearn.cluster import KMeans
import numpy as np

class GMM:
    def __init__(self, K):
        self.K = K
        self.gaussians = [Gaussian() for _ in xrange(self.K)]

    def initialize_gmm(self, X, debug=False):
        clusterer = KMeans(n_clusters=self.K)
        clusters = clusterer.fit_predict(X)

        if debug:
            print clusters
        
        for i, distribution in enumerate(self.gaussians):
            distribution.update_parameters(X[clusters==i])

    def get_component(self, x):
        return np.argmax([g.compute_probability(x) for g in self.gaussians])

def GMM_test():
    g = GMM(5)

    for i,distribution in enumerate(g.gaussians):
        distribution.mean = [i*3,i*3,i*3]

    print g.get_component([1.5,1.5,1.5])

    X = np.zeros((0,3))
    for i in range(5):
        X = np.concatenate((X, np.random.multivariate_normal([i*3]*3, np.eye(3), 10)), axis=0)

    g.initialize_gmm(X, debug=True)

    print [g.get_component(x) for x in X]
    

def main():
    GMM_test()

if __name__ == '__main__':
    main()