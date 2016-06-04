'''
Matrix-factorization using Alternative least square method using theano.
'''

import theano.tensor as T
import theano
import numpy as np
import time

np.random.seed(42)

class MF(object):

    def __init__(self, n, l, alpha, beta):
        '''
        X : data matrix to factorize
        l : number of latent factors to factorize the matrix into
        alpha : regularization parameter
        '''
        self.l = l
        self.alpha = alpha
        X = T.matrix()
        W = np.random.rand(n, l).astype(theano.config.floatX)
        H = np.random.rand(l, n).astype(theano.config.floatX)
        self.W = theano.shared(W)
        self.H = theano.shared(H)
        self.err = 0
        self.params = [self.W, self.H]
        cost = (T.sum((X - T.dot(self.W, self.H)) **
                     2)) + self.alpha * (T.sum(self.W ** 2) + T.sum(self.H ** 2))
        grad = T.grad(cost, wrt=self.params)
        #grad /= T.sum(grad)
        updates = [(param, param - beta * (param_grad / T.sqrt(T.sum(param_grad **2))))
                   for param, param_grad in zip(self.params, grad)]
        #updates2 = [(param_grad, param_grad / T.sqrt(T.sum(param_grad **2))) for param_grad in grad]
        #updates = updates + updates2
        self.mf = theano.function(inputs=[X], outputs=cost, updates=updates)

    def mat_fact(self, X, n_iter):
        '''
        n_ter : Number of iters to run SGD
        beta : learning_rate
        '''
        for i in xrange(n_iter):
            cost = self.mf(X)
            print("Cost at iteration %d is %f" % (i, cost))
        W, H = self.get_factors()
        self.err = np.sqrt(np.sum((X - np.dot(W, H)) ** 2))
        print self.err

    def get_factors(self):
        return self.W.get_value(), self.H.get_value()


if __name__ == "__main__":
    X = np.random.rand(10000, 10000).astype(theano.config.floatX)
    print X
    t = time.time()
    mf = MF(10000, 500, 0.002, 0.5)
    mf.mat_fact(X, 200)
    print (time.time() - t)
