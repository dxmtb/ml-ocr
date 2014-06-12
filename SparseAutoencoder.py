import numpy as np
import numpy.random as nrd
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from mnist import read_mnist
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')


class SparseAutoencoder(object):
    def __init__(self, feat_size, hidden_size):
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.w1 = theano.shared(nrd.randn(feat_size, hidden_size))
        self.b1 = theano.shared(nrd.randn(hidden_size))
        self.w2 = theano.shared(nrd.randn(hidden_size, feat_size))
        self.b2 = theano.shared(nrd.randn(feat_size))
        self.x = T.matrix('x')
        self.params = [self.w1, self.b1, self.w2, self.b2]
        self.o1 = None
        self.o2 = None

    def get_cost_updates(self, learning_rate=0.1, rho=0.05, alpha=0.1, beta=1):
        o1 = sigmoid(theano.dot(self.x, self.w1)+self.b1)  # output of hidden layer
        o2 = sigmoid(theano.dot(o1, self.w2)+self.b2)  # output layer
        cost_reconstruct = T.mean(((o2-self.x)**2).sum(axis=1))  # reconstruct error
        cost_regular = (T.sum(self.w1**2)+T.sum(self.b1**2)+
                        T.sum(self.w2**2)+T.sum(self.b2**2))
#        rho_real = o1.mean(axis=0)
#        cost_sparse = T.sum(rho * theano.log(rho/rho_real) +
#                            (1-rho) * theano.log((1-rho)/(1-rho_real)))
        cost = cost_reconstruct + alpha * cost_regular

        grad = T.grad(cost, self.params)
        updates = []

        for w, g in zip(self.params, grad):
            updates.append((w, w - learning_rate * g))

        return cost, updates

    def fit(self, train_image, batch_size=100):
        index = T.lscalar()
        data_shared = theano.shared(
            np.asarray(train_image.tolist(), dtype=theano.config.floatX))
        cost, updates = self.get_cost_updates()
        train = theano.function(
            [index], cost, updates=updates,
            givens={self.x: data_shared[index * batch_size: (index + 1) * batch_size]})
        logging.info("Start training")

        for epoch in xrange(20):
            costs = []
            logging.info("Epoch %d" % epoch)
            for ind in xrange(train_image.shape[0]/batch_size):
                costs.append(train(ind))
            logging.info("Costs %lf" % np.mean(costs))

        logging.info("Save to theta.npy")
        np.save("theta.npy", self.w1.get_value())

if __name__ == '__main__':
    train_image, train_label, test_image, test_label = read_mnist()
    sae = SparseAutoencoder(train_image[0].size, 100)

    sae.fit(train_image)
