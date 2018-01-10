import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

# Save model to file
def save_model(model, fname):
    fout = open(fname, "wb")
    pkl.dump(model, fout)
    fout.close()
    return

# Load model from file
def load_model(fname):
    fin = open(fname, "rb")
    model = pkl.load(fin)
    fin.close()
    return model

# Calculate free energy of a dataset given a model
# The free energy of training/test datasets should be more or less the same
# If the ratio of the free energies increases, then we are overfitting
def free_energy(model, data):
    F = 0.0
    for iv in range(data.shape[0]):
        if iv % 1000 == 0:
            print('\r %d/%d' % (iv,data.shape[0]), end='')
        v = data[iv, :]
        F -=  np.dot(v, model.b) + np.sum(np.log(1.0 + np.exp(model.c + np.dot(v, model.W))))
    F = np.divide(F, data.shape[0])
    print('\nFaverage: ', F)

    return F


class RBM:
    # Initialization function for various parameters
    def __init__(self, nvisible, nhidden, seed=None):

        # Define architecture parameters
        self.nh = nhidden
        self.nv = nvisible

        # Set random seed for reproducible results
        if seed is None:
            np.random.seed(0)
        else:
            np.random.seed(seed)

        # Weight matrix
        self.W = np.random.normal(loc = 0.0, scale = 0.01, size=(self.nv,self.nh))
        # Bias for v
        self.b = np.random.uniform(low=-1, high=1, size=(self.nv))
        # Bias for h
        self.c = np.zeros(self.nh)

        # Initialize buffers for hidden and visible units states
        # d: data state
        # r: reconstruction state
        self.h = {'d': np.zeros((self.nh)), 'r': np.zeros((self.nh))}
        self.v = {'d': np.zeros((self.nv)), 'r': np.zeros((self.nv))}

        # Initialize other parameters
        # Number of gibbs samplings steps to perform during reconstruction
        self.gibbs_sampling_steps = 1
        # Switch for binary or real values for hidden and visible states
        # If binary enabled, h and v are sampled from the conditional distributions
        # and have values {0,1}. If False, then probabilities are used for the
        # calculation of the gradients
        self.binary_h = True
        self.binary_v = True
        # Conditional probability function
        self.prob_function = self.sigmoid
        # Set learning rate and decay rate
        self.learning_rate = 0.01
        self.decay_rate = 1.0

        return

    # Sigmoid function
    def sigmoid(self, x):
        return np.divide(1.0,1.0 + np.exp(-x))

    # Activate and calculate conditional P(h|v)
    def positive_activate(self, v):
        return self.prob_function(self.c + np.dot(v, self.W))

    # Activate and calculate conditional P(v|h)
    def negative_activate(self, h):
        return self.prob_function(self.b + np.dot(self.W, h))

    # Gibbs sampling
    def gibbs_sampling(self, step=1):
        # First initialization
        if step == 1:
            v = self.v['d']
        else:
            # If binary update the previous h state
            if self.binary_h:
                self.h['r'] = np.multiply(1, np.random.uniform(size=(self.nh)) <= self.h['r'])
            # Recalculate v
            p = self.negative_activate(self.h['r'])
            if self.binary_v:
                v = np.multiply(1, np.random.uniform(size=(self.nv)) <= p)
            elif not self.binary_v:
                v = p
        # P(h|v) and sample h
        p = self.positive_activate(v)
        if self.binary_h:
            h = np.multiply(1, np.random.uniform(size=(self.nh)) <= p)
        elif not self.binary_h:
            h = p
        # P(v|h) and sample v
        p = self.negative_activate(h)
        if self.binary_v:
            self.v['r'] = np.multiply(1, np.random.uniform(size=(self.nv)) <= p)
        elif not self.binary_v:
            self.v['r'] = p
        # Last positive propagation, do not sample in this step (better delta gradients)
        self.h['r'] = self.positive_activate(self.v['r'])
        # If only one step of gibbs sampling, then h is hd
        if step == 1:
            self.h['d'] = h

        return

    # Update weights
    def update_W(self, lrate):
        positive = np.dot(self.v['d'].reshape(self.nv, 1), self.h['d'].reshape(1, self.nh))
        negative = np.dot(self.v['r'].reshape(self.nv, 1), self.h['r'].reshape(1, self.nh))
        self.W = self.W + lrate * (positive - negative)
        return

    # Update bias parameters
    def update_bias(self, lrate):
        self.b = self.b + lrate * (self.v['d'] - self.v['r'])
        self.c = self.c + lrate * (self.h['d'] - self.h['r'])
        return

    # Calculate squared error between the reconstructed sample and the input one
    def get_squared_error(self):
        return np.sum(np.square(self.v['d'] - self.v['r']))

    # Visualize weights histogram
    def w_histogram(self, fname=None):
        fig = plt.figure()

        ax = fig.add_subplot(311)
        ax.hist(np.ravel(self.W))
        ax.set_title("W")
        ax = fig.add_subplot(312)
        ax.hist(self.b)
        ax.set_title("visible bias")
        ax = fig.add_subplot(313)
        ax.hist(self.c)
        ax.set_title("hidden bias")

        if fname is not None:
            plt.savefig(fname, bbox_inches ="tight")
        else:
            plt.show()

        return

    # Visualize weights map
    def w_map(self, nx, ny, fname=None):
        fig = plt.figure(figsize=(ny,nx))

        for i in range(1,nx*ny+1):
            ax = fig.add_subplot(nx, ny, i)
            ax.imshow(self.W[:,i-1].reshape(28,28), cmap="Greys", interpolation="nearest")
            ax.axis('off')

        if fname is not None:
            plt.savefig(fname, bbox_inches ="tight")
        else:
            plt.show()
            
        return

    # Daydreamtime!!!
    def dream(self, vin, gibbs_sampling_steps=1):
        self.gibbs_sampling_steps = gibbs_sampling_steps
        self.v['d'] = vin
        for i in range(1, self.gibbs_sampling_steps + 1):
            self.gibbs_sampling(step=i)

        return self.v['r']

    # Training function
    def train(self, X, gibbs_sampling_steps=1, lrate=0.01,
              epochs=1, decay_rate=1.0, verbose=False,
              binary_h=True, binary_v=True):

        # Copy inputs
        V = X.copy()

        # Check number of samples
        dim = V.ndim
        nsamples = None
        if dim == 1:
            nsamples = 1
        elif dim == 2:
            nsamples = V.shape[0]

        # Set gibbs sampling
        self.gibbs_sampling_steps = gibbs_sampling_steps
        # Set learning rate
        self.learning_rate = lrate
        self.decay_rate = decay_rate
        lrate_epoch = self.learning_rate
        # Set binary/not binary states
        self.binary_h = binary_h
        self.binary_v = binary_v

        # Initiate training
        for e in range(epochs):
            err = 0.0
            # Shuffle samples
            np.random.shuffle(V)
            for sample in range(nsamples):
                if verbose and (sample+1) % 1000 == 0:
                    print('\r %d/%d' % (sample+1,nsamples), end='')
                # Set visible layer state
                if dim == 1:
                    self.v['d'] = V
                else:
                    self.v['d'] = V[sample,:]
                # Perform Gibbs sampling and update hidden/visible states for
                # contrastive divergence
                for i in range(1,self.gibbs_sampling_steps+1):
                    self.gibbs_sampling(step=i)
                err += self.get_squared_error()
                # Update the weights and biases
                self.update_bias(lrate=lrate_epoch)
                self.update_W(lrate=lrate_epoch)
            if verbose:
                print("\nEpoch: %d, lrate: %2.6f, Error: %4.4f" % (e,lrate_epoch,np.divide(err,nsamples)))
            lrate_epoch = lrate_epoch * decay_rate

        return


