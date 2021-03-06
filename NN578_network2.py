# David Beck
# CSC578-701
# HW2
#
# link to video: https://photos.app.goo.gl/B5ygQXek83UoyZR98

"""
10/9 Updated

10/2018 
NN578_network2.py
==============

Modified from the NNDL book code "network2.py" to be 
compatible with Python 3.

Also from "network2.py", the function ("load(filename)") is
added to this file, renamed as "load_network(filename)".
This function loads a saved network encoded in a json file.

"""

"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### 10/2018 nt:
#### Definitions of the cost functions (as function classes)
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``."""
        return 0.5*np.linalg.norm(y-a)**2

    ## 10/2018 nt: addition
    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        return -(y-a)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        return np.divide((a-y), (a*(1-a)), out=np.zeros_like(a), where=((a*(1-a))!=0))
    
class LogLikelihood(object):
    
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output ``y``."""
        loc = np.where(y==1)
        pl = -np.sum(np.nan_to_num(np.log(a[loc])))
        return pl
    
    @staticmethod
    def derivative(a, y):
        """Return the first derivative of the function."""
        loc = np.where(y==1)
        xarray = np.zeros_like(y)
        pl = np.divide(-1, a[loc], where=(a[loc]!=0))
        xarray[np.argmax(y)] = pl
        return xarray
    
#### 9/2018 nt:
#### Definitions of the activation functions (as function classes)
class Sigmoid(object):
    
    @staticmethod
    def fn(z):
        """The sigmoid function."""
        return np.divide(1.0, (1.0+np.exp(-z)))

    @classmethod
    def derivative(cls,z):
        """Derivative of the sigmoid function."""
        return cls.fn(z)*(1-cls.fn(z))

class Tanh(object):
    
    @staticmethod
    def fn(z):
        """The tanh function."""
        return np.divide(np.exp(z)-np.exp(-z), (np.exp(z)+np.exp(-z)))

    @classmethod
    def derivative(cls,z):
        """Derivative of the tanh function."""
        # double check if correct
        return 1-(cls.fn(z))**2
    
class ReLU(object):
    
    @staticmethod
    def fn(z):
        """The ReLU function."""
        return np.maximum(z, 0)

    @classmethod
    def derivative(cls,z):
        """Derivative of the ReLU function."""
        return np.heaviside(z, 0)
    
class Softmax(object):
    @staticmethod
    # Parameter z is an array of shape (len(z), 1).
    def fn(z):
        """The softmax of vector z."""
        return np.divide(np.exp(z), np.sum(np.exp(z)), out=np.zeros_like(z), where=(np.sum(np.exp(z)!=0)))

    @classmethod
    def derivative(cls,z):
        """Derivative of the softmax.  
        IMPORTANT: The derivative is an N*N matrix.
        """
        a = cls.fn(z) # obtain the softmax vector
        return np.diagflat(a) - np.dot(a, a.T)

    
#### Main Network class
class Network(object):

    ## 10/2018 nt: additional keyword arguments for hyper-parameters
    def __init__(self, sizes, cost=CrossEntropyCost, act_hidden=Sigmoid, \
                 act_output=None, regularization=None, dropoutpercent=0.0):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost
        ## 10/2018 nt: addition
        self.act_hidden = act_hidden
        if act_output == None:
            self.act_output = self.act_hidden
        else:
            self.act_output = act_output
        # change output function to Sigmoid in case of Tanh & not QuadraticCost
        if act_output == Tanh and cost != QuadraticCost: 
            self.act_output = Sigmoid
        self.regularization = regularization
        self.dropoutpercent = dropoutpercent
        # set flag for dropout
        if self.dropoutpercent != 0.0:
            self.dropout = True
            # list to store dropout layers
            self.dropoutlayers = []
        else: 
            self.dropout = False
    

    ## 10/2018 nt: convenience function for setting hyper-parameters
    def set_parameters(self, cost=QuadraticCost, act_hidden=Sigmoid, \
                       act_output=None, regularization=None, dropoutpercent=0.0):
        self.cost=cost
        self.act_hidden = act_hidden
        if act_output == None:
            self.act_output = self.act_hidden
        else:
            self.act_output = act_output
        # change output function to Sigmoid in case of Tanh & not QuadraticCost
        if act_output == Tanh and cost != QuadraticCost: 
            self.act_output = Sigmoid
        self.regularization = regularization
        self.dropoutpercent = dropoutpercent
        # set flag for dropout
        if self.dropoutpercent != 0.0:
            self.dropout = True
            # list to store dropout layers
            self.dropoutlayers = []
        else: 
            self.dropout = False
        
    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        idx = 0
        for b, w in zip(self.biases, self.weights):
            if idx < len(self.biases)-1:
                a = (self.act_hidden).fn(np.dot(w, a)+b)
                # apply dropout layer to hidden activation layer(s)
                if (self.dropout):
                    a *= self.dropoutlayers[idx-2]
                idx += 1
            else: 
                a = (self.act_output).fn(np.dot(w, a)+b)
        return a

    ## 9/2018 nt: additional parameter 'no_convert' to control the vectorization of the target.
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            no_convert=True):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):#xrange(epochs):         
            #random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]#xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
                
            print ("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda) # nt: for cost, always NO convert (default) for training
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            # check for dropout switch off for accuracy and evaluation_data, if it exists
            if (monitor_evaluation_cost == True and self.dropoutpercent != 0.0):
                self.dropout = False
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True) # nt: for accuracy, always _DO_ convert (argmax) for training
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                ## 9/2018 nt: changed the last parameter convert
                if no_convert:
                    cost = self.total_cost(evaluation_data, lmbda) # nt: if test/val data is already vectorized for y
                else:
                    cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                ## 9/2018 nt: changed the last parameter convert
                if no_convert:
                    accuracy = self.accuracy(evaluation_data, convert=True) #nt: _DO_ convert (argmax)
                else:
                    accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {} / {}".format(
                    ## 9/2018 nt: This seems like a bug!
                    #self.accuracy(evaluation_data), n_data))
                    accuracy, n_data))
            # check for dropout and if yes, switch back to dropout
            if (monitor_evaluation_cost == True and self.dropoutpercent != 0.0):
                self.dropout = True
            print ('')
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
        
    ## 10/2018: THIS NEEDS CHANGE to incorporate self.regularization. 
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        ### this is where the dropout layers are created
        if (self.dropout):
            # create new dropout layer(s)
            for h in self.sizes[:-1]:
                d = np.random.binomial(1, self.dropoutpercent, size=(h,1)) / self.dropoutpercent
                # append dropout layer to droplist
                self.dropoutlayers.append(d)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # below is where the L1/L2 regularization happens
            #L2 regularization
        if self.regularization == None or self.regularization == 'L2':
            self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        else: 
            #L1 regularization
            self.weights = [(w-eta*(lmbda/n))*abs(w)/w-(eta/len(mini_batch))*nw 
                            for w, nw in zip(self.weights, nabla_w)]            
            self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        idx = 0
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            ## 9/2018 nt: changed
            #activation = sigmoid(z)
            # check for output/hidden
            if idx < len(self.biases)-1:
                activation = (self.act_hidden).fn(z)
                # apply dropout layer to hidden activation layer(s)
                if (self.dropout):
                    #print(activation.shape)
                    activation *= self.dropoutlayers[idx-2]
                    #print("dropout idx: ", (idx-1))
                idx += 1
            else: 
                activation = (self.act_output).fn(z)
            activations.append(activation)

        # backward pass
        ## 9/2018 nt: Cost and activation functions are parameterized now.
        ##            Call the activation function of the output layer with z.
        #delta = (self.cost).delta(zs[-1], activations[-1], y)
        a_prime = (self.act_output).derivative(zs[-1]) # 9/2018 nt: changed, da/dz
        #delta = (self.cost).derivative(activations[-1], y) * a_prime # 9/2018 nt: changed, dC/da * da/dz
        c_prime = (self.cost).derivative(activations[-1], y) #10/2018 nt: split a line to accommodate Softmax
        if self.act_output == Softmax:
            delta = np.dot(a_prime, c_prime)
        else:
            delta = c_prime * a_prime # 9/2018 nt: changed, dC/da * da/dz
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):#xrange(2, self.num_layers):
            z = zs[-l]
            ## 9/2018 nt: Changed to call the activation function of the hidden layer with z.
            #sp = sigmoid_prime(z)
            sp = (self.act_hidden).derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    ## 10/2018: THIS NEEDS CHANGE to incorporate self.regularization. 
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        ## L2 regularization
        if self.regularization == None or self.regularization == 'L2':
            cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        else: 
            #L1 regularization
            cost += ((lmbda/len(data))*sum(np.abs(np.linalg.norm(w)) for w in self.weights))           
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Loading a Network from a json file
def load_network(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    #cost = getattr(sys.modules[__name__], data["cost"])
    #net = Network(data["sizes"], cost=cost)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
