import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Define sigmoid function
def sigm(x):
    return 1 / (1+np.exp(-x))

# Define derivative of sigmoid function
def sigm_prime(x):
    a = sigm(x)
    return a * (1 - a)

# Randomly initialize weights and biases for the NN
# Weights and biases stored in a dictionary
def setup_init_weights(nn_structure):
    # import numpy.random as randy
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = np.random.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = np.random.random_sample((nn_structure[l],))
        
    return W, b

# Set the initial mean accumulation values, delta_W, delta_b to zero
# These values represent the sum of the partial derivatives of the individual
# sample cost function calculations.
def init_delta_values(nn_structure):
    delta_W = {}
    delta_b = {}
    for l in range (1, len(nn_structure)):
        delta_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        delta_b[l] = np.zeros((nn_structure[l],))
    return delta_W, delta_b

# Given input vector, weight and bias matrix, calculate the sum
# of the inputs into node i at layer l (z) and the activated values (h),
# i.e. the output value of node i at layer l
def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l]
        h[l+1] = sigm(z[l+1])
    return h, z

# Calculate the delta of the output layer
def calculate_out_layer_delta(y, h_out, z_out):
    return -(y-h_out) * sigm_prime(z_out)

# Calculate the delta of the hidden layers
def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus_1,) * sigm_prime(z_l)

def train_nn(nn_structure, X, y, iter_num = 3000, alpha = 0.25):
    W, b = setup_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
            
        # first initialize the sum of partial derivatives to 0
        tri_W, tri_b = init_delta_values(nn_structure)
        avg_cost = 0
        
        # iterate through all training examples
        for i in range(len(y)):
            delta = {}
            
            # perform feed forward pass and return h, z values
            # h, z are used in the gradient descent loop
            h, z = feed_forward(X[i, :], W, b) # iterate through each row
            
            # iterate backwards (backpropogate errors)
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    # don't need to perform for input layer
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                        tri_W[l] += np.dot(delta[l+1][:, np.newaxis], np.transpose(h[l][:,np.newaxis]))
                        tri_b[l] += delta[l+1]
        
        # iterate through all layers and perform gradient descent for weight in each layer
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha*(1.0/m * tri_W[l])
            b[l] += -alpha*(1.0/m * tri_b[l])
            
        avg_cost = 1/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func