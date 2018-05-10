import numpy as np
from sklearn.metrics import mean_squared_error
import logging
import utils    
from time import time

np.random.seed(1234)

class Olga_ELM():
    def __init__(self, num_neurons, num_inputs):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.predictions = []
        self.true = []
        self.mse = []

    def tan_sigmoid(self, a):
        e_minus2a = np.exp(-2*a)
        return (1 - e_minus2a)/(1 + e_minus2a)

    def training(self, x, W, b, f, y):

        z = W.dot(x.T) + b        # z^(2) = W^(1)*a^(1) + b^(1)
        a = f(z)                # activation
        W_opt = np.linalg.pinv(a.T) @ y
        return W_opt.T

    def predicting(self, x, W, b, f, W_opt):

        z = W.dot(x.T)+b
        a = f(z)
        y = W_opt.dot(a)
        return y.T

    def extreme_learning_machine(self, x_train, y_train, x_test, y_test):

        tiny = 1e-12
        W = tiny*np.random.random_sample(size=(self.num_neurons, self.num_inputs))
        b = tiny*np.random.random_sample(size=(self.num_neurons, 1))
        print('W shape: {}, b shape: {}, x_train shape: {}'.format(W.shape, b.shape, x_train.shape))
        logging.info('training...')
        start_training = time()
        W_opt = self.training(x_train, W, b, self.tan_sigmoid, y_train)
        end_training = time()
        self.training_time = utils.timer(start_training, end_training, 'Training time')

        logging.info('testing...')
        for i in range(len(x_test)):
            y_pred = self.predicting(x_test[i], W, b, self.tan_sigmoid, W_opt)
            error = mean_squared_error(y_test[i], y_pred)
            self.predictions.append(y_pred)
            self.mse.append(error)
            self.true.append(y_test[i])
            print("finished test: {}".format(i))



