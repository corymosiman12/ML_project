import numpy as np
import logging
import utils
from time import time
import matplotlib.pyplot as plt

nn_structure = [9, 1000, 1]


def tan_sigmoid(a):
    e_minus2a = np.exp(-2*a)
    return (1 - e_minus2a)/(1 + e_minus2a)


def training(x, W, b, f, y):

    z = W.dot(x) + b        # z^(2) = W^(1)*a^(1) + b^(1)
    a = f(z)                # activation
    W_opt = y.dot(np.linalg.pinv(a))
    return W_opt


def predicting(x, W, b, f, W_opt):

    z = W.dot(x)+b
    a = f(z)
    y = W_opt.dot(a)
    return y


def transform_dict_for_nn(x_dict, y_dict):

    n = x_dict['u'].size   # 256^3 = 16777216
    y = np.empty(3 * n)
    x_size = x_dict['u'].shape[0]
    y_size = x_dict['u'].shape[1]
    logging.info('Use {} inputs'.format(nn_structure[0]))
    x = np.empty((nn_structure[0], 3 * n))  # 9 * number of examples (256*256)*number of velocity components
    keys = ['w', 'u', 'v', 'w', 'u']

    count = 0
    for key_i in range(1, 4):
        for i in range(x_size):
            for j in range(y_size):
                if i == (x_size - 1):
                    x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, 0, 0, 0])
                else:
                    x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, i + 1, i + 1, i + 1])
                if j == (y_size - 1):
                    y_ind = np.array([j - 1, j, 0, j - 1, j, 0, j - 1, j, 0])
                else:
                    y_ind = np.array([j - 1, j, j + 1, j - 1, j, j + 1, j - 1, j, j + 1])

                ind = count * n + i * x_size + j

                if nn_structure[0] == 9:
                    x[:, ind] = x_dict[keys[key_i]][x_ind, y_ind]
                elif nn_structure[0] == 27:
                    x[:, ind] = np.hstack((x_dict[keys[key_i]][x_ind, y_ind],
                                           x_dict[keys[key_i-1]][x_ind, y_ind],
                                           x_dict[keys[key_i+1]][x_ind, y_ind]))
                y[ind] = y_dict[keys[key_i]][i, j]
        count += 1

    return x, y


def untransform_y(y, shape):
    keys = ['u', 'v', 'w']
    n = shape[0]*shape[1]
    y_dict = dict({'u': np.empty(shape), 'v': np.empty(shape), 'w': np.empty(shape)})

    for ind in range(len(y)):
        k = ind // n
        i = (ind % n) // shape[0]
        j = (ind % n) % shape[0]
        y_dict[keys[k]][i, j] = y[ind]
    return y_dict


def extreme_learning_machine(x_train, y_train, x_test, y_test):

    y_predicted = np.empty_like(y_test)
    n = 3*x_train['u'].size
    logging.info('transforming dictionaries to input')
    x, y = transform_dict_for_nn(x_train, y_train)
    tiny = 1e-4
    W = tiny*np.random.random_sample((nn_structure[1], nn_structure[0]))
    b = tiny*np.random.random_sample((nn_structure[1], n))
    logging.info('training...')
    start_training = time()
    W_opt = training(x, W, b, tan_sigmoid, y)
    end_training = time()
    utils.timer(start_training, end_training, 'Training time')

    logging.info('testing...')
    for i in range(len(x_test)):
        x, y = transform_dict_for_nn(x_test[i], y_test[i])
        y_pred = predicting(x, W, b, tan_sigmoid, W_opt)
        error = np.linalg.norm(y - y_pred) / n
        y_predicted[i] = untransform_y(y_pred, y_test[i]['u'].shape)
        print('error', error)

    return y_predicted
