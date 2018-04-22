import numpy as np
import logging
import utils
from time import time

nn_structure = [27, 100, 1]


def tan_sigmoid(a):
    e_minus2a = np.exp(-2*a)
    return (1 - e_minus2a)/(1 + e_minus2a)


def training(x, W, b, f, y):

    z = W.dot(x) + b        # z^(2) = W^(1)*a^(1) + b^(1)
    a = f(z)                # activation
    W_opt = np.linalg.pinv(a.T) @ y.T
    return W_opt.T


def predicting(x, W, b, f, W_opt):

    z = W.dot(x)+b
    a = f(z)
    y = W_opt.dot(a)
    return y


def extreme_learning_machine(x_train, y_train, x_test, y_test):

    y_predicted = np.empty_like(y_test)
    n = 3*x_train['u'].size

    logging.info('transforming dictionaries to input')
    dimension = len(x_train['u'].shape)
    if dimension == 2:
        transform_in_nn = utils.transform_dict_for_nn
        untransform = utils.untransform_y
    else:
        transform_in_nn = utils.transform_dict_for_nn_3D
        untransform = utils.untransform_y_3D

    x, y = transform_in_nn(x_train, y_train, nn_structure[0])

    tiny = 1e-12
    W = tiny*np.random.random_sample(size=(nn_structure[1], nn_structure[0]))
    b = tiny*np.random.random_sample(size=(nn_structure[1], n))

    logging.info('training...')
    start_training = time()
    W_opt = training(x, W, b, tan_sigmoid, y)
    end_training = time()
    utils.timer(start_training, end_training, 'Training time')

    logging.info('testing...')
    for i in range(len(x_test)):
        x, y = transform_in_nn(x_test[i], y_test[i], nn_structure[0])
        y_pred = predicting(x, W, b, tan_sigmoid, W_opt)
        error = np.linalg.norm(y - y_pred) / n
        y_predicted[i] = untransform(y_pred, y_test[i]['u'].shape)
        print('error', error)

    return y_predicted
