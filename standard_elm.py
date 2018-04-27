import data
import utils
import os
import logging
import numpy as np
import utils
import scipy.ndimage as ndimage
import sys
import create_features as cf
import nn_functions as nnf
from keras.wrappers.scikit_learn import KerasRegressor
from time import time
plot_folder = './plots/'
Npoints_coarse2D = 256
Npoints_coarse3D = 64



# Load in 2D velocity data
velocity = data.load_data()
# data.example_of_data(velocity)
# form testing and training sets for velocity data
X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity)


def standard_elm_func(x_train, y_train, x_test, y_test):
    nn_structure = [9, 100, 1]
    y_predicted = np.empty_like(y_test)
    n = 3*x_train['u'].size
    logging.info('transforming dictionaries to input')
    x, y = utils.transform_dict_for_nn(X_train, y_train, nn_structure[0])
    x = np.transpose(x)
    y = np.transpose([y])
    print("original shape", x.shape)
    tr_set = np.concatenate((y, x), 1)[:1000] #standard format for elm function - y_train + x_train
    print(tr_set.shape)
    
    logging.info('training...')
   
    # create a classifier
    elmk = elm.ELMKernel()
    #training
    start_training = time()
    tr_result = elmk.train(tr_set)
    end_training = time()
    utils.timer(start_training, end_training, 'Training time')
    
   
    

    #print("shape of y_predicted", y_predicted.shape)
    logging.info('testing...')
    for i in range(len(x_test)):
        x, y = utils.transform_dict_for_nn(x_test[i], y_test[i], nn_structure[0] )
        x = np.transpose(x)
        y = np.transpose([y])
        te_set = np.concatenate((y, x), 1)[:1000]
        print("test shape",te_set.shape)
        te_result =  elmk.test(te_set)
        y_predicted[i] = te_result.predicted_targets  
    print("y_predicted \n")
    #print(y_predicted)
    return y_predicted
    
