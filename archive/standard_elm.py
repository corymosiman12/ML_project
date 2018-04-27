import data
import utils
import os
import logging
import numpy as np
import extreme_learning_machine as elm
import plotting
import utils
import scipy.ndimage as ndimage
import sys
import nn_keras as nnk
from keras.wrappers.scikit_learn import KerasRegressor
import plotting

plot_folder = './plots/'
Npoints_coarse2D = 256
Npoints_coarse3D = 64

def main():

    # Load in 2D velocity data
    velocity = data.load_data()
    # data.example_of_data(velocity)
    # form testing and training sets for velocity data
    X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity)


    # Data transformation
    #print(X_test[0]['u'].shape)
    print("len of y",len(y_test))
    # print("shape of y", y_test.shape)
    #print(y_train)

    #print(X_train['u'].shape)

    import elm as standard_elm
    # create a classifier
    elmk = standard_elm.ELMKernel()
    nn_structure = [9, 100, 1]
    x, y = utils.transform_dict_for_nn(X_train, y_train, nn_structure[0])
    x = np.transpose(x)
    y = np.transpose([y])

    tr_set = np.concatenate((y, x), 1) #standard format for elm function - y_train + x_train

    x_test, y_test = utils.transform_dict_for_nn(X_test[0], y_test[0], nn_structure[0])
    #x_test = np.transpose(x_test)
    #y_test = np.transpose([y_test])

    #te_set = np.concatenate((y_test, x_test), 1)

    # load dataset
    dataa = standard_elm.read("boston.data")

    # create a classifier
    elmk = standard_elm.elmk.ELMKernel()


    # split data in training and testing sets
    # use 80% of dataset to training and shuffle data before splitting
    tr_set, te_set = standard_elm.split_sets(dataa, training_percent=.8, perm=True)

    #train and test
    # results are Error objects
    tr_result = elmk.train(tr_set)
    te_result = elmk.test(te_set)
    print(te_result.get_accuracy())
    te_result.predicted_targets

if __name__ == '__main__':
    main()