import data
import logging
import numpy as np
import extreme_learning_machine as elm
import sys
import plotting
import utils
import os
import nn_keras as nnk

def main():

    np.random.seed(1234)

    # Define base variables
    Npoints_coarse2D = 256
    Npoints_coarse3D = 64
    data_output_folder_base = './data_output/'
    plot_folder_base = './plots/'
    data_output_folder = data_output_folder_base
    plot_folder = plot_folder_base

    # Set logging configuration
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.INFO)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

    ########################## DEFINE MODEL ##########################
    # Choose which model you will use for the analysis:
    # FF_1L = Feed forward single layer with keras
    # FF_2L = Feed forward two layer with keras
    # Olga_ELM = Extreme learning machine created by Olga
    model_type = 'FF_1L'
    assert model_type == 'FF_1L' \
        or model_type == 'FF_2L' \
        or model_type == 'Olga_ELM', 'Incorrect model_type: %r' % model_type

    # Define the number of inputs to be used for creating the feature vectors.  See below for requirements:
    """
    FF_1L:      num_neurons_L2 = None
    FF_2L:      Define all
    Olga_ELM:   num_epochs = None, num_neurons_L2 = None
    Rahul_ELM:  num_epochs = None, num_neurons_L2 = None
    """
    num_features = 27    # pass as single integer
    assert num_features == 9 or num_features == 27, 'Incorrect number of features: %r' % num_features
    # num_epochs = [1, 10, 15] # pass as list to iterate through or None
    num_epochs = [50]
    # num_neurons_L1 = [5]
    num_neurons_L1 = list(np.arange(20, 210, 20))   # pass as list to iterate through
    # num_neurons_L1 = [15, 21]
    num_neurons_L2 = [5, 6]     # pass as list to iterate through or None

    # Define activation function to use for 'FF_1L' and 'FF_2L'
    activation_function = 'tanh'
    assert activation_function == 'relu' \
        or activation_function == 'tanh' \
        or activation_function == 'sigmoid', 'Incorrect activation function: %r' % num_features
    if model_type == 'FF_1L' or model_type == 'FF_2L':
        logging.info("Using {} activation function".format(activation_function))
    ########################## FORMAT TRAINING AND TESTING ##########################
    # Select number of dimensions to use for analysis: 2 or 3
    dimension = 3
    assert dimension == 2 or dimension == 3, 'Incorrect number of dimensions: %r' % dimension
    if dimension == 3:
        assert num_features == 27, 'Incorrect number of features for 3D: %r' % num_features
    # Select filter type to use: gaussian, median, or noise
    filter_type = "physical_sharp"
    assert filter_type == "gaussian" \
        or filter_type == "median" \
        or filter_type == "noise" \
        or filter_type == "fourier_sharp" \
        or filter_type == "physical_sharp", \
        'Incorrect filter type: %r' % filter_type

    # Define arguments based on required dimensions
    if dimension == 2:
        Npoints_coarse = Npoints_coarse2D
    else:
        Npoints_coarse = Npoints_coarse3D

    # Update plot folder with model, dimensions, filter_type, num_features
    if model_type == 'FF_1L' or model_type == 'FF_2L':
        plot_folder = os.path.join(plot_folder, "{}".format(model_type),
                        "{}dim_{}_{}feat_{}".format(dimension, filter_type, num_features, activation_function))
    else:
        plot_folder = os.path.join(plot_folder, "{}".format(model_type),
                                "{}dim_{}_{}feat".format(dimension, filter_type, num_features))
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    # Load in data
    velocity = data.load_data(dimension)

    # Form train and test sets. Below just applies filter, keeping it in shapes of [256, 256] or [64, 64, 64]
    X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity, Npoints_coarse=Npoints_coarse, filter_type=filter_type)

    # Reshape training into arrays of observations (rows) and features (columns)
    # Observations are of all components of velocity (u, v, and w)
    X_train_final, y_train_final = utils.final_transform(X_train, y_train, n_features=num_features,
                                                        dimension=dimension, train=True)

    # Reshape testing into list based on sigma [0.9, 1, 1.1] of arrays
    # of observations (rows) and features (columns)
    # Observations are of all components of velocity (u, v, and w)
    X_test_final, y_test_final = utils.final_transform(X_test, y_test, n_features=num_features,
                                                        dimension=dimension)

    ########################## RUN MODEL ##########################
    predictions, mse = utils.run_all(model_type, X_train_final, y_train_final, X_test_final, y_test_final,
                                    num_features, num_epochs, num_neurons_L1, num_neurons_L2, plot_folder,
                                    X_test, y_test, dimension, activation_function)


if __name__ == '__main__':
    main()


########################## NOTES ##########################
    """
    X_test_final and y_test_final are both lists of dictionaries:
        X_test_final[0]: Filter applied with sigma = 1
        X_test_final[1]: Filter applied with sigma = 1.1
        X_test_final[2]: Filter applied with sigma = 0.9
        And similar for y_test_final
    Each list element has a dictionary with 3 keys: 'u', 'v', 'w'. The shapes are equivalent to the training set:
        X_test_final[0]['u'].shape = (256*256, n_features)
        y_test_final[0]['u'].shape = (256*256, 1)
    """
    """
    X_train: dictionary
        dict.keys(): 'u', 'v', 'w'
        dict.values():
            - 256x256 randomly created array from 2048x2048 array using shifting strategy
            - Random filter applied with sigma = 1
    y_train: dictionary
        dict.keys(): 'u', 'v', 'w'
        dict.values():
            - 256x256 randomly create array from 2048x2048
            - y_train = X_train before filter applied (i.e. NN tries to recover unfiltered)
    X_test: list of dictionaries where dict.keys() and dict.values() are same as X_train, except:
        X_test[0]: Filter applied with sigma = 1
        X_test[1]: Filter applied with sigma = 1.1
        X_test[2]: Filter applied with sigma = 0.9
    y_test: list of dictionaries
        - y_test = X_test before filters applied
    """

    # logging.info('\nNN is Extreme learning machine (algorithm from the paper)\n')
    # y_predict = elm.extreme_learning_machine(X_train, y_train, X_test, y_test)
    # plotting.plot_velocities_and_spectra(X_test, y_test, y_predict, plot_folder)
    # plotting.plot_vorticity_pdf(X_test, y_test, y_predict, plot_folder)