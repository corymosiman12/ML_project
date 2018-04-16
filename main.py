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
import create_features as cf
import nn_functions as nnf
import nn_keras as nnk
from keras.wrappers.scikit_learn import KerasRegressor
import plotting

plot_folder = './plots/'
Npoints_coarse2D = 256
Npoints_coarse3D = 64



def main():
    plot_folder = './plots/'
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

    # Load in 2D velocity data
    velocity = data.load_data()
    # data.example_of_data(velocity)
    # form testing and training sets for velocity data
    X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity)
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

    # Define the number of inputs to be used for creating the feature vectors
    n_features = 9

    # # Load in 3D velocity data
    # velocity = data.load_data(dimension=3)
    # data.example_of_data(velocity)
    # # form testing and training sets for velocity data
    # X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity, Npoints_coarse3D)

    ########################## OLGA START ##########################
    # logging.info('NN is Extreme learning machine (algorithm from the paper)\n')
    # y_predict = elm.extreme_learning_machine(X_train, y_train, X_test, y_test)
    # plotting.plot_velocities_and_spectra(X_test, y_test, y_predict, plot_folder)

    # logging.info('Compare trasform functions')
    # x1, y1 = utils.transform_dict_for_nn(X_train, y_train, 9)
    # x2 = cf.form_features(X_train)['u']
    # y2 = cf.my_reshaper(y_train)['u']
    # print(x1.shape, x2.shape, y1.shape, y2.shape)
    #
    # x1 = x1[:, :256*256].T
    # y1 = y1[:256*256].reshape(256*256, 1)
    #
    # print(x1.shape, x2.shape, y1.shape, y2.shape)
    # print(False in np.equal(y1, y2))
    # print(False in np.equal(x1, x2))
    #
    y_train_reshaped = cf.my_reshaper(y_train)
    y_test_reshaped = cf.my_reshaper(y_test)
    logging.info("X_train_enc['u'] shape: {}".format(X_train_enc['u'].shape))
    logging.info("y_train_reshaped['u'] shape: {}\n".format(y_train_reshaped['u'].shape))
    ########################## OLGA START ##########################
    # logging.info('NN is Extreme learning machine (algorithm from the paper)\n')
    # y_predict = elm.extreme_learning_machine(X_train, y_train, X_test, y_test)
    # plotting.plot_velocities_and_spectra(X_test, y_test, y_predict, plot_folder)

    # logging.info('Compare trasform functions')
    # x1, y1 = utils.transform_dict_for_nn(X_train, y_train, n_features)
    # x2 = cf.form_features(X_train)['u']
    # y2 = cf.my_reshaper(y_train)['u']
    # print(x1.shape, x2.shape, y1.shape, y2.shape)

    # x1 = x1[:, :256*256].T
    # y1 = y1[:256*256].reshape(256*256, 1)

    # print(x1.shape, x2.shape, y1.shape, y2.shape)
    # print(False in np.equal(y1, y2))
    # print(False in np.equal(x1, x2))

    # ind = np.where(x1 != x2)
    # print(ind)
    # print(x1[ind])
    # print(x2[ind])
    # # value which supposed to be
    # print(X_train['u'][-1, 0], X_train['u'][-1, 1])


    # ########################## CORY START ##########################

    # Create single layer model
    # Define the number of inputs to be used for creating the feature vectors
    n_features = [9, 27]
    num_epochs = [5, 10, 15]
    num_neurons = [50, 100, 150]
    
    for features in n_features:
        # Define on what your model will be trained: u, v, or w
        key = 'u'

        # X_train_final.shape = (256*256, n_features)
        # y_train_final.shape = (256*256, 1)
        X_train_final, y_train_final = utils.final_transform(X_train, y_train, n_features = features, train=True, index= key)

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
        X_test_final, y_test_final = utils.final_transform(X_test, y_test, features)
        for epochs in num_epochs:
            for neurons in num_neurons:
                logging.info('Evaluating model for {} features, {} epochs, and {} neurons'.format(str(features), str(epochs), str(neurons)))

                # Create folder for plots
                plot_folder = './plots/'
                plot_folder = os.path.join(plot_folder, '{}_features'.format(str(features)),
                                                        '{}_neurons'.format(str(neurons)), 
                                                        '{}_epochs'.format(str(epochs)))
                if not os.path.isdir(plot_folder):
                    os.makedirs(plot_folder)
                
                model = nnk.my_keras(epochs, neurons, features)

                # Evaluate model, validating on same test set key as trained on
                model.evaluate_model(X_train_final, y_train_final, X_test_final[0][key], y_test_final[0][key], plot_folder)
                
                # Predict on each of the test sets and plot MSE:
                # MSE plotting currently not working
                model.evaluate_test_sets(X_test_final, y_test_final)

                plotting.plot_velocities_and_spectra(X_test, y_test, model.predictions, plot_folder)
    # ########################## CORY END ##########################

if __name__ == '__main__':
    main()




# ########################## LEFTOVERS ##########################

# # x_train_enc = create_features.form_features(x_train)
# # print(len(x_train_enc.keys()), len(x_train_enc['u'][256].keys()))
# # x_test_enc = create_features.form_features(x_test)
# # print(type(x_train_enc), type(x_test_enc))
#
# # reformat testing and training sets into true feature vectors
# # note: feature vectors stored within dict()
# X_train_enc = cf.form_features(X_train)
# X_test_enc = cf.form_features(X_test)
#
# y_train_reshaped = cf.my_reshaper(y_train)
# y_test_reshaped = cf.my_reshaper(y_test)
# logging.info("X_train_enc['u'] shape: {}".format(X_train_enc['u'].shape))
# logging.info("y_train_reshaped['u'] shape: {}\n".format(y_train_reshaped['u'].shape))