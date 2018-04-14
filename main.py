import data
import logging
import numpy as np
import extreme_learning_machine as elm
import plotting
import utils
import scipy.ndimage as ndimage
import sys
import create_features as cf
import nn_functions as nnf
# import nn_keras as nnk
# from keras.wrappers.scikit_learn import KerasRegressor
# import 

plot_folder = './plots/'

def main():

    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

    # Load in velocity data
    velocity = data.load_data()
    # data.example_of_data(velocity)
    # form testing and training sets for velocity data
    X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity)

    ########################## OLGA START ##########################
    # logging.info('NN is Extreme learning machine (algorithm from the paper)\n')
    # y_predict = elm.extreme_learning_machine(X_train, y_train, X_test, y_test)
    # plotting.plot_velocities_and_spectra(X_test, y_test, y_predict, plot_folder)

    logging.info('Compare trasform functions')
    x1, y1 = utils.transform_dict_for_nn(X_train, y_train, 9)
    x2 = cf.form_features(X_train)['u']
    y2 = cf.my_reshaper(y_train)['u']
    print(x1.shape, x2.shape, y1.shape, y2.shape)

    x1 = x1[:, :256*256].T
    y1 = y1[:256*256].reshape(256*256, 1)

    print(x1.shape, x2.shape, y1.shape, y2.shape)
    print(False in np.equal(y1, y2))
    print(False in np.equal(x1, x2))

    ind = np.where(x1 != x2)
    print(ind)
    print(x1[ind])
    print(x2[ind])
    # value which supposed to be
    print(X_train['u'][-1, 0], X_train['u'][-1, 1])


    ########################## OLGA END ##########################
    
    ######################################

    # ########################## CORY START ##########################
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
    #
    # # Create single layer model
    # epochs = 5
    # num_neurons = 100
    # model = nnk.my_keras(epochs, num_neurons)
    # model.evaluate_model(X_train_enc['u'], y_train_reshaped['u'], X_test_enc[0]['u'], y_test_reshaped[0]['u'])
    #
    # # Predict on each of the test sets and plot MSE:
    # model.evaluate_test_sets(X_test_enc, y_test_reshaped)
    # ########################## CORY END ##########################

if __name__ == '__main__':
    main()