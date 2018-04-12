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
import nn_keras as nnk
from keras.wrappers.scikit_learn import KerasRegressor
# import 

plot_folder = './plots/'

def main():

    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

    velocity = data.load_data()
    # data.example_of_data(velocity)
    x_train, y_train, x_test, y_test = data.form_train_test_sets(velocity)

    #################################
    logging.info('NN is Extreme learning machine (algorithm from the paper)\n')
    y_predict = elm.extreme_learning_machine(x_train, y_train, x_test, y_test)

    logging.info('Plot predicted velocities')
    for test_example in range(3):
        plotting.imagesc([y_test[test_example]['u'][0:32,0:32],
                          x_test[test_example]['u'][0:32,0:32],
                          y_predict[test_example]['u'][0:32,0:32]],
                         [r'$u_{true}$', r'$u_{filtered}$',  r'$u_{predicted}$'], plot_folder + 'u_'+ str(test_example))
        plotting.imagesc([y_test[test_example]['v'][0:32,0:32],
                          x_test[test_example]['v'][0:32,0:32],
                          y_predict[test_example]['v'][0:32,0:32]],
                         [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'], plot_folder + 'v_'+ str(test_example))
        plotting.imagesc([y_test[test_example]['w'][0:32,0:32],
                          x_test[test_example]['w'][0:32,0:32],
                          y_predict[test_example]['w'][0:32,0:32]],
                         [r'$u_{true}$', r'$u_{filtered}$',  r'$u_{predicted}$'], plot_folder + 'w_'+ str(test_example))

        logging.info('Calculate ang plot spectra')
        utils.spectral_density([y_test[test_example]['u'], y_test[test_example]['v'], y_test[test_example]['w']],
                               [2*np.pi/256, 2*np.pi/256], [256, 256], plot_folder+'true' + str(test_example))
        utils.spectral_density([x_test[test_example]['u'], x_test[test_example]['v'], x_test[test_example]['w']],
                               [2*np.pi/256, 2*np.pi/256], [256, 256], plot_folder+'filtered' + str(test_example))
        utils.spectral_density([y_predict[test_example]['u'], y_predict[test_example]['v'], y_predict[test_example]['w']],
                               [2*np.pi/256, 2*np.pi/256], [256, 256], plot_folder+'predicted' + str(test_example))
        plotting.spectra(plot_folder, plot_folder+'spectra' + str(test_example), test_example)
    ######################################

    # x_train_enc = create_features.form_features(x_train)
    # print(len(x_train_enc.keys()), len(x_train_enc['u'][256].keys()))
    # x_test_enc = create_features.form_features(x_test)
    # print(type(x_train_enc), type(x_test_enc))
    # Load in velocity data
    velocity = data.load_data()

    # form testing and training sets for velocity data
    X_train, y_train, X_test, y_test = data.form_train_test_sets(velocity)

    # reformat testing and training sets into true feature vectors
    # note: feature vectors stored within dict()
    X_train_enc = cf.form_features(X_train)
    X_test_enc = cf.form_features(X_test)

    y_train_reshaped = cf.my_reshaper(y_train)
    y_test_reshaped = cf.my_reshaper(y_test)
    logging.info("X_train_enc['u'] shape: {}".format(X_train_enc['u'].shape))
    logging.info("y_train_reshaped['u'] shape: {}\n".format(y_train_reshaped['u'].shape))

    # Create single layer model
    epochs = 5
    num_neurons = 100
    model = nnk.my_keras(epochs, num_neurons)
    model.evaluate_model(X_train_enc['u'], y_train_reshaped['u'], X_test_enc[0]['u'], y_test_reshaped[0]['u'])

    # Predict on each of the test sets and plot MSE:
    model.evaluate_test_sets(X_test_enc, y_test_reshaped)


if __name__ == '__main__':
    main()