import data
import logging
import numpy as np
import extreme_learning_machine as elm
import plotting
import utils
import scipy.ndimage as ndimage
import sys
import create_features
import nn_functions as nnf

plot_folder = './plots/'

def main():

    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))


    velocity = data.load_data()
    # data.example_of_data(velocity)

    #################################
    x_train, y_train, x_test, y_test = data.form_train_test_sets(velocity)
    y_predicted = elm.extreme_learning_machine(x_train, y_train, x_test, y_test)

    plotting.imagesc([y_test[1]['u'], x_test[1]['u'], y_predicted[1]['u']], [r'$u_{true}$', r'$u_{filtered}$',  r'$u_{predicted}$'], plot_folder + 'u_1')
    plotting.imagesc([y_test[1]['v'], x_test[1]['v'], y_predicted[1]['v']],
                     [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'], plot_folder + 'v_1')
    plotting.imagesc([y_test[1]['w'], x_test[1]['w'], y_predicted[1]['w']], [r'$u_{true}$', r'$u_{filtered}$',  r'$u_{predicted}$'], plot_folder + 'w_w')
    utils.spectral_density([y_test[1]['u'], x_test[1]['u'], y_predicted[1]['u']], [2*np.pi/256, 2*np.pi/256], [256, 256], plot_folder+'predicted')
    plotting.spectra(plot_folder, plot_folder+'spectra')
    ######################################

    # x_train_enc = create_features.form_features(x_train)
    # print(len(x_train_enc.keys()), len(x_train_enc['u'][256].keys()))
    # x_test_enc = create_features.form_features(x_test)
    # print(type(x_train_enc), type(x_test_enc))

if __name__ == '__main__':
    main()