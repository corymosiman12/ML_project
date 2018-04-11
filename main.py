import data
import logging
import numpy as np
import scipy.ndimage as ndimage
import sys
import create_features as cf
import nn_functions as nnf
import nn_keras as nnk
from keras.wrappers.scikit_learn import KerasRegressor
# import 

def main():

    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))

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