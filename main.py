import data
import logging
import numpy as np
import scipy.ndimage as ndimage
import sys
import create_features
import nn_functions as nnf

def main():

    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))


    velocity = data.load_data()
    # data.example_of_data(velocity)
    x_train, y_train, x_test, y_test = data.form_train_test_sets(velocity)
    x_train_enc = create_features.form_features(x_train)
    # print(len(x_train_enc.keys()), len(x_train_enc['u'][256].keys()))
    x_test_enc = create_features.form_features(x_test)
    print(type(x_train_enc), type(x_test_enc))

if __name__ == '__main__':
    main()