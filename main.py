import data
import logging
import numpy as np
import scipy.ndimage as ndimage
import sys

def main():

    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
    logging.info('platform {}'.format(sys.platform))
    logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    logging.info('numpy {}'.format(np.__version__))
    logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))


    velocity = data.load_data()
    # data.example_of_data(velocity)
    x_train, y_train, x_test, y_test = data.form_train_test_sets(velocity)

if __name__ == '__main__':
    main()