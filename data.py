import numpy as np
import logging
import scipy.ndimage as ndimage


import plotting
import utils


data_folder = '../data/'
plot_folder = './plots/'
filename_npz = data_folder + 'isotropic2048.npz'
filename_3d_u = data_folder + 'HIT_u.bin'
filename_3d_v = data_folder + 'HIT_v.bin'
filename_3d_w = data_folder + 'HIT_w.bin'
filename_cvs = None
# filename_cvs = '../data/isotropic2048.csv'

Npoints_fine = 2048

dx_fine = np.divide([np.pi, np.pi], [Npoints_fine, Npoints_fine])
dx_coarse = np.divide([np.pi, np.pi], [256, 256])

# n_sets = 4  # number of subsets for training and test data


def load_data(dimension=2):
    """
    Load data from either .csv or .npz file.
    If you already have the isotropic2048.npz file in your directory,
    use this file to load the data (it is more efficient, do this by using the line filename_cvs = None and
    COMMENTING OUT filename_cvs = '../data/isotropic2048.csv.
    Function can load 3d data as well as 2d. 3d fine data is 256*256*256.
    :param dimension: 2 or 3
    :return: dictionary of 'u', 'v', 'w' data array (256^3 for 3d and 2048^2 for 2d)
    """
    logging.info('Fine data')
    velocity = dict()
    if dimension == 2:
        if filename_cvs:
            # load from csv file
            data = np.genfromtxt(filename_cvs, delimiter=',')[1:]
            # Normalize
            data /= np.max(np.abs(data))
            # split velocities
            velocity['u'] = data[:, 0].reshape((2048, 2048)).T
            velocity['v'] = data[:, 1].reshape((2048, 2048)).T
            velocity['w'] = data[:, 2].reshape((2048, 2048)).T
            # save in .npz (numpy zip) format
            np.savez(filename_npz, u=velocity['u'], v=velocity['v'], w=velocity['w'])
            logging.info('Data saved in {}isotropic2048.npz'.format(data_folder))
        else:
            # load from .npz file (much faster)
            velocity = np.load(filename_npz)
    elif dimension == 3:
        velocity['u'] = np.reshape(np.fromfile(filename_3d_u, dtype=np.float32), (256, 256, 256))
        velocity['v'] = np.reshape(np.fromfile(filename_3d_v, dtype=np.float32), (256, 256, 256))
        velocity['w'] = np.reshape(np.fromfile(filename_3d_w, dtype=np.float32), (256, 256, 256))
        for key, value in velocity.items():
            velocity[key] = np.swapaxes(value, 0, 2) # to put x index in first place

    return velocity


def example_of_data(velocity, Npoints_coarse=256):

    logging.info('Example of Coarse data')
    # dx_fine = np.divide([np.pi, np.pi], [velocity['u'].shape[0], Npoints_fine])
    # dx_coarse = np.divide([np.pi, np.pi], [256, 256])
    vel_coarse = utils.sparse_dict(velocity, Npoints_coarse, 0)

    logging.info('Example of data filtering')
    vel_filtered = dict()
    for key, value in vel_coarse.items():
        vel_filtered[key] = ndimage.gaussian_filter(value, sigma=1)

    logging.info('Plot data')
    # plotting.imagesc([vel_coarse['u'], vel_filtered['u']], ['true', 'filtered'], plot_folder + 'data')

    # plotting.imagesc([velocity['u'], velocity['v'], velocity['w']], [r'$u$', r'$v$', r'$w$'], plot_folder + 'fine_data')
    # plotting.imagesc([vel_coarse['u'], vel_coarse['v'], vel_coarse['w']], [r'$u$', r'$v$', r'$w$'], plot_folder + 'coarse_data')
    # plotting.imagesc([vel_filtered['u'], vel_filtered['v'], vel_filtered['w']], ['R', 'G', 'B'],
    # plot_folder + 'gaussian filter')
    # [r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$']
    # logging.info('Calculating spectra')
    utils.spectral_density(velocity, plot_folder+'fine_grid')
    utils.spectral_density(vel_coarse, plot_folder+'coarse_grid')
    utils.spectral_density(vel_filtered, plot_folder+'filtered')
    plotting.spectra(plot_folder, plot_folder+'spectra', '')


def form_train_test_sets(velocity, Npoints_coarse=256, filter_type='gaussian'):
    """
    Implement the shifting strategy and return 4 data structures:
    data_train, data_test, filtered_train, filtered_test.
    :param velocity:  velocity dictionary
    :param Npoints_coarse: number of points in coarse array (should be 256 in 2d and 64 in 3d)
    (where each key `u, v, w` corresponds to a 2048^2 array for 2d and 256^3 for 3d)
    :return:
    """
    logging.info('Creating training and test data sets')
    dimension = len(velocity['u'].shape)
    Npoints_fine = velocity['u'].shape[0]
    logging.info('Data dimension is {}'.format(dimension))
    if Npoints_fine % Npoints_coarse != 0:
        logging.warning('Warning: Nonzero remainder')

    number_of_examples = int((Npoints_fine/Npoints_coarse)**dimension)
    # training set
    ind = np.random.choice(range(number_of_examples), size=4, replace=False)
    data_train = utils.sparse_dict(velocity, Npoints_coarse, ind[0])  # use first random number to draw training set
    # testing sets
    data_test = []          # I use list of dictionaries.
    for i in ind[1:]:       # use other to random numbers to draw test sets
        data_test.append(utils.sparse_dict(velocity, Npoints_coarse, i))

    # Filtering
    filtered_train = dict()
    filtered_test = [dict(), dict(), dict()]

    if filter_type == 'gaussian':
        for key, value in data_train.items():
            filtered_train[key] = ndimage.gaussian_filter(value, sigma=1,  mode='wrap', truncate=500)
        sigma = [1, 1.1, 0.9]
        for i in range(3):
            for key, value in data_test[i].items():
                filtered_test[i][key] = ndimage.gaussian_filter(value, sigma=sigma[i], mode='wrap', truncate=500)

    elif filter_type == 'median':
        for key, value in data_train.items():
            filtered_train[key] = ndimage.median_filter(value, size=10,  mode='wrap')
        s = [10, 13, 7]
        for i in range(3):
            for key, value in data_test[i].items():
                filtered_test[i][key] = ndimage.median_filter(value, size=s, mode='wrap')

    elif filter_type == 'noise':
        for key, value in data_train.items():
            kappa = np.random.normal(0, 1, size=value.shape)
            filtered_train[key] = value + 0.2*kappa
        mu = [0.2, 0.22, 0.18]
        for i in range(3):
            for key, value in data_test[i].items():
                kappa = np.random.normal(0, 1, size=value.shape)
                filtered_test[i][key] = value + mu[i]*kappa
    else:
        logging.error('Filter type is not defined.')

    return filtered_train, data_train, filtered_test, data_test
