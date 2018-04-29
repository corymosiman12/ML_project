import numpy as np
import logging
import scipy.ndimage as ndimage
import plotting
import utils
import filters
import os


data_folder = '../data/'
# plot_folder = './plots/'
filename_npz = data_folder + 'isotropic2048.npz'
filename_3d_u = data_folder + 'HIT_u.bin'
filename_3d_v = data_folder + 'HIT_v.bin'
filename_3d_w = data_folder + 'HIT_w.bin'
filename_cvs = None
# filename_cvs = '../data/isotropic2048.csv'

# Npoints_fine = 2048

# dx_fine = np.divide([np.pi, np.pi], [Npoints_fine, Npoints_fine])
# dx_coarse = np.divide([np.pi, np.pi], [256, 256])

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
        max_value = max(np.max(np.abs(velocity['u'])), np.max(np.abs(velocity['v'])), np.max(np.abs(velocity['w'])))
        for key, value in velocity.items():
            velocity[key] = np.swapaxes(value, 0, 2)    # to put x index in first place
            velocity[key] /= max_value

    return velocity


def example_of_data(velocity, Npoints_coarse=256, plot_folder='./plots/'):

    logging.info('Example of Coarse data')
    dimension = len(velocity['u'].shape)
    if dimension == 3:
        k_cutoff = 4
    else:
        k_cutoff = 15
    # utils.spectral_density(velocity, plot_folder+'fine_grid')
    plot_folder = os.path.join(plot_folder, 'spectra_{}D/'.format(dimension))
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)

    vel_coarse = utils.sparse_dict(velocity, Npoints_coarse, 0)
    utils.spectral_density(vel_coarse, plot_folder+'coarse_grid')

    logging.info('Data filtering')
    vel_filtered = dict()

    logging.info('Gaussian')
    for key, value in vel_coarse.items():
        vel_filtered[key] = ndimage.gaussian_filter(value, sigma=1,  mode='wrap', truncate=500)
    utils.spectral_density(vel_filtered, os.path.join(plot_folder, 'gaussian'))
    logging.info('Noise')
    for key, value in vel_coarse.items():
        kappa = np.random.normal(0, 1, size=value.shape)
        vel_filtered[key] = value + 0.2 * kappa
    utils.spectral_density(vel_filtered,os.path.join(plot_folder, 'noise'))
    logging.info('Sharp in Fourier space')
    vel_filtered = filters.filter_sharp(vel_coarse, filter_type='fourier_sharp', scale_k=k_cutoff)
    utils.spectral_density(vel_filtered, os.path.join(plot_folder, 'fourier_sharp'))
    logging.info('Sharp in Physical space')
    vel_filtered = filters.filter_sharp(vel_coarse, filter_type='physical_sharp', scale_k=k_cutoff)
    utils.spectral_density(vel_filtered, os.path.join(plot_folder, 'physical_sharp'))
    logging.info('Median')
    for key, value in vel_coarse.items():
        vel_filtered[key] = ndimage.median_filter(value, size=4,  mode='wrap')
    utils.spectral_density(vel_filtered, os.path.join(plot_folder, 'median'))

    logging.info('Plot data')
    # plotting.imagesc([vel_coarse['u'], vel_filtered['u']], ['true', 'filtered'], plot_folder + 'data')

    # plotting.imagesc([velocity['u'], velocity['v'], velocity['w']], [r'$u$', r'$v$', r'$w$'], plot_folder + 'fine_data')
    # plotting.imagesc([vel_coarse['u'], vel_coarse['v'], vel_coarse['w']], [r'$u$', r'$v$', r'$w$'], plot_folder + 'coarse_data')
    # plotting.imagesc([vel_filtered['u'], vel_filtered['v'], vel_filtered['w']], ['R', 'G', 'B'],
    # plot_folder + 'gaussian filter')
    # [r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$']

    logging.info('Plot spectra')
    plotting.spectra(plot_folder, os.path.join(plot_folder, 'spectra_{}D'.format(dimension)), '')


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
        sigma = [1, 1.1, 0.9]
        for key, value in data_train.items():
            filtered_train[key] = ndimage.gaussian_filter(value, sigma=sigma[0],  mode='wrap', truncate=500)
        for i in range(3):
            for key, value in data_test[i].items():
                filtered_test[i][key] = ndimage.gaussian_filter(value, sigma=sigma[i], mode='wrap', truncate=500)

    elif filter_type == 'median':
        s = [4, 5, 3]
        for key, value in data_train.items():
            filtered_train[key] = ndimage.median_filter(value, size=s[0],  mode='wrap')
        for i in range(3):
            for key, value in data_test[i].items():
                filtered_test[i][key] = ndimage.median_filter(value, size=s[i], mode='wrap')

    elif filter_type == 'noise':
        mu = [0.2, 0.22, 0.18]
        for key, value in data_train.items():
            kappa = np.random.normal(0, 1, size=value.shape)
            filtered_train[key] = value + mu[0]*kappa
        for i in range(3):
            for key, value in data_test[i].items():
                kappa = np.random.normal(0, 1, size=value.shape)
                filtered_test[i][key] = value + mu[i]*kappa

    elif filter_type == 'fourier_sharp' or filter_type == 'physical_sharp':
        k = [4, 5, 3]
        filtered_train = filters.filter_sharp(data_train, filter_type=filter_type, scale_k=k[0])
        for i in range(3):
            filtered_test[i] = filters.filter_sharp(data_train, filter_type=filter_type, scale_k=k[i])

    else:
        logging.error('Filter type is not defined.')

    return filtered_train, data_train, filtered_test, data_test
