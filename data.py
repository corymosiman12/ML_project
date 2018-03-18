import numpy as np
import logging
import scipy.ndimage as ndimage
import sys

import plotting
import utils


data_folder = '../data/'
plot_folder = './plots/'
filename_npz = data_folder + 'isotropic2048.npz'
filename_cvs = None
# filename_cvs = '../data/isotropic2048.csv'

Npoints_fine = [2048, 2048]
Npoints_coarse = [256, 256]
dx_fine = np.divide([np.pi, np.pi], Npoints_fine)
dx_coarse = np.divide([np.pi, np.pi], Npoints_coarse)

logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.DEBUG)
logging.info('platform {}'.format(sys.platform))
logging.info('python {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
logging.info('numpy {}'.format(np.__version__))
logging.info('64 bit {}\n'.format(sys.maxsize > 2 ** 32))


if filename_cvs:
    # load from csv file
    data = np.genfromtxt(filename_cvs, delimiter=',')[1:]
    # Normalize
    data /= np.max(np.abs(data))
    # split velocities
    velocity = dict()
    velocity['u'] = data[:, 0].reshape((2048, 2048))
    velocity['v'] = data[:, 1].reshape((2048, 2048))
    velocity['w'] = data[:, 2].reshape((2048, 2048))
    # save in .npz (numpy zip) format
    np.savez(filename_npz, u=velocity['u'], v=velocity['v'], w=velocity['w'])
    logging.info('Data saved in {}/isotropic2048.npz'.format(data_folder))
else:
    # load from .npz file (much faster)
    velocity = np.load(filename_npz)

logging.info('Fine data')
plotting.imagesc([velocity['u'], velocity['v'], velocity['w']], [r'$u$', r'$v$', r'$w$'], plot_folder + 'fine_data')
# logging.info('Calculating fine grid spectrum')
# utils.spectral_density([velocity['u'], velocity['v'], velocity['w']], dx_fine, Npoints_fine, plot_folder+'fine_grid')

logging.info('Coarse data')
vel_coarse = utils.sparse_dict(velocity, Npoints_coarse[0])
plotting.imagesc([vel_coarse['u'], vel_coarse['v'], vel_coarse['w']],
                 [r'$u$', r'$v$', r'$w$'], plot_folder + 'coarse_data')
# logging.info('Calculating coarse grid spectrum')
# utils.spectral_density([vel_coarse['u'], vel_coarse['v'], vel_coarse['w']], dx_coarse, Npoints_coarse,
#                        plot_folder+'coarse_grid')

logging.info('Filtering data')
vel_filtered = dict()
for key, value in vel_coarse.items():
    vel_filtered[key] = ndimage.gaussian_filter(value, sigma=1)

plotting.imagesc([vel_filtered['u'], vel_filtered['v'], vel_filtered['w']],
                 [r'$\widetilde{u}$', r'$\widetilde{v}$', r'$\widetilde{w}$'], plot_folder + 'gaussian filter')
# logging.info('Calculating filtered data spectrum')
# utils.spectral_density([vel_filtered['u'], vel_filtered['v'], vel_filtered['w']], dx_coarse, Npoints_coarse,
#                        plot_folder+'filtered')


# plotting.spectra(plot_folder, plot_folder+'spectra')