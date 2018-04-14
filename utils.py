import numpy as np
from numpy.fft import fftfreq, fft2
import logging

def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>2}:{:05.2f} \t {}".format(int(minutes), seconds, label))

def shell_average_2D(spect2D, N_point, k_2d):
    """ Compute the 1D, shell-averaged, spectrum of the 2D Fourier-space
    variable.
    :param spect2D: 2-dimensional complex or real Fourier-space scalar
    :return: 1D, shell-averaged, spectrum
    """
    i = 0
    F_k = np.zeros(N_point[0]*N_point[1])
    k_array = np.empty_like(F_k)
    for ind_x, kx in enumerate(k_2d[0]):
        for ind_y, ky in enumerate(k_2d[1]):
                k_array[i] = round(np.sqrt(kx**2 + ky**2))
                F_k[i] = np.pi*k_array[i]*spect2D[ind_x, ind_y]
                i += 1
    all_F_k = sorted(list(zip(k_array, F_k)))

    x, y = [all_F_k[0][0]], [all_F_k[0][1]]
    n = 1
    for k, F in all_F_k[1:]:
        if k == x[-1]:
            n += 1
            y[-1] += F
        else:
            y[-1] /= n
            x.append(k)
            y.append(F)
            n = 1
    return x, y


def spectral_density(vel_array, dx, N_points, fname):
    """
    Write the 1D power spectral density of var to text file. Method
    assumes a real input in physical space.
    """
    k = 2*np.pi*np.array([fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1])])

    spect2d = 0
    for array in vel_array:
        fft_array = fft2(array)
        spect2d += np.real(fft_array * np.conj(fft_array))
    logging.debug('done transform')
    x, y = shell_average_2D(spect2d, N_points, k)
    logging.debug('done shell average')
    fh = open(fname + '.spectra', 'w')
    fh.writelines(["%s\n" % item for item in y])
    fh.close()


def sparse_array(data_value, n_coarse_points, start):
    """ Takes the velocity array (a 2048x2048 array),
    and using the number of coarse points defined (256),
    randomly selects an initial starting row and column index,
    selects every nth point (8 in our case),
    then returns a smaller array (256,256)
    :param data_value:      velocity array (2048x2048 array)
    :param n_coarse_points: number of coarse points
    :param start:           starting point
    :return:                smaller array (256,256)
    """
    if data_value.shape[0] % n_coarse_points:
        logging.warning('Error: sparse_dict(): Nonzero remainder')

    n_th = int(data_value.shape[0] / n_coarse_points)
    i = int(start % n_th)
    j = int(start // n_th)

    # From point i (row start point) and j (column start point), 
    # to the end of the array take each nth point
    sparse_data = data_value[i::n_th, j::n_th].copy()

    # returns a array that is n_coarse_points by n_coarse_points
    return sparse_data


def sparse_dict(data_dict, n_coarse_points, start):
    ''' Takes the velocity dictionary (where each key `u, v, w` corresponds to a 2048x2048 array),
    and call sparse_arraysparse_array(value, n_coarse_points, start) for each key in dictionary
    :param data_dict:       velocity dictionary (where each key `u, v, w` corresponds to a 2048x2048 array)
    :param n_coarse_points: number of coarse points
    :param start:           initial starting point
    :return:                a dictionary of smaller arrays (256,256) with the same keys (`u, v, w`)
    '''

    sparse = dict()
    for key, value in data_dict.items():
        sparse[key] = sparse_array(value, n_coarse_points, start)
    return sparse


def transform_dict_for_nn(x_dict, y_dict, n_input):

    n = x_dict['u'].size   # 256^3 = 16777216
    y = np.empty(3 * n)
    x_size = x_dict['u'].shape[0]
    y_size = x_dict['u'].shape[1]
    logging.info('Use {} inputs'.format(n_input))
    x = np.empty((n_input, 3 * n))  # 9 * number of examples (256*256)*number of velocity components
    keys = ['w', 'u', 'v', 'w', 'u']

    count = 0
    for key_i in range(1, 4):
        for i in range(x_size):
            for j in range(y_size):
                # stencil 5*5
                if n_input == 25:
                    if i == (x_size - 2):
                        x_ind = np.array([i - 2, i - 2, i - 2, i - 2, i - 2,
                                          i - 1, i - 1, i - 1, i - 1, i - 1,
                                          i, i, i, i, i,
                                          i + 1, i + 1, i + 1, i + 1, i + 1,
                                          0, 0, 0, 0, 0])
                    elif i == (x_size - 1):
                        x_ind = np.array([i - 2, i - 2, i - 2, i - 2, i - 2,
                                          i - 1, i - 1, i - 1, i - 1, i - 1,
                                          i, i, i, i, i,
                                          0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1])

                    else:
                        x_ind = np.array([i - 2, i - 2, i - 2, i - 2, i - 2,
                                          i - 1, i - 1, i - 1, i - 1, i - 1,
                                          i, i, i, i, i,
                                          i + 1, i + 1, i + 1, i + 1, i + 1,
                                          i + 2, i + 2, i + 2, i + 2, i + 2])
                    # y index
                    if j == (y_size - 2):
                        y_ind = np.array([j - 2, j - 1, j, j + 1, 0,
                                          j - 2, j - 1, j, j + 1, 0,
                                          j - 2, j - 1, j, j + 1, 0])
                    elif j == (y_size - 1):
                        y_ind = np.array([j - 2, j - 1, j, 0, 1,
                                          j - 2, j - 1, j, 0, 1,
                                          j - 2, j - 1, j, 0, 1])
                    else:
                        y_ind = np.array([j - 2, j - 1, j, j + 1, j + 2,
                                          j - 2, j - 1, j, j + 1, j + 2,
                                          j - 2, j - 1, j, j + 1, j + 2])

                # stencil 3*3
                else:
                    if i == (x_size - 1):
                        x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, 0, 0, 0])
                    else:
                        x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, i + 1, i + 1, i + 1])
                    if j == (y_size - 1):
                        y_ind = np.array([j - 1, j, 0, j - 1, j, 0, j - 1, j, 0])
                    else:
                        y_ind = np.array([j - 1, j, j + 1, j - 1, j, j + 1, j - 1, j, j + 1])

                ind = count * n + i * x_size + j

                if n_input == 9:
                    x[:, ind] = x_dict[keys[key_i]][x_ind, y_ind]
                elif n_input == 27:
                    x[:, ind] = np.hstack((x_dict[keys[key_i]][x_ind, y_ind],
                                           x_dict[keys[key_i-1]][x_ind, y_ind],
                                           x_dict[keys[key_i+1]][x_ind, y_ind]))
                y[ind] = y_dict[keys[key_i]][i, j]
        count += 1

    return x, y


def untransform_y(y, shape):

    keys = ['u', 'v', 'w']
    n = shape[0]*shape[1]
    y_dict = dict({'u': np.empty(shape), 'v': np.empty(shape), 'w': np.empty(shape)})

    for ind in range(len(y)):
        k = ind // n
        i = (ind % n) // shape[0]
        j = (ind % n) % shape[0]
        y_dict[keys[k]][i, j] = y[ind]
    return y_dict