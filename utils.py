import sys
import numpy as np
from numpy.fft import fftfreq, fft2
import logging

def shell_average_2D(spect2D, N_point, k_2d):
    """ Compute the 1D, shell-averaged, spectrum of the 2D Fourier-space
    variable.
    :param spect2D: 2-dimensional complex or real Fourier-space scalar
    :param km:  wavemode of each n-D wavevector
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

def final_transform(X, y, n_features, train=False, index=False):
    """
    To get values by index: [begin_index:end_index]
        u: [:256*256]
        v: [256*256:256*256*2]
        w: [256*256*2:]
    """
    i = 256*256
    if train:
        """
        transform_dict_for_nn() returns:
            1. array with shape = (n_inputs, 256*256*3)
            2. vector with shape = (256*256*3, )
        tr = transformed
        """
        X, y = transform_dict_for_nn(X, y, n_features)
        if index.lower() == 'u':
            X = X[:, :i].T
            y = y[:i].reshape(i, 1)
            return (X, y)
        elif index.lower() == 'v':
            X = X[:, i:i*2].T
            y = y[i:i*2].reshape(i, 1)
            return (X, y)
        elif index.lower() == 'w':
            X = X[:, i*2:].T
            y = y[i*2:].reshape(i, 1)
            return (X, y)
        else:
            logging.fatal('Invalid index to train on. Enter "u", "v", "w"')
            sys.exit()
    else:
        """
        X_test_0_tr, y_test_0_tr: sigma = 1 filtered 
        X_text_1_tr, y_test_1_tr: sigma = 1.1 filtered
        X_test_2_tr, y_test_2_tr: sigma = 0.9 filtered
        """
        X_test_0_tr, y_test_0_tr = transform_dict_for_nn(X[0], y[0], n_features)
        X_test_1_tr, y_test_1_tr = transform_dict_for_nn(X[1], y[1], n_features)
        X_test_2_tr, y_test_2_tr = transform_dict_for_nn(X[2], y[2], n_features)

        X_test_0_u = X_test_0_tr[:, :i].T
        y_test_0_u = y_test_0_tr[:i].reshape(i, 1)
        X_test_0_v = X_test_0_tr[:, i:i*2].T
        y_test_0_v = y_test_0_tr[i:i*2].reshape(i, 1)
        X_test_0_w = X_test_0_tr[:, i*2:].T
        y_test_0_w = y_test_0_tr[i*2:].reshape(i, 1)

        X_test_0 = {'u': X_test_0_u, 'v': X_test_0_v, 'w': X_test_0_w}
        y_test_0 = {'u': y_test_0_u,'v': y_test_0_v,'w': y_test_0_w}
        
        X_test_1_u = X_test_1_tr[:, :i].T
        y_test_1_u = y_test_1_tr[:i].reshape(i, 1)
        X_test_1_v = X_test_1_tr[:, i:i*2].T
        y_test_1_v = y_test_1_tr[i:i*2].reshape(i, 1)
        X_test_1_w = X_test_1_tr[:, i*2:].T
        y_test_1_w = y_test_1_tr[i*2:].reshape(i, 1)

        X_test_1 = {'u': X_test_1_u,'v': X_test_1_v,'w': X_test_1_w}
        y_test_1 = {'u': y_test_1_u,'v': y_test_1_v,'w': y_test_1_w}

        X_test_2_u = X_test_2_tr[:, :i].T
        y_test_2_u = y_test_2_tr[:i].reshape(i, 1)
        X_test_2_v = X_test_2_tr[:, i:i*2].T
        y_test_2_v = y_test_2_tr[i:i*2].reshape(i, 1)
        X_test_2_w = X_test_2_tr[:, i*2:].T
        y_test_2_w = y_test_2_tr[i*2:].reshape(i, 1)

        X_test_2 = {'u': X_test_2_u,'v': X_test_2_v,'w': X_test_2_w}
        y_test_2 = {'u': y_test_2_u,'v': y_test_2_v,'w': y_test_2_w}

        X_test = [X_test_0, X_test_1, X_test_2]
        y_test = [y_test_0, y_test_1, y_test_2]

        return (X_test, y_test)
