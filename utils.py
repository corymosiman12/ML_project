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


def sparse_array(data_value, M, start):

    if data_value.shape[0] % M:
        logging.warning('Error: sparse_dict(): Nonzero remainder')

    n_th = int(data_value.shape[0] / M)
    i = int(start % n_th)
    j = int(start // n_th)
    sparse_data = data_value[i::n_th, j::n_th].copy()
    return sparse_data


def sparse_dict(data_dict, M, start):

    sparse = dict()
    for key, value in data_dict.items():
        sparse[key] = sparse_array(value, M, start)
    return sparse