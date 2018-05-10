import logging
import numpy as np
from numpy.fft import fftfreq, fftn, ifftn
import scipy.ndimage as ndimage


def box_kernel(k, limit):
    """Create 3D array of Tophat filter.
        k - array of wave numbers;
        limit - cutoff wavenumber."""
    if len(k) == 3:     # 3D
        a = np.zeros((len(k[0]), len(k[1]), len(k[2])), dtype=np.float32)
        for indx, kx in enumerate(k[0]):
            for indy, ky in enumerate(k[1]):
                for indz, kz in enumerate(k[2]):
                    a[indx, indy, indz] = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    else:   # 2D
        a = np.zeros((len(k[0]), len(k[1])), dtype=np.float32)
        for indx, kx in enumerate(k[0]):
            for indy, ky in enumerate(k[1]):
                a[indx, indy] = np.sqrt(kx ** 2 + ky ** 2)
    kernel = np.piecewise(a, [a <= limit, a > limit], [1, 0])
    return kernel


def sinc_kernel(k, limit):
    """ Create 3D array of sinc(s) filter (sharp filter in physical space)
    :param k: array of wave numbers;
    :param limit:
    :return: kernel array
    """
    if len(k) == 3:     # 3D
        a = np.zeros((len(k[0]), len(k[1]), len(k[2])), dtype=np.float32)
        for indx, kx in enumerate(k[0]):
            for indy, ky in enumerate(k[1]):
                for indz, kz in enumerate(k[2]):
                    a[indx, indy, indz] = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    else:   # 2D
        a = np.zeros((len(k[0]), len(k[1])), dtype=np.float32)
        for indx, kx in enumerate(k[0]):
            for indy, ky in enumerate(k[1]):
                a[indx, indy] = np.sqrt(kx ** 2 + ky ** 2)

    kernel = np.sinc(a/limit)
    return kernel


def filter_sharp(data, filter_type, scale_k):
    """ Tophat filter in Fourier space for dictionary of 3D arrays.
        data - dictionary of numpy arrays;
        scale_k - wave number, which define size of filter."""

    assert filter_type == "fourier_sharp" \
        or filter_type == "physical_sharp", 'Incorrect filter type: %r' % filter

    dimension = len(data['u'].shape)
    N_points = np.array(data['u'].shape)
    dx = 2*np.pi/N_points

    # FFT
    FFT = dict()
    for key, value in data.items():
        FFT[key] = fftn(value)
    k = []
    for i in range(dimension):
        k.append(fftfreq(N_points[i], dx[i]))
    # Filtering
    if filter_type == 'fourier_sharp':
        kernel = box_kernel(k, scale_k)
    elif filter_type == 'physical_sharp':
        kernel = sinc_kernel(k, scale_k)
    result = dict()
    fft_filtered = dict()
    for key, value in FFT.items():
        fft_filtered[key] = np.multiply(value, kernel)
    FFT.clear()
    for key, value in fft_filtered.items():
        result[key] = ifftn(value).real

    fft_filtered.clear()

    return result

def filter_sharp_array(data_array, filter_type, scale_k):
    """ Sharp filter in Fourier space for arrays.
        data - numpy arrays;
        scale_k - wave number, which define size of filter."""

    assert filter_type == "fourier_sharp" \
        or filter_type == "physical_sharp", 'Incorrect filter type: %r' % filter

    dimension = len(data_array.shape)
    N_points = np.array(data_array.shape)
    dx = 2*np.pi/N_points

    FFT= fftn(data_array)
    k = []
    for i in range(dimension):
        k.append(fftfreq(N_points[i], dx[i]))
    if filter_type == 'fourier_sharp':
        kernel = box_kernel(k, scale_k)
    elif filter_type == 'physical_sharp':
        kernel = sinc_kernel(k, scale_k)

    fft_filtered = np.multiply(FFT, kernel)
    result = ifftn(fft_filtered).real

    return result


def filter(data_array, filter_type, k):

    if filter_type == 'gaussian':
        filtered_array = ndimage.gaussian_filter(data_array, sigma=k,  mode='wrap', truncate=500)
    elif filter_type == 'median':
        filtered_array = ndimage.median_filter(data_array, size=k,  mode='wrap')
    elif filter_type == 'noise':
        kappa = np.random.normal(0, 1, size=data_array.shape)
        filtered_array = data_array + k*kappa
    elif filter_type == 'fourier_sharp' or filter_type == 'physical_sharp':
        filtered_array = filter_sharp_array(data_array, filter_type=filter_type, scale_k=k)
    else:
        logging.error('Filter type is not defined.')
    return filtered_array


def filter_size(filter_type, dim):
    if filter_type == 'gaussian':
        k = [1, 1.1, 0.9]
    elif filter_type == 'median':
        k = [4, 5, 3]
    elif filter_type == 'noise':
        k = [0.2, 0.22, 0.18]
    elif filter_type == 'fourier_sharp' or filter_type == 'physical_sharp':
        if dim == 3:
            k = [4, 5, 3]
        else:
            k = [15, 16, 14]
    else:
        logging.error('Filter type is not defined.')
    return k