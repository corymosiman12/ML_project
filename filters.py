import logging
import numpy as np
from numpy.fft import fftfreq, fftn, ifftn

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