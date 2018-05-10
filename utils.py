import sys
import numpy as np
from numpy.fft import fftfreq, fft2, fftn
import logging
import nn_keras as nnk
import extreme_learning_machine as olga_elm
# import elm as standard_elm
import plotting
import os


def timer(start, end, label):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("{:0>2}:{:05.2f} \t {}".format(int(minutes), seconds, label))
    return "{:0>2}:{:05.2f}".format(int(minutes), seconds)


def pdf_from_array_with_x(array, bins, range):
    pdf, edges = np.histogram(array, bins=bins, range=range, normed=1)
    x = (edges[1:] + edges[:-1]) / 2
    return x, pdf


def calc_vorticity_magnitude(vel_dict):
    shape = vel_dict['u'].shape
    assert len(shape) == 3, "Incorrect dimension for vorticity calculation"
    dx = np.divide([np.pi] * 3, np.array(shape))

    vorticity = np.empty((3, shape[0], shape[1], shape[2]))
    du_dx, du_dy, du_dz = np.gradient(vel_dict['u'], dx[0], dx[1], dx[2])
    dv_dx, dv_dy, dv_dz = np.gradient(vel_dict['v'], dx[0], dx[1], dx[2])
    dw_dx, dw_dy, dw_dz = np.gradient(vel_dict['w'], dx[0], dx[1], dx[2])
    vorticity[0] = dw_dy - dv_dz
    vorticity[1] = du_dz - dw_dx
    vorticity[2] = dv_dx - du_dy
    vorticity /= np.max(np.abs(vorticity))
    # vort_magnitude = np.sqrt(vorticity[0] ** 2 + vorticity[1] ** 2 + vorticity[2] ** 2)
    return vorticity

def shell_average(spect, N_points, k):
    """ Compute the 1D, shell-averaged, spectrum of the 2D or 3D Fourier-space
    variable.
    :param spect: 2D or 3D complex or real Fourier-space scalar
    :return: 1D, shell-averaged, spectrum
    """
    i = 0
    F_k = np.zeros(tuple(N_points)).flatten()
    k_array = np.empty_like(F_k)
    for ind_x, kx in enumerate(k[0]):
        for ind_y, ky in enumerate(k[1]):
            if len(N_points) == 2:
                k_array[i] = round(np.sqrt(kx**2 + ky**2))
                F_k[i] = np.pi*k_array[i]*spect[ind_x, ind_y]
                i += 1
            else:
                for ind_z, kz in enumerate(k[2]):
                    k_array[i] = round(np.sqrt(kx ** 2 + ky ** 2 + kz ** 2))
                    F_k[i] = 2 * np.pi * k_array[i] ** 2 * spect[ind_x, ind_y, ind_z]
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


def spectral_density(vel_dict, fname):
    """
    Write the 1D power spectral density of var to text file. Method
    assumes a real input in physical space.
    """
    N_points = np.array(vel_dict['u'].shape)
    dx = 2*np.pi/N_points
    spectrum = 0
    if len(N_points) == 2:
        k = 2 * np.pi * np.array([fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1])])
        for key, array in vel_dict.items():
            fft_array = fft2(array)
            spectrum += np.real(fft_array * np.conj(fft_array))
    elif len(N_points) == 3:
        k = 2 * np.pi * np.array(
            [fftfreq(N_points[0], dx[0]), fftfreq(N_points[1], dx[1]), fftfreq(N_points[2], dx[2])])
        for key, array in vel_dict.items():
            fft_array = fftn(array)
            spectrum += np.real(fft_array * np.conj(fft_array))

    # logging.debug('done transform')
    x, y = shell_average(spectrum, N_points, k)
    # logging.debug('done shell average')

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
    if int(data_value.shape[0] / n_coarse_points) == 1:
        return data_value

    n_th = int(data_value.shape[0] / n_coarse_points)

    if len(data_value.shape) == 2:
        i = int(start % n_th)
        j = int(start // n_th)
        # From point i (row start point) and j (column start point),
        # to the end of the array take each nth point
        sparse_data = data_value[i::n_th, j::n_th].copy()

        # returns a array that is n_coarse_points by n_coarse_points
    elif len(data_value.shape) == 3:
        k = int(start % n_th)
        j = int(start // n_th % n_th)
        i = int(start // n_th // n_th)
        sparse_data = data_value[i::n_th, j::n_th, k::n_th].copy()

    return sparse_data


def sparse_dict(data_dict, n_coarse_points, start):
    """Takes the velocity dictionary (where each key `u, v, w` corresponds to a 2048x2048 array),
    and call sparse_arraysparse_array(value, n_coarse_points, start) for each key in dictionary
    :param data_dict:       velocity dictionary (where each key `u, v, w` corresponds to a 2048x2048 array)
    :param n_coarse_points: number of coarse points
    :param start:           initial starting point
    :return:                a dictionary of smaller arrays (256,256) with the same keys (`u, v, w`)
    """

    sparse = dict()
    for key, value in data_dict.items():
        sparse[key] = sparse_array(value, n_coarse_points, start)
    logging.info('Coarse data shape is {}'.format(sparse['u'].shape))
    return sparse


def transform_dict_for_nn(x_dict, y_dict, n_input):
    """ Transform dictionary of 'u', 'v', 'w' into array of form [256*256*3, n_input]
    (combining all 3 velocities in 1 array)
    :param x_dict: dictionary with ['u'], ['v'], ['w'] of training data (filtered velocity)
    :param y_dict: dictionary with ['u'], ['v'], ['w'] of true data
    :param n_input: number of input parameters 9, 27 or 25(5*5 stencil)
    :return: (x, y) where x is array of form [256*256*3, n_input] and y is array of form [256*256*3, 1]
    """

    n = x_dict['u'].size   # 256^2
    y = np.empty(3 * n)
    x_size = x_dict['u'].shape[0]
    y_size = x_dict['u'].shape[1]
    logging.info('Use {} inputs'.format(n_input))
    x = np.empty((n_input, 3 * n))  # 9 * number of examples (256*256)*number of velocity components
    keys = ['w', 'u', 'v', 'w', 'u']

    count = 0
    for key_i in range(1, 4):  # for each velocity u, v, w
        for i in range(x_size):
            for j in range(y_size):
                # stencil 5*5
                if n_input == 25:
                    if i == (x_size - 2):       # need to check if it is corner
                        x_ind = np.array([i - 2, i - 2, i - 2, i - 2, i - 2,
                                          i - 1, i - 1, i - 1, i - 1, i - 1,
                                          i, i, i, i, i,
                                          i + 1, i + 1, i + 1, i + 1, i + 1,
                                          0, 0, 0, 0, 0])
                    elif i == (x_size - 1):     # need to check if it is corner
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
                    if j == (y_size - 2):       # need to check if it is corner
                        y_ind = np.array([j - 2, j - 1, j, j + 1, 0,
                                          j - 2, j - 1, j, j + 1, 0,
                                          j - 2, j - 1, j, j + 1, 0])
                    elif j == (y_size - 1):     # need to check if it is corner
                        y_ind = np.array([j - 2, j - 1, j, 0, 1,
                                          j - 2, j - 1, j, 0, 1,
                                          j - 2, j - 1, j, 0, 1])
                    else:
                        y_ind = np.array([j - 2, j - 1, j, j + 1, j + 2,
                                          j - 2, j - 1, j, j + 1, j + 2,
                                          j - 2, j - 1, j, j + 1, j + 2])

                # stencil 3*3
                else:
                    if i == (x_size - 1):       # need to check if it is corner
                        x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, 0, 0, 0])
                    else:
                        x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, i + 1, i + 1, i + 1])
                    if j == (y_size - 1):       # need to check if it is corner
                        y_ind = np.array([j - 1, j, 0, j - 1, j, 0, j - 1, j, 0])
                    else:
                        y_ind = np.array([j - 1, j, j + 1, j - 1, j, j + 1, j - 1, j, j + 1])

                ind = count * n + i * y_size + j

                if n_input == 9:
                    x[:, ind] = x_dict[keys[key_i]][x_ind, y_ind]
                elif n_input == 27:     # if 27 input parameters, combine 3*9 values
                    x[:, ind] = np.hstack((x_dict[keys[key_i]][x_ind, y_ind],
                                           x_dict[keys[key_i-1]][x_ind, y_ind],
                                           x_dict[keys[key_i+1]][x_ind, y_ind]))
                y[ind] = y_dict[keys[key_i]][i, j]
        count += 1

    return x.T, y.reshape(3*n, 1)


def transform_dict_for_nn_3D(x_dict, y_dict, n_input):
    """ Transform dictionary of 'u', 'v', 'w' into array of form [64*64*64*3, n_input]
    (combining all 3 velocities in 1 array)
    :param x_dict: dictionary with ['u'], ['v'], ['w'] of training data (filtered velocity)
    :param y_dict: dictionary with ['u'], ['v'], ['w'] of true data
    :param n_input: number of input parameters 9, 27 or 25(5*5 stencil)
    :return: (x, y) where x is array of form [64*64*64*3, 1] and y is array of form [64*64*64*3, 1]
    """

    n = x_dict['u'].size   # 64^3 = 16777216
    y = np.empty(3 * n)
    size = x_dict['u'].shape
    logging.info('Use {} inputs'.format(n_input))
    x = np.empty((n_input, 3 * n))  # 27 x number of examples (64^3)*number of velocity components
    keys = ['w', 'u', 'v', 'w', 'u']

    count = 0
    for key_i in range(1, 4):  # for each velocity u, v, w
        for i in range(size[0]):
            for j in range(size[1]):
                for k in range(size[2]):
                # stencil 3*3*3
                    if i == (size[0] - 1):       # need to check if it is corner
                        x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, 0, 0, 0]*3)
                    else:
                        x_ind = np.array([i - 1, i - 1, i - 1, i, i, i, i + 1, i + 1, i + 1]*3)
                    if j == (size[1] - 1):       # need to check if it is corner
                        y_ind = np.array([j - 1, j, 0, j - 1, j, 0, j - 1, j, 0]*3)
                    else:
                        y_ind = np.array([j - 1, j, j + 1, j - 1, j, j + 1, j - 1, j, j + 1]*3)
                    if k == (size[2] - 1):       # need to check if it is corner
                        z_ind = np.hstack((np.array([k - 1]*9), np.array([k]*9), np.array([0]*9)))
                    else:
                        z_ind = np.hstack((np.array([k - 1]*9), np.array([k]*9), np.array([k + 1]*9)))

                    ind = count * n + i * size[1]*size[2] + j * size[2] + k

                    x[:, ind] = x_dict[keys[key_i]][x_ind, y_ind, z_ind]
                    y[ind] = y_dict[keys[key_i]][i, j, k]
        count += 1

    return x.T, y.reshape(3*n, 1)


def untransform_y(y, shape):
    """ Trunsform array of form [shape[0]*shape[1]*3] (in our case [256*256*3])
    back into dictionary of 'u', 'v' and 'w' with arrays of form shape ((256, 256)).
    :param y: array of form [256*256*3]
    :param shape: shape of array in output dictionary ((256, 256))
    :return: dictionary of 'u', 'v' and 'w' with arrays of form shape
    """

    keys = ['u', 'v', 'w']
    n = shape[0]*shape[1]
    y_dict = dict({'u': np.empty(shape), 'v': np.empty(shape), 'w': np.empty(shape)})

    for ind in range(len(y)):
        k = ind // n                        # velocity index
        i = (ind % n) // shape[1]           # x inxex
        j = (ind % n) % shape[1]            # y inxex
        y_dict[keys[k]][i, j] = y[ind]
    return y_dict


def untransform_y_3D(y, shape):
    """ Trunsform array of form [shape[0]*shape[1]*shape[2]*3] (in our case [64*64*64*3])
    back into dictionary of 'u', 'v' and 'w' with arrays of form shape ((64, 64, 64)).
    :param y: array of form [64*64*64*3]
    :param shape: shape of array in output dictionary ((64, 64, 64))
    :return: dictionary of 'u', 'v' and 'w' with arrays of form shape
    """

    keys = ['u', 'v', 'w']
    n = shape[0]*shape[1]*shape[2]
    y_dict = dict({'u': np.empty(shape), 'v': np.empty(shape), 'w': np.empty(shape)})

    for ind in range(len(y)):
        v_ind = ind // n                                      # velocity index
        i = (ind % n) // (shape[1] * shape[2])                # x inxex
        j = (ind % n) % (shape[1] * shape[2]) // shape[2]     # y index
        k = (ind % n) % (shape[1] * shape[2]) % shape[2]      # z index
        y_dict[keys[v_ind]][i, j, k] = y[ind]
    return y_dict


def final_transform(X, y, n_features, dimension, train=False):
    """ Return training examples as observations (rows) x features (columns)
    :param X: dict with keys 'u', 'v', 'w' (train) or list of dict with keys 'u', 'v', 'w' (test).
                The value of all dictionaries is a (256 row) x (256 column) array.  Filter/noise
                has already been applied.
    :param y: same as X
    :param n_features: defines the number of features for each training example
    :param train: basically a switcher for the two different data structures (list vs. dict)
    """
    if dimension == 2:
        i = 256*256
        transform_in_nn = transform_dict_for_nn
    else:
        i = 64*64*64
        transform_in_nn = transform_dict_for_nn_3D
    if train:
        X, y = transform_in_nn(X, y, n_features)
        return X, y

    else:
        """
        X_test_0_tr, y_test_0_tr: sigma = 1 filtered 
        X_text_1_tr, y_test_1_tr: sigma = 1.1 filtered
        X_test_2_tr, y_test_2_tr: sigma = 0.9 filtered
        """
        X_test = ['', '', '']
        y_test = ['', '', '']
        for test_case in range(len(X)):
            X_test[test_case], y_test[test_case] = transform_in_nn(X[test_case], y[test_case], n_features)
        
        return X_test, y_test

def save_results(plot_folder, predictions, true, mse):
    header = "sigma=0.9, sigma=1.0, sigma=1.1"
    prediction_file = os.path.join(plot_folder, 'y_predictions.txt')
    true_file = os.path.join(plot_folder, 'y_actual.txt')
    mse_file = os.path.join(plot_folder, 'mse.txt')
    for i in range(len(true)):
        predictions[i] = predictions[i].flatten()
        true[i] = true[i].flatten()
    np.savetxt(prediction_file, predictions, header=header)
    np.savetxt(true_file, true, header=header)
    np.savetxt(mse_file, mse, header=header)


def save_loss_per_epoch(plot_folder, train_loss, val_loss):
    header = "train_loss, val_loss"
    train_loss_array = np.array(train_loss)
    val_loss_array = np.array(val_loss)
    final = np.column_stack((train_loss_array, val_loss_array))
    epoch_file = os.path.join(plot_folder, "loss_per_epoch.txt")
    np.savetxt(epoch_file, final, header=header)


def run_all(model_type, X_train_final, y_train_final, X_test_final, y_test_final,
            num_features, num_epochs, num_neurons_L1, num_neurons_L2, base_plot_folder, 
            X_test, y_test, dimension, activation_function):
    predictions = []
    mse = []
    training_time = []
    if dimension == 2:
        untransform = untransform_y
        shape = [256, 256]
    else:
        untransform = untransform_y_3D
        shape = [64, 64, 64]
    if model_type == 'FF_1L':
        for epochs in num_epochs:
            for neurons in num_neurons_L1:
                logging.info('Evaluating model {} for {} features, {} epochs, and {} neurons'.format(model_type, num_features, epochs, neurons))
    
                # Create folder for plots
                plot_folder = base_plot_folder
                plot_folder = os.path.join(plot_folder, '{}neurons_{}epochs'.format(neurons, epochs))
                if not os.path.isdir(plot_folder):
                    os.makedirs(plot_folder)

                # Initialize model
                model = nnk.my_keras(epochs, neurons, num_features, activation_function)
    
                # Evaluate model, validating on same test set key as trained on
                model.evaluate_model(X_train_final, y_train_final, X_test_final[0], y_test_final[0], plot_folder)

                # Record training time for each model
                training_time.append("{}epochs {}neurons: {}".format(epochs, neurons, model.training_time))

                # Predict on each of the test sets and plot MSE:
                # MSE plotting currently not working
                model.evaluate_test_sets(X_test_final, y_test_final)

                # print(model.predictions.shape, type(model.predictions))
                untransformed_predictions = []
                for p in model.predictions:
                    untransformed_predictions.append(untransform(p, shape))
                plotting.plot_velocities_and_spectra(X_test, y_test, untransformed_predictions, plot_folder)
                predictions.append(model.predictions)
                mse.append(model.mse)

                save_results(plot_folder, model.predictions, model.true, model.mse)

        # Write training times to files
        training_file = os.path.join(base_plot_folder, "training_time.txt")
        with open(training_file, "w+") as f:
            for time in training_time:
                f.write(time + '\n')
        return predictions, mse

    elif model_type == 'FF_2L':
        for epochs in num_epochs:
            for neurons in num_neurons_L1:
                for neurons2 in num_neurons_L2:
                    logging.info('Evaluating model {} for {} features, {} epochs, and {}x{} neurons'.format(model_type, num_features, epochs, neurons, neurons2))
        
                    # Create folder for plots
                    plot_folder = base_plot_folder
                    plot_folder = os.path.join(plot_folder, '{}_{}neurons_{}epochs'.format(neurons, neurons2, epochs))
                    if not os.path.isdir(plot_folder):
                        os.makedirs(plot_folder)

                    # Initialize model
                    model = nnk.my_keras(epochs, neurons, num_features, activation_function, neurons2)
        
                    # Evaluate model, validating on same test set key as trained on
                    model.evaluate_model(X_train_final, y_train_final, X_test_final[0], y_test_final[0], plot_folder, two_layer=True)

                    # Record training time for each model
                    training_time.append("{}epochs {}x{}neurons: {}".format(epochs, neurons, neurons2, model.training_time))

                    # Predict on each of the test sets and plot MSE:
                    # MSE plotting currently not working
                    model.evaluate_test_sets(X_test_final, y_test_final)
        
                    untransformed_predictions = []
                    for p in model.predictions:
                        untransformed_predictions.append(untransform(p, shape))
                    plotting.plot_velocities_and_spectra(X_test, y_test, untransformed_predictions, plot_folder)
                    predictions.append(model.predictions)
                    mse.append(model.mse)

                    save_results(plot_folder, model.predictions, model.true, model.mse)

        # Write training times to files
        training_file = os.path.join(base_plot_folder, "training_time.txt")
        with open(training_file, "w+") as f:
            for time in training_time:
                f.write(time + '\n')
        return predictions, mse

    elif model_type == 'Olga_ELM':
        for neurons in num_neurons_L1:
            logging.info('Evaluating model {} for {} features and {} neurons'.format(model_type, num_features, neurons))

            # Create folder for plots
            plot_folder = base_plot_folder
            plot_folder = os.path.join(plot_folder, '{}_neurons'.format(neurons))
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)
            
            # Initialize model
            model = olga_elm.Olga_ELM(neurons, num_features)

            # Train and test model
            model.extreme_learning_machine(X_train_final, y_train_final, X_test_final, y_test_final)

            # Record training time for each model
            training_time.append("{}neurons: {}".format(neurons, model.training_time))

            untransformed_predictions = []
            for p in model.predictions:
                untransformed_predictions.append(untransform(p, shape))
            plotting.plot_velocities_and_spectra(X_test, y_test, untransformed_predictions, plot_folder)
            predictions.append(model.predictions)
            mse.append(model.mse)

            save_results(plot_folder, model.predictions, model.true, model.mse)

        # Write training times to files
        training_file = os.path.join(base_plot_folder, "training_time.txt")
        with open(training_file, "w+") as f:
            for time in training_time:
                f.write(time + '\n')
        return predictions, mse


