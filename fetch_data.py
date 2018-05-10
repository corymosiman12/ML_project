import numpy as np
import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotting
import utils
import data
from sklearn.metrics import mean_squared_error



# mpl.rcParams['figure.figsize'] = 6.5, 2.2
# plt.rcParams['figure.autolayout'] = True

mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rc('text', usetex=True)
mpl.rcParams['axes.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['axes.titlesize'] = 1.5 * plt.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = plt.rcParams['font.size']
mpl.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
mpl.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
# plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.width'] = 1
# mpl.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
plt.rcParams['axes.linewidth'] = 1


plot_folder_base = './plots/'
plot_folder = plot_folder_base

model_type = 'FF_1L'
dimension = 3
filter_type = ['physical_sharp'] #['gaussian', 'noise', 'physical_sharp'] #, 'median']
num_features = [27]
neurons = np.arange(20, 160, 20)

activation = 'tanh'
num_epochs = 50

def calc_spectra_mse(spectra_true, spectra_predicted):
    return np.linalg.norm(spectra_true - spectra_predicted)/len(spectra_true)

def plot_mse(neurons, mse, filter_type, dim):

    print(dim, filter_type)

    if filter_type == 'physical_sharp':
        filter_type ='sharp'
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    ax.plot(neurons, mse[0], '-', linewidth=2, label='9 features')
    ax.plot(neurons[np.argmin(mse[0])], np.min(mse[0]), 'ro', linewidth=2)
    if dim == 2:
        ax.plot(neurons[1:], mse[1, 1:], '-', linewidth=2, label='27 features')
        ax.plot(neurons[np.argmin(mse[1, 1:])+1], np.min(mse[1, 1:]), 'ro', linewidth=2, label='minimum')
    ax.set_title('ELM: {} filter {}D'.format(filter_type, dim))
    ax.set_ylabel('MSE')
    ax.set_xlabel('Number of neurons')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.axis(xmin=20, xmax=200)
    plt.legend(loc=0)
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(os.path.join(plot_folder_base, "{}".format(model_type), '{}dim_{}_mse'.format(dim, filter_type)))


def plot_mse_spectra(neurons, mse_spectra, filter_type, dim):

    if filter_type == 'physical_sharp':
        filter_type = 'sharp'
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    ax.plot(neurons, mse_spectra[0], '-', linewidth=2, label='9 features')
    ax.plot(neurons[np.argmin(mse_spectra[0])], np.min(mse_spectra[0]), 'ro', linewidth=2)
    if dim == 2:
        ax.plot(neurons[1:], mse_spectra[1, 1:], '-', linewidth=2, label='27 features')
        ax.plot(neurons[np.argmin(mse_spectra[1, 1:]) + 1], np.min(mse_spectra[1, 1:]), 'ro', linewidth=2, label='minimum')
    ax.set_title('ELM: {} filter {}D'.format(filter_type, dim))
    ax.set_ylabel('MSE')
    ax.set_xlabel('Number of neurons')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.axis(xmin=20, xmax=200)
    plt.legend(loc=0)

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(os.path.join(plot_folder_base, "{}".format(model_type), '{}dim_{}_mse_spectra'.format(dim, filter_type)))
    plt.close('all')


def plot_mse_by_filter(neurons, mse, mse_spectra, filter_type, dim):

    if filter_type == 'physical_sharp':
        filter_type = 'sharp'
    fig = plt.figure(figsize=(4.5, 3.5))
    ax = plt.gca()
    if dim == 2:
        ax.plot(neurons, mse[0], 'gs-', ls='dashed', lw=2, label='{} features'.format(num_features[0]))
    else:
        ax.plot(neurons, mse[0], 'go-', lw=2, label='{} features'.format(num_features[0]))

    ax.plot(neurons[np.argmin(mse[0])], np.min(mse[0]), 'ro', linewidth=2)
    print(filter_type, ' MSE ', num_features[0], neurons[np.argmin(mse[0])], np.min(mse[0]))
    if dim == 2:
        ax.plot(neurons[1:], mse[1, 1:], 'go-',  lw=2, label='27 features')
        ax.plot(neurons[np.argmin(mse[1, 1:]) + 1], np.min(mse[1, 1:]), 'ro', linewidth=2)
        print(filter_type, ' MSE ', num_features[1], neurons[np.argmin(mse[1, 1:]) + 1], np.min(mse[1, 1:]))
    ax.set_ylabel('MSE', color='g')
    ax.set_xlabel('Number of Neurons')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.ticklabel_format(useOffset=False)
    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.axis(xmin=20, xmax=200)

    ax2 = ax.twinx()
    if dim == 2:
        ax2.plot(neurons, mse_spectra[0], 'bs-', ls='dashed', lw=2, label='{} features'.format(num_features[0]))
    else:
        ax2.plot(neurons, mse_spectra[0], 'bo-', lw=2, label='{} features'.format(num_features[0]))
    ax2.plot(neurons[np.argmin(mse_spectra[0])], np.min(mse_spectra[0]), 'ro', linewidth=2)
    print(filter_type, ' MSE spectra', num_features[0], neurons[np.argmin(mse_spectra[0])], np.min(mse_spectra[0]))
    if dim == 2:
        ax2.plot(neurons[1:], mse_spectra[1, 1:], 'bo-', lw=2, label='27 features')
        ax2.plot(neurons[np.argmin(mse_spectra[1, 1:]) + 1], np.min(mse_spectra[1, 1:]), 'ro', linewidth=2)
        print(filter_type, ' MSE spectra', num_features[1], neurons[np.argmin(mse_spectra[1, 1:]) + 1], np.min(mse_spectra[1, 1:]))
    # ax2.set_title('{} filter'.format(filter_type, dim))
    ax2.set_ylabel('spectra MSE', color='b')
    ax2.set_xlabel('Number of neurons')
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    # ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    ax2.axis(xmin=20, xmax=200)
    plt.legend(loc=0)

    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.12, top=0.92)
    folder = os.path.join(plot_folder_base, "{}".format(model_type))
    fig.savefig(os.path.join(folder, '{}dim_{}_{}_mse'.format(dim, model_type, filter_type)))
    plt.close('all')


def plot_mse_subplots(mse, mse_spectra, dim):

    fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(6.5, 6.5))
    for i in range(len(filter_type)):
        ind = i * 2
        if filter_type[i] == 'physical_sharp':
            filter_type[i] = 'sharp'
        ax[i, 0].plot(neurons, mse[ind], '-', linewidth=2, label='9 features')
        ax[i, 0].plot(neurons[np.argmin(mse[ind])], np.min(mse[ind]), 'ro', linewidth=2)
        ax[i, 1].plot(neurons, mse_spectra[ind], '-', linewidth=2, label='9 features')
        ax[i, 1].plot(neurons[np.argmin(mse_spectra[ind])], np.min(mse_spectra[ind]), 'ro', linewidth=2)

        ax[i, 0].plot(neurons[1:], mse[ind+1, 1:], '-', linewidth=2, label='27 features')
        ax[i, 0].plot(neurons[np.argmin(mse[ind+1, 1:])+1], np.min(mse[ind+1, 1:]), 'ro', linewidth=2)
        ax[i, 1].plot(neurons[1:], mse_spectra[ind+1, 1:], '-', linewidth=2, label='27 features')
        ax[i, 1].plot(neurons[np.argmin(mse_spectra[ind+1, 1:])+1], np.min(mse_spectra[ind+1, 1:]), 'ro', linewidth=2)

        ax[i, 0].set_title('{} filter'.format(filter_type[i]))
        ax[i, 1].set_title('{} filter'.format(filter_type[i]))
        ax[i, 0].set_ylabel('MSE')
        ax[i, 1].set_ylabel('spectra MSE')
        # ax[i, 0].set_ylabel(filter_type[i])

    # ax[0, 0].set_title('MSE')
    # ax[0, 1].set_title('spectra MSE')
    #
    ax[len(filter_type)-1, 0].set_xlabel('Number of neurons')
    ax[len(filter_type)-1, 1].set_xlabel('Number of neurons')
    ax[len(filter_type)-1, 0].axis(xmin=20, xmax=200)
    # ax[len(filter_type)-1, 0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    # ax[len(filter_type)-1, 1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax[len(filter_type)-1, 0].xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax[len(filter_type)-1, 0].xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax[len(filter_type)-1, 1].xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax[len(filter_type)-1, 1].xaxis.set_minor_locator(ticker.MultipleLocator(10))
    plt.legend(loc=0)
    #
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.35, hspace=0.25)
    fig.savefig(os.path.join(plot_folder_base, "{}".format(model_type)) + "/{}dim_all_mse".format(dim))
    plt.close('all')


########################################################################################################################
################# START HERE ##################################
########################################################################################################################



assert model_type == 'FF_1L' \
    or model_type == 'FF_2L' \
    or model_type == 'Olga_ELM', 'Incorrect model_type: %r' % model_type

# Fetch data from different files
mse = np.zeros((4*len(num_features), len(neurons)))
mse_spectra = np.zeros_like(mse)
folder_base = os.path.join(plot_folder_base, "{}".format(model_type))
for i in range(len(filter_type)):
    assert filter_type[i] == "gaussian" or filter_type[i] == "median" or filter_type[i] == "noise" \
           or filter_type[i] == "fourier_sharp" or filter_type[i] == "physical_sharp", \
           'Incorrect filter type: %r' % filter_type

    for j in range(len(num_features)):
        if model_type == 'Olga_ELM':
            plot_folder = os.path.join(folder_base, "{}dim_{}_{}feat".format(dimension, filter_type[i],
                                                                             num_features[j]))
        else:
            plot_folder = os.path.join(folder_base, "{}dim_{}_{}feat_{}".format(dimension, filter_type[i],
                                                                                num_features[j], activation))
        print(plot_folder)
        assert os.path.isdir(plot_folder), 'Incorrect case'
        mse_tmp, mse_spectra_tmp = [], []
        for n in neurons:
            if model_type == 'Olga_ELM':
                folder = os.path.join(plot_folder, '{}_neurons'.format(n))
            else:
                folder = os.path.join(plot_folder, '{}neurons_{}epochs'.format(n, num_epochs))
            mse_tmp.append(np.loadtxt(os.path.join(folder, 'mse.txt'))[0])
            mse_spectra_tmp.append(mean_squared_error(np.loadtxt(os.path.join(folder, 'true0.spectra')),
                                                      np.loadtxt(os.path.join(folder, 'predicted0.spectra'))))
        mse[i*len(num_features)+j] = np.array(mse_tmp).copy()
        mse_spectra[i*len(num_features)+j] = np.array(mse_spectra_tmp).copy()

# # Normalize by gaussian error
mse /= np.max(mse[:len(num_features), 1:])
mse_spectra /= np.max(mse_spectra[:len(num_features), 1:])

# plot mse in separate files
# for i in range(len(filter_type)):
#     ind = i*len(num_features)
#     plot_mse(neurons, mse[ind:ind+len(num_features)], filter_type=filter_type[i], dim=dimension)
#     plot_mse_spectra(neurons, mse_spectra[ind:ind+len(num_features)], filter_type=filter_type[i], dim=dimension)

# plot mse by filter type
for i in range(len(filter_type)):
    if dimension == 2:
        start = 0
    else:
        start = 1
    ind = i * len(num_features)
    plot_mse_by_filter(neurons[start:], mse[ind:ind+len(num_features), start:],
                       mse_spectra[ind:ind+len(num_features), start:],
                       filter_type=filter_type[i], dim=dimension)





#######################################################################################################################
##### Plot vorticity and tau
#######################################################################################################################
# model_type = 'Olga_ELM'
# dimension = 3
# folder_base = os.path.join(plot_folder_base, "{}".format(model_type))
# n_best_mse_27 = np.array([100, 30, 30, 60])
# n_best_mse_sp_27 = np.array([30, 200, 160, 150])
# velocity = data.load_data(dimension)
# x_train, y_train, x_test, y_test = data.form_train_test_sets(velocity, Npoints_coarse=64)
# y_mse_dict = [dict()]*3
# y_mse_sp_dict = [dict()]*3

# folder = os.path.join(plot_folder_base, model_type, 'tau_mse')
# if not os.path.isdir(folder): os.makedirs(folder)

# for i in range(len(n_best_mse_9)):
#     folder_mse = os.path.join(folder_base, "{}dim_{}_{}feat".format(dimension, filter_type[i], 27),
#                               '{}_neurons'.format(n_best_mse_9[i]))
#     print(folder_mse)
#     folder_mse_sp = os.path.join(folder_base, "{}dim_{}_{}feat".format(dimension, filter_type[i], 27),
#                                  '{}_neurons'.format(n_best_mse_sp_9[i]))
#     print(folder_mse_sp)
#     y_mse = np.loadtxt(os.path.join(folder_mse, 'y_predictions.txt'))
#     y_mse_sp = np.loadtxt(os.path.join(folder_mse_sp, 'y_predictions.txt'))
#     for j in range(3):
#         y_mse_dict[j] = utils.untransform_y_3D(y_mse[j], (64, 64, 64))
#         y_mse_sp_dict[j] = utils.untransform_y_3D(y_mse_sp[j], (64, 64, 64))
#     folder_filter = os.path.join(folder, filter_type[i])
#     if not os.path.isdir(folder_filter): os.makedirs(folder_filter)
#     plotting.plot_tau(x_test, y_test, y_mse_dict, folder_filter, filter_type[i], y_mse_sp_dict)



# folder = os.path.join(plot_folder_base, model_type, 'omega')
# if not os.path.isdir(folder): os.makedirs(folder)
#
# for i in range(len(n_best_mse_9)):
#     folder_mse = os.path.join(folder_base, "{}dim_{}_{}feat".format(dimension, filter_type[i], 27),
#                               '{}_neurons'.format(n_best_mse_9[i]))
#     print(folder_mse)
#     folder_mse_sp = os.path.join(folder_base, "{}dim_{}_{}feat".format(dimension, filter_type[i], 27),
#                                  '{}_neurons'.format(n_best_mse_sp_9[i]))
#     print(folder_mse_sp)
#     y_mse = np.loadtxt(os.path.join(folder_mse, 'y_predictions.txt'))
#     y_mse_sp = np.loadtxt(os.path.join(folder_mse_sp, 'y_predictions.txt'))
#     for j in range(3):
#         y_mse_dict[j] = utils.untransform_y_3D(y_mse[j], (64, 64, 64))
#         y_mse_sp_dict[j] = utils.untransform_y_3D(y_mse_sp[j], (64, 64, 64))
#     folder_filter = os.path.join(folder, filter_type[i])
#     if not os.path.isdir(folder_filter): os.makedirs(folder_filter)
#     plotting.plot_vorticity_pdf(x_test, y_test, y_mse_dict, folder_filter, y_mse_sp_dict)

