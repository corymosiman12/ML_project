import numpy as np
import os
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



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
    ax.plot(neurons[1:], mse[1, 1:], '-', linewidth=2, label='27 features')
    ax.plot(neurons[np.argmin(mse[1])], np.min(mse[1]), 'ro', linewidth=2, label='minimum')
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
        filter_type ='sharp'
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    ax.plot(neurons, mse_spectra[0], '-', linewidth=2, label='9 features')
    ax.plot(neurons[np.argmin(mse_spectra[0])], np.min(mse_spectra[0]), 'ro', linewidth=2)
    ax.plot(neurons[1:], mse_spectra[1, 1:], '-', linewidth=2, label='27 features')
    ax.plot(neurons[np.argmin(mse_spectra[1])], np.min(mse_spectra[1]), 'ro', linewidth=2, label='minimum')
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



model_type = 'Olga_ELM'
dimension = 2
filter_type = ['gaussian', 'physical_sharp']
num_features = [9, 27]
neurons = np.arange(20, 150, 20)

assert model_type == 'FF_1L' \
    or model_type == 'FF_2L' \
    or model_type == 'Olga_ELM', 'Incorrect model_type: %r' % model_type

mse = np.empty((8, 19))
mse_spectra = np.empty_like(mse)
for i in range(len(filter_type)):
    assert filter_type[i] == "gaussian" \
           or filter_type[i] == "median" \
           or filter_type[i] == "noise" \
           or filter_type[i] == "fourier_sharp" \
           or filter_type[i] == "physical_sharp", \
           'Incorrect filter type: %r' % filter_type
    for j in range(len(num_features)):

        plot_folder = os.path.join(plot_folder_base, "{}".format(model_type),
                                   "{}dim_{}_{}feat".format(str(dimension), filter_type[i],str(num_features[j])))
        print(plot_folder)
        assert os.path.isdir(plot_folder), 'Incorrect case'
        mse_tmp = []
        mse_spectra_tmp = []
        for n in neurons:
            folder = os.path.join(plot_folder, '{}_neurons'.format(n))
            mse_tmp.append(np.loadtxt(os.path.join(folder, 'mse.txt'))[0])
            mse_spectra_tmp.append(calc_spectra_mse(np.loadtxt(os.path.join(folder, 'true0.spectra')),
                                               np.loadtxt(os.path.join(folder, 'predicted0.spectra'))))
        mse[i*2+j] = np.array(mse_tmp).copy()
        mse_spectra[i*2+j] = np.array(mse_spectra_tmp).copy()

mse_spectra /= 1e3
for i in range(len(filter_type)):
    ind = i*2
    plot_mse(neurons, mse[ind:ind+2], filter_type=filter_type[i], dim=dimension)
    plot_mse_spectra(neurons, mse_spectra[ind:ind+2], filter_type=filter_type[i], dim=dimension)


fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, figsize=(6.5, 10))
print(ax.shape)
for i in range(len(filter_type)):
    ax[i, 0].plot(neurons, mse_spectra[0], '-', linewidth=2, label='9 features')
    ax[i, 0].plot(neurons[np.argmin(mse_spectra[0])], np.min(mse_spectra[0]), 'ro', linewidth=2)
    ax[i, 0].plot(neurons[1:], mse_spectra[1, 1:], '-', linewidth=2, label='27 features')
    ax[i, 0].plot(neurons[np.argmin(mse_spectra[1, 1:])], np.min(mse_spectra[1, 1:]), 'ro', linewidth=2, label='minimum')
    ax[i, 1].plot(neurons, mse_spectra[0], '-', linewidth=2, label='9 features')
    ax[i, 1].plot(neurons[np.argmin(mse_spectra[0])], np.min(mse_spectra[0]), 'ro', linewidth=2)
    ax[i, 1].plot(neurons[1:], mse_spectra[1, 1:], '-', linewidth=2, label='27 features')
    ax[i, 1].plot(neurons[np.argmin(mse_spectra[1, 1:])], np.min(mse_spectra[1, 1:]), 'ro', linewidth=2, label='minimum')

    ax[i, 0].set_title('ELM: {} filter 2D'.format(filter_type))
    ax[i, 1].set_title('ELM: {} filter 2D'.format(filter_type))
    ax[i, 0].set_ylabel('MSE')
    ax[i, 1].set_ylabel('spectra MSE')



ax[len(filter_type)-1, 0].set_xlabel('Number of neurons')
ax[len(filter_type)-1, 1].set_xlabel('Number of neurons')
ax[len(filter_type)-1, 0].axis(xmin=20, xmax=200)
ax[len(filter_type)-1, 0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax[len(filter_type)-1, 1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax[len(filter_type)-1, 0].xaxis.set_major_locator(ticker.MultipleLocator(50))
ax[len(filter_type)-1, 0].xaxis.set_minor_locator(ticker.MultipleLocator(10))
ax[len(filter_type)-1, 1].xaxis.set_major_locator(ticker.MultipleLocator(50))
ax[len(filter_type)-1, 1].xaxis.set_minor_locator(ticker.MultipleLocator(10))
plt.legend(loc=0)

fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
fig.savefig(os.path.join(plot_folder_base, "{}".format(model_type), '2dim_all_mse'))
plt.close('all')






