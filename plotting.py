import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc
import glob
import logging
import utils
import os



# mpl.style.use(['dark_background','mystyle'])
# mpl.style.use(['mystyle'])

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


def imagesc(Arrays, titles, name=None):
    axis = [0, np.pi, 0, np.pi]

    cmap = plt.cm.jet  # define the colormap
    # cmap = plt.cm.binary
    norm = mpl.colors.Normalize(vmin=-0.7, vmax=0.7)

    if len(Arrays) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(10, 4))
        # fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(4, 2.5))

        k = 0
        for ax in axes.flat:
            im = ax.imshow(Arrays[k].T, origin='lower', cmap=cmap, norm=norm, interpolation="nearest", extent=axis)
            ax.set_title(titles[k])
            ax.set_adjustable('box-forced')
            ax.set_xlabel(r'$x$')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            k += 1
        axes[0].set_ylabel(r'$y$')
        cbar_ax = fig.add_axes([0.89, 0.18, 0.017, 0.68])  # ([0.85, 0.15, 0.05, 0.68])
        fig.subplots_adjust(left=0.07, right=0.87, wspace=0.1, bottom=0.05, top=0.98)
        fig.colorbar(im, cax=cbar_ax, ax=axes.ravel().tolist())
    else:
        fig = plt.figure(figsize=(6.5, 5))
        ax = plt.gca()
        im = ax.imshow(Arrays[0].T, origin='lower', cmap=cmap, interpolation="nearest")
        plt.colorbar(im, fraction=0.05, pad=0.04)
    if name:
        # pickle.dump(ax, open(self.folder + name, 'wb'))
        fig.savefig(name)
    del ax, im, fig, cmap
    gc.collect()
    plt.close()


def spectra(folder, fname, ind):

    ind = str(ind)
    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()
    if ind == '':
        files = ['fine_grid.spectra', 'coarse_grid.spectra', 'filtered.spectra']
        labels = ['fine grid', 'coarse grid', 'filtered']
    else:
        files = ['predicted' + ind + '.spectra', 'filtered' + ind + '.spectra', 'true' + ind + '.spectra']
        labels = ['predicted', 'filtered', 'true']

    for k in range(len(files)):
        f = open(os.path.join(folder, files[k]), 'r')
        data = np.array(f.readlines()).astype(np.float)
        x = np.arange(len(data))
        ax.loglog(x, data, '-', linewidth=2, label=labels[k])

    # y = 1e9 * np.power(x[1:], -5./3)
    # ax.loglog(x[1:], y, 'r--', label=r'$-5/3$ slope')
    ax.set_title('Spectra')
    ax.set_ylabel(r'$E$')
    ax.set_xlabel(r'k')
    # ax.axis(ymin=1e2)
    plt.legend(loc=0)

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(fname)
    plt.close('all')


def plot_vorticity_pdf(x_test, y_test, y_predict, plot_folder):
    """ Plot normalized vorticity pdf."""
    shape = y_predict[0]['u'].shape

    if len(shape) == 2:
        dx = np.divide([np.pi, np.pi], np.array(shape))

        logging.info('Plot vorticity pdf')
        for test_example in range(3):

            fig = plt.figure(figsize=(4, 3))
            ax = plt.gca()
            _, du_dy = np.gradient(y_test[test_example]['u'], dx[0], dx[1])
            dv_dx, _ = np.gradient(y_test[test_example]['v'], dx[0], dx[1])
            vorticity = dv_dx - du_dy
            vorticity /= np.max(np.abs(vorticity))
            x, y = utils.pdf_from_array_with_x(vorticity, bins=100, range=[-1, 1])
            ax.semilogy(x, y, label='true')

            _, du_dy = np.gradient(x_test[test_example]['u'], dx[0], dx[1])
            dv_dx, _ = np.gradient(x_test[test_example]['v'], dx[0], dx[1])
            vorticity = dv_dx - du_dy
            vorticity /= np.max(np.abs(vorticity))
            x, y = utils.pdf_from_array_with_x(vorticity, bins=100, range=[-1, 1])
            ax.semilogy(x, y, label='filtered')

            _, du_dy = np.gradient(y_predict[test_example]['u'], dx[0], dx[1])
            dv_dx, _ = np.gradient(y_predict[test_example]['v'], dx[0], dx[1])
            vorticity = dv_dx - du_dy
            vorticity /= np.max(np.abs(vorticity))
            x, y = utils.pdf_from_array_with_x(vorticity, bins=100, range=[-1, 1])
            ax.semilogy(x, y, label='predicted')

            ax.set_title('Vorticity pdf')
            ax.set_ylabel('pdf')
            ax.set_xlabel(r'$\omega$')
            ax.axis(ymin=1e-3)
            plt.legend(loc=0)

            fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
            fig.savefig(plot_folder + 'omega_{}'.format(test_example))
            plt.close('all')


def plot_velocities_and_spectra(x_test, y_test, y_predict, plot_folder):
    logging.info('Plot predicted velocities')
    for test_example in range(3):
        imagesc([y_test[test_example]['u'][0:32, 0:32],
                 x_test[test_example]['u'][0:32, 0:32],
                 y_predict[test_example]['u'][0:32, 0:32]],
                [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'], 
                os.path.join(plot_folder, 'u_{}'.format(test_example)))
        imagesc([y_test[test_example]['v'][0:32, 0:32],
                 x_test[test_example]['v'][0:32, 0:32],
                 y_predict[test_example]['v'][0:32, 0:32]],
                [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'], 
                os.path.join(plot_folder, 'v_{}'.format(test_example)))
        imagesc([y_test[test_example]['w'][0:32, 0:32],
                 x_test[test_example]['w'][0:32, 0:32],
                 y_predict[test_example]['w'][0:32, 0:32]],
                [r'$u_{true}$', r'$u_{filtered}$', r'$u_{predicted}$'], 
                os.path.join(plot_folder, 'w_{}'.format(test_example)))

        logging.info('Calculate ang plot spectra')
        utils.spectral_density(y_test[test_example],
                               os.path.join(plot_folder, 'true{}'.format(test_example)))
        utils.spectral_density(x_test[test_example],
                               os.path.join(plot_folder, 'filtered{}'.format(test_example)))
        utils.spectral_density(y_predict[test_example],
                                os.path.join(plot_folder, 'predicted{}'.format(test_example)))

        spectra(plot_folder, os.path.join(plot_folder, 'spectra{}'.format(test_example)), test_example)


