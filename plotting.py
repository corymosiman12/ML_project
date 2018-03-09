import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc
import glob



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
    norm = mpl.colors.Normalize(vmin=-0.7, vmax=0.7)

    if len(Arrays) > 1:
        fig, axes = plt.subplots(nrows=1, ncols=len(Arrays), sharey=True, figsize=(10, 4))
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


def spectra(folder, fname):

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    files = glob.glob(folder + '*.spectra')
    labels = ['filtered', 'fine', 'coarse']

    for k in range(len(files)):
        f = open(files[k], 'r')
        data = np.array(f.readlines()).astype(np.float)
        x = np.arange(len(data))
        ax.loglog(x, data, '-', linewidth=2, label=labels[k])

    # y = 7.2e14 * np.power(x[1:], -5 / 3)
    # ax.loglog(x[1:], y, 'r--', label=r'$-5/3$ slope')
    ax.set_title('Spectra')
    ax.set_ylabel(r'$E$')
    ax.set_xlabel(r'k')
    ax.axis(ymin=1e5)
    plt.legend(loc=0)

    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.87)
    fig.savefig(fname)

