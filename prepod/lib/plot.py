import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from prepod.lib.constants import PLOT_STYLING


def plot_set_styles(ax):
    """"""
    # Ax styling
    ax.set_facecolor(PLOT_STYLING['bc'])
    ax.grid(color=PLOT_STYLING['gc'])

    # Plot styling
    styles = PLOT_STYLING['plot_styles']

    return styles


def plot_set_props(ax, props):
    """"""
    ax.set_xlim(props['xlim'])
    ax.set_xticks(props['xticks'])
    ax.set_xticklabels(props['xticklabels'])
    ax.set_xlabel(props['xlabel'])
    ax.set_ylabel(props['ylabel'])
    ax.legend(props['legend'])
    return ax


def plot_raw_vs_filt(raw, filt, n_sec, t0=60, show=True):
    """"""
    start = np.floor(raw.fs * t0).astype('int')
    n_samples = np.floor(raw.fs * n_sec).astype('int')
    x = raw.axes[0][:n_samples]
    y_raw = raw.data[start:start+n_samples, 0]
    y_filt = filt.data[start:start+n_samples, 0]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    styles = plot_set_styles(ax)
    ax.plot(x, y_raw, c=PLOT_STYLING['c'][0], **styles)
    ax.plot(x, y_filt, c=PLOT_STYLING['c'][1], **styles)
    props = {
        'xlim': [0, 0+n_sec],
        'xticks': np.arange(n_sec+1),
        'xticklabels': np.arange(t0, t0+n_sec+1),
        'xlabel': 's',
        'ylabel': 'µV',
        'legend': ['raw', 'filtered']
    }
    plot_set_props(ax, props)

    if show:
        plt.show()

