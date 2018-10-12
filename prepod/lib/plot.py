import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from prepod.lib.constants import PLOT_STYLING


def plot_set_styles(ax):
    """Applies plot styling to ax

    Params
    ------
        ax : ax
            axes to apply styles to

    Returns
    -------
        styles : dict
            `plt.plot` styles (colors, lw, alpha level etc.)
    """
    # Ax styling
    ax.set_facecolor(PLOT_STYLING['bc'])
    ax.grid(color=PLOT_STYLING['gc'])

    # Plot styling
    styles = PLOT_STYLING['plot_styles']

    return styles


def plot_set_props(ax, props):
    """Adds properties to ax

    Params
    ------
        ax : ax
            axes to apply changes to
        props : dict
            values to update properties with

    Returns
    -------
        ax : ax
            axes with applied changes
    """
    props = PLOT_STYLING['props'].copy().update(props)
    ax.set_xlim(props['xlim'])
    ax.set_xlim(props['ylim'])
    ax.set_xticks(props['xticks'])
    ax.set_yticks(props['yticks'])
    ax.set_xticklabels(props['xticklabels'])
    ax.set_yticklabels(props['yticklabels'])
    ax.set_xlabel(props['xlabel'])
    ax.set_ylabel(props['ylabel'])
    ax.legend(props['legend'])
    return ax


def plot_raw_vs_filt(raw, filt, n_sec, t0=60, show=True):
    """Plots raw signal against filtered signal

    Params
    ------
        raw : Data
            raw signal
        filt : Data
            filtered signal
        n_sec : int
            amount of seconds to plot data for
        t0 : int
            plot signal starting at t0 (in seconds)
        show : boolean
            whether to call `plt.show`

    Returns
    -------
        None
    """
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

