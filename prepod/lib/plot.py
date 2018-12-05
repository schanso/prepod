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
    init_props = PLOT_STYLING['props']
    init_props.update(props)
    props = init_props.copy()
    if props['xlim'] is not None:
        ax.set_xlim(props['xlim'])
    if props['ylim'] is not None:
        ax.set_ylim(props['ylim'])
    if props['xticks'] is not None:
        ax.set_xticks(props['xticks'])
    if props['yticks'] is not None:
        ax.set_yticks(props['yticks'])
    if props['xticklabels'] is not None:
        ax.set_xticklabels(props['xticklabels'])
    if props['yticklabels'] is not None:
        ax.set_yticklabels(props['yticklabels'])
    if props['xlabel'] is not None:
        ax.set_xlabel(props['xlabel'])
    if props['ylabel'] is not None:
        ax.set_ylabel(props['ylabel'])
    if props['legend'] is not None:
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
        'xlim': [min(x), max(x)],
        'xticks': np.linspace(min(x), max(x), n_sec+1),
        'xticklabels': np.arange(t0, t0+n_sec+1),
        'xlabel': 's',
        'ylabel': 'µV',
        'legend': ['raw', 'filtered']
    }
    plot_set_props(ax, props)

    if show:
        plt.show()


def plot_accuracies(x, y, lcut, hcut, path_out=None, bis_crit=None, drop_perc=None, drop_from=None, show=True):
    """"""
    n_runs = len(x)
    mean, std = np.mean(y), np.std(y)
    fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=1)
    styles = plot_set_styles(ax)
    ax.bar(x, y, color=PLOT_STYLING['c'][-1], **styles)
    # ax.plot((x[0], x[-1]), (mean,mean), c=PLOT_STYLING['c'][1])
    # props = {
    #     # 'xticks': np.arange(n_sec + 1),
    #     # 'xticklabels': np.arange(t0, t0 + n_sec + 1),
    #     'xlabel': 'left out',
    #     'ylabel': 'acc'
    # }
    # plot_set_props(ax, props)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Left out', fontsize=15)
    ax.set_xticklabels(x, rotation=270)
    ax.set_ylabel('Acc.', fontsize=15)
    plt.suptitle('Prediction accuracies from leave-two-subjects-out CV', fontsize=20)
    plt.title(('Bandpass-filtered at {}/{} Hz, '
              + 'mean over all {} combinations: {:.3f} ({:.3f} std).\n'
              + 'BIS-cutoff: {}, dropped: {} from {}').format(
        lcut, hcut, n_runs, mean, std, bis_crit, drop_perc, drop_from
    ), fontsize=15)

    if path_out:
        plt.savefig(path_out, bbox_inches='tight')

    if show:
        plt.show()

