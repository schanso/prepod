import os
from datetime import datetime as dt

import mne
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

import prepod.lib.constants as const
import prepod.lib.helpers as hlp
import prepod.lib.io as io
import prepod.lib.prep as prep


# PARAMS
study = 'Sudocu'
win_length = 60
regions = ['full']
freq_bands = ['alpha']
locs = const.SENSOR_LOCS
path_out = '/Users/jannes/Projects/delir/results/plots/psd/{}_{}_{}.png'.format(dt.now().strftime('%Y%m%d%H%M%S'),
                                                                                regions[0],
                                                                                freq_bands[0])
n_components = 4
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# PATHS
path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
path_log = path_data + 'info/log.csv'
dir_raw = path_data + 'rec/sudocu/brainvision/raw'
dir_filtered = path_data + 'rec/sudocu/brainvision/filtered/'
if not os.path.exists(dir_filtered):
    os.makedirs(dir_filtered)
dir_bis = path_data + 'rec/sudocu/bis/'


# INFO
fnames_raw = hlp.return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in const.EXCLUDE_SUBJ]


# MAIN
for region in regions:
    dir_out_raw = '{}/{}/{}'.format(dir_raw, 'npy', region)
    dir_out_filtered = dir_filtered + region
    if not os.path.exists(dir_out_filtered):
        os.makedirs(dir_out_filtered)

    for freq_band in freq_bands:
        lcut, hcut = const.FREQ_BANDS[freq_band]
        dir_signal = '{}/{}'.format(dir_out_filtered, freq_band)
        if not os.path.exists(dir_signal):
            os.makedirs(dir_signal)
        fname_merged = 'complete_{}s.npy'.format(win_length)
        path_out_merged = '{}/{}'.format(dir_signal, fname_merged)

        try:
            _data = io.load_pickled(path_in=path_out_merged)
        except Exception:
            datasets = []
            for subj_id in subj_ids:
                curr_fname = hlp.return_fnames(dir_in=dir_signal, substr=subj_id)
                path_signal = '{}/{}'.format(dir_signal, hlp.return_fnames(dir_in=dir_signal, substr=subj_id))
                path_bis = dir_bis + subj_id + '/'
                data = io.load_wyrm(path_in=path_signal)
                data.markers = prep.create_markers(data, win_length)
                data.label = prep.fetch_labels(path_labels, study, subj_id)
                data = prep.match_bis(data, path_bis)
                data = prep.segment_data(data, win_length)
                data.subj_id = prep.append_subj_id(data, subj_id)
                datasets.append(data)
            _data = prep.merge_subjects(datasets)

        data = prep.subset_data(_data, bis_crit=None, drop_perc=None, drop_from=None, use_min=None, use_from=None)
        w, a = prep.apply_csp(data, return_as='patterns')
        pos = np.array([[el['x'], el['y']] for el in locs for ch in _data.axes[2] if el['ch_name'] == ch])
        fig, axes = plt.subplots(figsize=(10, 4), ncols=n_components, nrows=2)
        for i in range(0,int(n_components/2)):
            mne.viz.plot_topomap(w[:,i], pos, axes=axes[0, i], show=False, outlines='skirt', contours=0)
            mne.viz.plot_topomap(a[:,i], pos, axes=axes[1, i], show=False, outlines='skirt', contours=0)
        for i in range(-int(n_components/2),0):
            mne.viz.plot_topomap(w[:,i], pos, axes=axes[0, i], show=False, outlines='skirt', contours=0)
            mne.viz.plot_topomap(a[:,i], pos, axes=axes[1, i], show=False, outlines='skirt', contours=0)

        cols = ['CSP{}'.format(str(el)) for el in range(1, n_components+1)]
        rows = ['Filter', 'Pattern']
        for ax, col in zip(axes[0], cols):
            ax.set_title(col, fontsize=12, pad=30)
        for ax, row in zip(axes[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=12, labelpad=60)
        fig.tight_layout()
        plt.savefig(path_out, dpi=300)
