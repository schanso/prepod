import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

import prepod.lib.constants as const
import prepod.lib.helpers as hlp
import prepod.lib.io as io
import prepod.lib.plot as plot
import prepod.lib.prep as prep

# PATHS

path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
dir_raw = path_data + 'rec/sudocu/brainvision/raw'
dir_bis = path_data + 'rec/sudocu/bis/'
dir_plots = '/Users/jannes/Projects/delir/results/plots/'

# INFO

study = 'Sudocu'
fnames_raw = hlp.return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in const.EXCLUDE_SUBJ]

# INDIVIDUALS

for subj_id in subj_ids:
    path_bis = dir_bis + subj_id + '/'
    path_out = dir_plots + 'bis/' + subj_id + '_bis.png'
    label = prep.fetch_labels(path_labels, study, subj_id)
    bis = io.read_bis(path_bis)
    bis_start = bis['SystemTime'].iloc[0]
    bis_end = bis['SystemTime'].iloc[bis.shape[0]-1]
    bis['t_delta'] = bis['SystemTime'].diff().apply(lambda x: x.total_seconds())
    x = np.linspace(0, (sum(bis['t_delta'][1:])+1)/60, len(bis['t_delta']))
    y = bis['BIS']
    c = const.PLOT_STYLING['c'][0] if label == '1' else const.PLOT_STYLING['c'][1]
    print(c, label)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, y, c=c)
    plot.plot_set_styles(ax)
    props = {
        'xlim': [min(x), max(x)],
        'ylim': [0, 100],
        'yticks': [0, 20, 40, 60, 80, 100],
        'xlabel': 'Time [min]',
        'ylabel': 'BIS',
        'legend': ['label {}'.format(label)]
    }
    plot.plot_set_props(ax, props)
    plt.title('{} (Time: {:.1f} min, Avg: {:.1f})'.format(subj_id, (sum(bis['t_delta'][1:])+1)/60, np.mean(y)))
    plt.savefig(path_out, bbox_inches='tight')

# AVERAGES

label0 = []
label1 = []
subj_ids = [el for el in subj_ids if el != '2456']  # 2456 too short
for subj_id in subj_ids:
    path_bis = dir_bis + subj_id + '/'
    path_out = dir_plots + 'bis/mean_bis_first60.png'
    label = prep.fetch_labels(path_labels, study, subj_id)
    bis = io.read_bis(path_bis)
    y = list(bis['BIS'].iloc[:3601])
    if label == '1':
        label1.append(y)
    else:
        label0.append(y)
    print(subj_id, label, len(y))
x = np.linspace(0, 60, 3601)
mean0 = np.mean(label0, axis=0)
mean1 = np.mean(label1, axis=0)
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(x, mean0, c=const.PLOT_STYLING['c'][0])
ax.plot(x, mean1, c=const.PLOT_STYLING['c'][1])
plot.plot_set_styles(ax)
props = {
    'xlim': [0, 60],
    'ylim': [0, 100],
    'yticks': [0, 20, 40, 60, 80, 100],
    'xlabel': 'Time [min]',
    'ylabel': 'BIS',
    'legend': ['label 0', 'label 1']
}
plot.plot_set_props(ax, props)
plt.title('Mean BIS, first 60 minutes')
plt.savefig(path_out, bbox_inches='tight')