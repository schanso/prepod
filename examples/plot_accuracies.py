from datetime import datetime as dt

import matplotlib as mpl
mpl.use('TkAgg')
mpl.rc('font', family = 'serif', serif = 'cmr10', size=10)
import matplotlib.pyplot as plt
import pandas as pd

target_freq = 'slow'
target_region = 'fronto-parietal'
bis_crit = 40
win_length = 60
path_in = '/Users/jannes/Projects/delir/results/acc/20190623162000_res.csv'
path_out = '/Users/jannes/Projects/delir/results/plots/acc/{}_{}_{}_{}s_{}bis_{}.png'.format(dt.now().strftime('%Y%m%d%H%M%S'),
                                                                                             target_freq,
                                                                                             target_region,
                                                                                             win_length,
                                                                                             bis_crit,
                                                                                             'minute_blocks')
names = ['temp_idx','date','mean_lda','std_lda','mean_svm','std_svm','region','freq_band','test_size','n_leave_out','win_length','bis_crit','drop_perc','drop_from','solver','shrink','kernel','include_external','external_factors','n_runs','use_min','use_from','below_500_lda', 'below_500_svm']
df = pd.read_csv(path_in, header=None, names=names, sep=',', skiprows=0)

target_cols = ['mean_lda','std_lda','mean_svm','std_svm','region','freq_band','win_length','bis_crit','use_from','use_min']
use_froms = ['beginning', 'end']
use_mins = ['(60, 45)', '(45, 30)', '(30, 15)', '(15, 0)']

fig, axes = plt.subplots(figsize=(7, 7), ncols=1, nrows=2)
for idx, use_from in enumerate(use_froms):
    mask0 = df['freq_band'] == target_freq
    mask1 = df['use_from'] == use_from
    mask2 = df['use_min'].isin(use_mins)
    mask3 = df['bis_crit'] == bis_crit
    mask4 = df['win_length'] == win_length
    mask5 = df['region'] == target_region

    masked = df[mask0 & mask1 & mask2 & mask3 & mask4 & mask5]
    masked = masked[target_cols]
    data = masked.groupby('use_min').max()
    data = data.reset_index()
    if idx == 1:
        data = data.iloc[::-1]
    errors = data[['std_lda', 'std_svm']]

    data.plot(x='use_min', y=['mean_lda', 'mean_svm'], kind='bar', ax=axes[idx], color=['#fe2151', '#999999'], edgecolor = '#000000', label=['LDA', 'SVM'])
    axes[idx].set_ylim([0, 1])
    axes[idx].legend(loc = 3)
    axes[idx].set_ylabel('Accuracy', fontsize=12, labelpad=10)
    axes[idx].axhline(y=0.5, linewidth=1, color='orange', linestyle='--')
    axes[idx].text(-.65, 0.5, 'chance\nlevel', color='orange', fontsize=8, va='center', ha='center')
    if idx == 0:
        axes[idx].set_xlabel('Minute blocks from start of surgery', fontsize=12)
        axes[idx].set_xticklabels(['[0, 15)', '[15, 30)', '[30, 45)', '[45, 60)'], rotation='horizontal')
    else:
        axes[idx].set_xlabel('Minute blocks before end of surgery', fontsize=12)
        axes[idx].set_xticklabels(['[60, 45)', '[45, 30)', '[30, 15)', '[15, 0)'], rotation='horizontal')

    for i, v in enumerate(data['mean_lda']):
        axes[idx].text(i-.24, v + .025, '{:.3f}'.format(v), color='#000000')
    for i, v in enumerate(data['mean_svm']):
        axes[idx].text(i+.025, v + .025, '{:.3f}'.format(v), color='#000000')

fig.subplots_adjust(hspace=0.5)
plt.savefig(path_out, dpi=300)
