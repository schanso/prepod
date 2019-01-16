import os

import numpy as np
import pandas as pd

import prepod.lib.constants as const
import prepod.lib.helpers as hlp
import prepod.lib.io as io
import prepod.lib.models as models
import prepod.lib.plot as plot
import prepod.lib.prep as prep


# PARAMS

study = 'Sudocu'
region = 'frontal'
freq_band = 'alpha'
lcut, hcut = const.FREQ_BANDS[freq_band]

test_size = .5
n_leave_out = 2
win_length = 60
bis_crit = 50
drop_perc = .5
drop_from = 'beginning'

solver = 'svd'
shrink = False
kernel = 'linear'


# PATHS

path_data = '/Users/jannes/Projects/delir/data/'
path_labels = path_data + 'info/sudocu_info/subject_data.csv'
path_log = path_data + 'info/log.csv'
dir_raw = path_data + 'rec/sudocu/brainvision/raw'
dir_filtered = path_data + 'rec/sudocu/brainvision/filtered/'
if not os.path.exists(dir_filtered):
    os.makedirs(dir_filtered)
dir_bis = path_data + 'rec/sudocu/bis/'
dir_out_raw = '{}/{}/{}'.format(dir_raw, 'npy', region)
dir_out_filtered = dir_filtered + region
if not os.path.exists(dir_out_filtered):
    os.makedirs(dir_out_filtered)
dir_signal = '{}/{}'.format(dir_out_filtered, freq_band)
if not os.path.exists(dir_signal):
    os.makedirs(dir_signal)
fname_merged = 'complete_{}s.npy'.format(win_length)
path_out_merged = '{}/{}'.format(dir_signal, fname_merged)

# INFO

fnames_raw = hlp.return_fnames(dir_in=dir_raw)
subj_ids = sorted(list(set([el.split('_')[0] for el in fnames_raw])))
subj_ids = [el for el in subj_ids if el not in const.EXCLUDE_SUBJ]


# CLASSIFICATION (VANILLA LDA + SVM)

data = io.load_pickled(path_in=path_out_merged)
data = prep.subset_data(data, bis_crit=bis_crit, drop_perc=drop_perc, drop_from=drop_from)

x, acc_lda_all_runs, acc_svm_all_runs = [], [], []
print('Start training on subjects: {}'.format(', '.join(subj_ids)))
for i in range(len(subj_ids)):
    try:
        data_train, data_test, left_out = models.train_test_cv(data, n_leave_out=n_leave_out, idx=i)
        data_train = prep.apply_csp(data_train, return_as='logvar')
        data_test = prep.apply_csp(data_test, return_as='logvar')
        acc_lda = models.lda(data_train=data_train, data_test=data_test, solver=solver, shrinkage=shrink)
        acc_svm = models.svm(data_train, data_test, kernel=kernel)
        acc_lda_all_runs.append(acc_lda)
        acc_svm_all_runs.append(acc_svm)
        x.append(', '.join(left_out))

        print('Run {}/{}: {:.3f} (LDA), {:.3f} (SVM) (left out: {})'.format(
            i+1, len(subj_ids), acc_lda, acc_svm, left_out))
    except IndexError:
        print('Run {}/{}: Not enough data for {}, will continue.'.format(
            i + 1, len(subj_ids), left_out))

print('Mean over all runs:\n LDA: {} (std: {})\n SVM: {} (std: {})'.format(
    np.mean(acc_lda_all_runs), np.std(acc_lda_all_runs),
    np.mean(acc_svm_all_runs), np.std(acc_svm_all_runs)
))


# RESULT DOC

dir_out_results = '/Users/jannes/Projects/delir/results/acc'
fname_results = 'from_eeg.csv'
path_out_results = '{}/{}'.format(dir_out_results, fname_results)

params = {
    'study': study,
    'region': region,
    'freq_band': freq_band,
    'lcut': lcut,
    'hcut': hcut,
    'test_size': test_size,
    'n_leave_out': n_leave_out,
    'win_length': win_length,
    'bis_crit': bis_crit,
    'drop_perc': drop_perc,
    'drop_from': drop_from,
    'n_subj': len(subj_ids),
    'subj_ids': subj_ids,
    'n_runs': len(x),
    'run_order': x
}

lda_params = {
    'classifier': 'LDA',
    'solver': solver,
    'shrink': shrink,
    'kernel': '',
    'run_accs': acc_lda_all_runs,
    'mean_acc': np.mean(acc_lda_all_runs),
    'std_acc': np.std(acc_lda_all_runs)
}

svm_params = {
    'classifier': 'SVM',
    'solver': '',
    'shrink': '',
    'kernel': kernel,
    'run_accs': acc_svm_all_runs,
    'mean_acc': np.mean(acc_svm_all_runs),
    'std_acc': np.std(acc_svm_all_runs)
}

d_lda, d_svm = params.copy(), params.copy()
d_lda.update(lda_params)
d_svm.update(svm_params)

df_lda = pd.DataFrame.from_records([d_lda])
df_svm = pd.DataFrame.from_records([d_svm])

with open(path_out_results, 'a') as f:
    df_lda.to_csv(f, header=False)
    df_svm.to_csv(f, header=False)


# # FIGURES
#
# path_out_fig = '{}/{}_subset.png'.format(dir_signal, freq_band)
# plot.plot_accuracies(x, acc_all_runs, lcut=lcut, hcut=hcut, path_out=path_out_fig, bis_crit=bis_crit, drop_perc=drop_perc, drop_from=drop_from, show=False)
# path_out_fig = '{}/{}_subset.svg'.format(dir_signal, freq_band)
# plot.plot_accuracies(x, acc_all_runs, lcut=lcut, hcut=hcut, path_out=path_out_fig, bis_crit=bis_crit, drop_perc=drop_perc, drop_from=drop_from, show=False)

